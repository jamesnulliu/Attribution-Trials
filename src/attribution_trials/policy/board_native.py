"""Board-native (non-LLM) chess backbone for the static and recurrent families.

A move head and a think-time head, both conditioned on the latent as a
*hidden* vector (concatenated to the board encoding; no prompt).

* **Input** is the board itself (FEN -> 12x64 piece planes + side-to-move),
  not engine values, so real Lichess positions are scored with no engine
  dependency (the ingested trajectories carry ``fen_before`` +
  ``legal_actions``).
* The move head is **factored from/to**: ``logit(move) = s_from[from_sq] +
  s_to[to_sq]``, softmaxed over the position's *legal* moves (masked).
* Batching is **variable-length + masked**: trajectories are right-padded on
  time and on the legal-move axis, with a ``step_mask`` / ``action_mask`` so
  padding contributes nothing to the loss or its gradient.
* ``trunk="mlp"`` (default) or ``"conv"`` (spatial 12x8x8);
  ``timing_model="lognormal"`` (default) or ``"zi_lognormal"``.

torch is imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass

from attribution_trials.interface import DecisionPoint
from attribution_trials.latent.base import Injection, InjectionKind
from attribution_trials.latent.structured import DIMENSIONS, history_features
from attribution_trials.policy.base import PolicyBackbone
from attribution_trials.prediction import MoveDistribution, Prediction, TimingPrediction
from attribution_trials.train.base import Trajectory

#: 12 piece-type planes x 64 squares, plus one side-to-move scalar.
BOARD_DIM = 12 * 64 + 1

#: Piece char -> plane index (white PNBRQK then black pnbrqk).
_PIECE_TO_PLANE = {c: i for i, c in enumerate("PNBRQKpnbrqk")}

#: Masked-out logit fill (large finite negative; every step keeps >=1 valid
#: action so log-softmax is finite, and a finite fill keeps backward clean).
_NEG = -1e30


def _square_index(sq: str) -> int:
    """``"e2"`` -> 12 (``rank*8 + file``, ``a1`` == 0). Defensive on junk."""
    if len(sq) < 2:
        return 0
    file = ord(sq[0]) - 97  # 'a' -> 0
    rank = ord(sq[1]) - 49  # '1' -> 0
    if not (0 <= file < 8 and 0 <= rank < 8):
        return 0
    return rank * 8 + file


def board_planes(fen: str) -> list[float]:
    """FEN -> ``BOARD_DIM`` floats: 12 piece planes (a1=0) + side-to-move.

    Pure stdlib (parses only the placement + active-colour FEN fields), so the
    encoder has no python-chess dependency -- the heavy import stayed in the
    parser; everything downstream of :class:`GameRecord` is light.
    """
    planes = [0.0] * BOARD_DIM
    parts = fen.split()
    placement = parts[0] if parts else ""
    for r_fen, row in enumerate(placement.split("/")):
        rank_index = 7 - r_fen  # FEN lists rank 8 first; a1 is rank_index 0
        file = 0
        for ch in row:
            if ch.isdigit():
                file += int(ch)
                continue
            pidx = _PIECE_TO_PLANE.get(ch)
            if pidx is not None and 0 <= file < 8:
                planes[pidx * 64 + rank_index * 8 + file] = 1.0
            file += 1
    side = parts[1] if len(parts) > 1 else "w"
    planes[BOARD_DIM - 1] = 1.0 if side == "w" else 0.0
    return planes


@dataclass
class BoardBatch:
    """A batch of (variable-length) trajectories as padded tensors ``[T,B,*]``.

    Time-major for the recurrence. Padding is masked everywhere it matters:
    ``step_mask`` marks real decision steps, ``action_mask`` marks real legal
    moves within the padded legal axis ``A``.
    """

    feats: object  # FloatTensor [T, B, n_features]  (injector input)
    board: object  # FloatTensor [T, B, BOARD_DIM]
    legal_from: object  # LongTensor  [T, B, A]  (from-square of each action)
    legal_to: object  # LongTensor  [T, B, A]  (to-square of each action)
    action_mask: object  # BoolTensor  [T, B, A]  (True = real legal move)
    move_idx: object  # LongTensor  [T, B]     (index of the played move)
    times: object  # FloatTensor [T, B]     (observed think-time, seconds)
    step_mask: object  # BoolTensor  [T, B]     (True = real decision step)
    lengths: object  # LongTensor  [B]        (real decisions per player)
    n_steps: int
    n_traj: int
    # Player id per batch column (column order), so identity-keyed injectors
    # (static-individual / per-player embedding) can look up their vector.
    player_ids: tuple[str, ...] = ()

    def to(self, device) -> BoardBatch:
        f = lambda x: x.to(device)  # noqa: E731 - terse tensor mover
        return BoardBatch(
            feats=f(self.feats),
            board=f(self.board),
            legal_from=f(self.legal_from),
            legal_to=f(self.legal_to),
            action_mask=f(self.action_mask),
            move_idx=f(self.move_idx),
            times=f(self.times),
            step_mask=f(self.step_mask),
            lengths=f(self.lengths),
            n_steps=self.n_steps,
            n_traj=self.n_traj,
            player_ids=self.player_ids,
        )


class BoardNativeBackbone(PolicyBackbone):
    """Factored-move board policy with latent conditioning (hidden only)."""

    accepts = (InjectionKind.HIDDEN,)

    def __init__(
        self,
        checkpoint: str | None = None,
        *,
        latent_dim: int = 4,
        hidden_dim: int = 64,
        condition: str = "concat",  # only "concat" is implemented
        seed: int = 0,
        timing_model: str = "lognormal",  # "lognormal" | "zi_lognormal"
        trunk: str = "mlp",  # "mlp" (default) | "conv" (spatial 12x8x8)
        conv_channels: int = 32,
    ) -> None:
        self.checkpoint = checkpoint
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.condition = condition
        self.seed = seed
        self.timing_model = timing_model
        self.trunk = trunk
        self.conv_channels = conv_channels
        self._net = None

    # --- network -------------------------------------------------------
    def _build(self):
        if self._net is not None:
            return self._net
        try:
            import torch
            from torch import nn
        except ImportError as e:  # pragma: no cover - env-dependent
            raise ImportError(
                "torch required for BoardNativeBackbone; install the 'train' "
                "extra: pip install '.[train]'."
            ) from e

        kind = self.trunk
        chans = self.conv_channels

        class _Net(nn.Module):
            def __init__(self, board_dim, latent_dim, hidden):
                super().__init__()
                self.kind = kind
                if kind == "conv":
                    # 12 piece planes as 8x8 channels -> two 3x3 convs (the
                    # spatial structure an MLP over the flat 769 floats cannot
                    # see), then fuse the side-to-move scalar + latent.
                    self.conv = nn.Sequential(
                        nn.Conv2d(12, chans, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(chans, chans, 3, padding=1),
                        nn.ReLU(),
                    )
                    self.fuse = nn.Sequential(
                        nn.Linear(chans * 64 + 1 + latent_dim, hidden),
                        nn.ReLU(),
                    )
                else:
                    self.trunk = nn.Sequential(
                        nn.Linear(board_dim + latent_dim, hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                    )
                self.from_head = nn.Linear(hidden, 64)
                self.to_head = nn.Linear(hidden, 64)
                # Think-time log-mean from the latent (latent drives timing);
                # log-std is a free scalar.
                self.mu = nn.Linear(latent_dim, 1)
                self.log_sigma = nn.Parameter(torch.zeros(1))
                # Zero-inflation logit (P(think-time == 0), e.g. premoves) for
                # the discrete/zero-inflated head -- also latent-driven.
                self.zero_logit = nn.Linear(latent_dim, 1)

            def forward(self, board, latent):
                if self.kind == "conv":
                    lead = board.shape[:-1]
                    planes = board[..., :768].reshape(-1, 12, 8, 8)
                    stm = board[..., 768:769].reshape(-1, 1)
                    c = self.conv(planes).flatten(1)  # [N, chans*64]
                    lat = latent.reshape(-1, latent.shape[-1])
                    h = self.fuse(torch.cat([c, stm, lat], dim=-1))
                    h = h.reshape(*lead, -1)
                else:
                    h = self.trunk(torch.cat([board, latent], dim=-1))
                return self.from_head(h), self.to_head(h)

        # Seed init from a forked RNG (like NeuralInjector) so the network is
        # reproducible and independent of global RNG state: every condition
        # trained with the same seed starts from identical backbone weights.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self._net = _Net(BOARD_DIM, self.latent_dim, self.hidden_dim)
        return self._net

    def parameters(self):
        return self._build().parameters()

    def to(self, device):
        self._build().to(device)
        return self

    # --- encoding (raw trajectories -> tensors) ------------------------
    def encode_batch(self, trajectories: list[Trajectory]) -> BoardBatch:
        import torch

        B = len(trajectories)
        lengths = [len(t.decisions) for t in trajectories]
        T = max(lengths) if lengths else 0
        A = 1
        for t in trajectories:
            for dp in t.decisions:
                A = max(A, len(dp.legal_actions))

        cols = {k: [] for k in ("f", "bd", "lf", "lt", "am", "mi", "tm", "sm")}
        for t in trajectories:
            sub = {k: [] for k in cols}
            for i in range(T):
                if i < len(t.decisions):
                    dp, obs = t.decisions[i], t.observations[i]
                    h = history_features(dp)
                    la = dp.legal_actions
                    k = len(la)
                    pad = A - k
                    sub["f"].append([h[d] for d in DIMENSIONS])
                    sub["bd"].append(board_planes(str(dp.state)))
                    sub["lf"].append([_square_index(m[:2]) for m in la] + [0] * pad)
                    sub["lt"].append([_square_index(m[2:4]) for m in la] + [0] * pad)
                    sub["am"].append([True] * k + [False] * pad)
                    sub["mi"].append(la.index(obs.move) if obs.move in la else 0)
                    # Preserve real 0s (premoves) -- the log-normal clamps
                    # them anyway, but the zero-inflated head needs the mass.
                    spent = obs.time_spent
                    sub["tm"].append(float(spent) if spent is not None else 1e-3)
                    sub["sm"].append(True)
                else:  # right-pad: one dummy valid action so log-softmax is OK
                    sub["f"].append([0.0] * len(DIMENSIONS))
                    sub["bd"].append([0.0] * BOARD_DIM)
                    sub["lf"].append([0] * A)
                    sub["lt"].append([0] * A)
                    sub["am"].append([True] + [False] * (A - 1))
                    sub["mi"].append(0)
                    sub["tm"].append(1.0)
                    sub["sm"].append(False)
            for k in cols:
                cols[k].append(sub[k])

        def tb(x, dtype):  # [B, T, ...] python -> [T, B, ...] tensor
            return torch.tensor(x, dtype=dtype).transpose(0, 1).contiguous()

        return BoardBatch(
            feats=tb(cols["f"], torch.float32),
            board=tb(cols["bd"], torch.float32),
            legal_from=tb(cols["lf"], torch.long),
            legal_to=tb(cols["lt"], torch.long),
            action_mask=tb(cols["am"], torch.bool),
            move_idx=tb(cols["mi"], torch.long),
            times=tb(cols["tm"], torch.float32),
            step_mask=tb(cols["sm"], torch.bool),
            lengths=torch.tensor(lengths, dtype=torch.long),
            n_steps=T,
            n_traj=B,
            player_ids=tuple(t.player_id for t in trajectories),
        )

    # --- loss + eval ---------------------------------------------------
    def move_logp(self, latent_seq, batch: BoardBatch):
        """Full per-step log-prob over the legal actions: ``[T, B, A]``.

        Illegal/pad slots get a large (finite) negative before the softmax, so
        they carry ~zero probability.
        """
        import torch

        net = self._build()
        from_l, to_l = net(batch.board, latent_seq)  # [T,B,64] each
        logits = from_l.gather(-1, batch.legal_from) + to_l.gather(-1, batch.legal_to)  # [T,B,A]
        logits = logits.masked_fill(~batch.action_mask, _NEG)
        return torch.log_softmax(logits, dim=-1)

    def timing_mu_sigma(self, latent_seq):
        """``(mu, sigma)`` of the log-normal think-time head for a latent."""
        import torch

        net = self._build()
        mu = net.mu(latent_seq).squeeze(-1)
        sigma = torch.nn.functional.softplus(net.log_sigma) + 1e-3
        return mu, sigma

    def _timing_nll_steps(self, latent_seq, batch: BoardBatch):
        """Per-step think-time NLL ``[T, B]`` for the configured timing model.

        ``lognormal`` (default): a continuous log-normal on seconds -- simple
        but mis-specified for Lichess's 1s-quantized, zero-inflated clocks.
        ``zi_lognormal``: a **zero-inflated** log-normal -- a learned mass
        ``pi`` on a 0s premove (``time < 0.5``) and a log-normal on the rest::

            t == 0:  -log pi
            t  > 0:  -log(1 - pi) + lognormal_nll(t)

        Both ``mu`` / ``pi`` are functions of the (evolving) latent, so the
        state drives *whether* a move is a premove and *how long* the rest are.
        """
        import torch

        net = self._build()
        mu, sigma = self.timing_mu_sigma(latent_seq)
        logt = torch.log(batch.times.clamp_min(1e-3))
        z = (logt - mu) / sigma
        pos_nll = 0.5 * z * z + torch.log(sigma) + 0.5 * 1.8378771 + logt
        if self.timing_model != "zi_lognormal":
            return pos_nll
        pi = torch.sigmoid(net.zero_logit(latent_seq).squeeze(-1))
        pi = pi.clamp(1e-6, 1 - 1e-6)
        is_zero = batch.times < 0.5
        return torch.where(is_zero, -torch.log(pi), -torch.log1p(-pi) + pos_nll)

    def _logp_chosen(self, latent_seq, batch: BoardBatch):
        """Log-prob of the played move at every step: ``[T, B]`` (masked-safe).

        Returns ``(chosen_logp, mu, sigma)``.
        """
        logp = self.move_logp(latent_seq, batch)
        chosen = logp.gather(-1, batch.move_idx.unsqueeze(-1)).squeeze(-1)
        mu, sigma = self.timing_mu_sigma(latent_seq)
        return chosen, mu, sigma

    def trajectory_loss(self, latent_seq, batch: BoardBatch, lam: float, step_mask=None):
        """Masked move-NLL + ``lam`` * timing-NLL over the valid steps.

        ``step_mask`` (``[T,B]`` bool) overrides ``batch.step_mask`` -- the
        trainer passes the *train* mask while fitting and the held-out *eval*
        mask while scoring, both built from the per-user temporal split.
        """
        chosen, _, _ = self._logp_chosen(latent_seq, batch)
        mask = batch.step_mask if step_mask is None else step_mask
        maskf = mask.to(chosen.dtype)
        denom = maskf.sum().clamp_min(1.0)
        move_nll = -(chosen * maskf).sum() / denom

        tnll = self._timing_nll_steps(latent_seq, batch)
        timing_nll = (tnll * maskf).sum() / denom

        return {
            "loss": move_nll + lam * timing_nll,
            "move_nll": move_nll,
            "timing_nll": timing_nll,
        }

    def per_traj_move_nll(self, latent_seq, batch: BoardBatch, step_mask=None):
        """Per-player mean move-NLL over the valid window: a ``[B]`` tensor.

        The bootstrap unit is the player, so this returns one number per
        trajectory (averaged over that player's scored steps only).
        """
        chosen, _, _ = self._logp_chosen(latent_seq, batch)
        mask = batch.step_mask if step_mask is None else step_mask
        maskf = mask.to(chosen.dtype)
        denom = maskf.sum(dim=0).clamp_min(1.0)
        return -(chosen * maskf).sum(dim=0) / denom

    def per_traj_timing_nll(self, latent_seq, batch: BoardBatch, step_mask=None):
        """Per-player mean think-time NLL over the window: a ``[B]`` tensor.

        Uses the configured timing model (``timing_model``): a log-normal, or
        the zero-inflated log-normal that respects Lichess's 1s-quantized,
        zero-inflated clocks.
        """
        tnll = self._timing_nll_steps(latent_seq, batch)
        mask = batch.step_mask if step_mask is None else step_mask
        maskf = mask.to(tnll.dtype)
        denom = maskf.sum(dim=0).clamp_min(1.0)
        return (tnll * maskf).sum(dim=0) / denom

    # --- temporal-split helpers ---------------------------------------
    @staticmethod
    def split_indices(trajectories: list[Trajectory], train_frac: float = 0.7) -> list[int]:
        """Per-player boundary step: ``[0, b)`` train, ``[b, len)`` held out.

        A move-fraction split. The pipeline uses the session-aware
        :func:`~attribution_trials.experiments.ec.session_split_indices`, which
        falls back to this boundary for users with a single session.
        """
        out = []
        for t in trajectories:
            n = len(t.decisions)
            if n < 2:
                out.append(n)  # nothing to hold out
            else:
                out.append(min(n - 1, max(1, round(train_frac * n))))
        return out

    def train_eval_masks(self, batch: BoardBatch, splits: list[int]):
        """``(train_mask, eval_mask)`` ``[T,B]`` from per-player boundaries."""
        import torch

        device = batch.step_mask.device
        T, _ = batch.step_mask.shape
        tgrid = torch.arange(T, device=device).unsqueeze(1)  # [T,1]
        sp = torch.tensor(splits, device=device).unsqueeze(0)  # [1,B]
        train = batch.step_mask & (tgrid < sp)
        held = batch.step_mask & (tgrid >= sp)
        return train, held

    # --- single-step prediction ---------------------------------------
    def predict(self, dp: DecisionPoint, injection: Injection | None = None) -> Prediction:
        import torch

        net = self._build()
        if injection is None or injection.vector is None:
            latent = torch.zeros(self.latent_dim)
        else:
            latent = torch.tensor(injection.vector, dtype=torch.float32)
        board = torch.tensor(board_planes(str(dp.state)), dtype=torch.float32)
        la = dp.legal_actions
        lf = torch.tensor([_square_index(m[:2]) for m in la])
        lt = torch.tensor([_square_index(m[2:4]) for m in la])
        with torch.no_grad():
            from_l, to_l = net(board, latent)
            logits = from_l[lf] + to_l[lt]
            probs = torch.softmax(logits, dim=-1)
            mu = net.mu(latent).squeeze(-1)
            sigma = torch.nn.functional.softplus(net.log_sigma) + 1e-3
        moves = {m: float(probs[i]) for i, m in enumerate(la)}
        return Prediction(
            moves=MoveDistribution(probs=moves),
            timing=TimingPrediction(mu=float(mu), sigma=float(sigma)),
            latent=injection.vector if injection else None,
        )

    @property
    def name(self) -> str:
        return f"BoardNativeBackbone({self.checkpoint or 'scratch'})"
