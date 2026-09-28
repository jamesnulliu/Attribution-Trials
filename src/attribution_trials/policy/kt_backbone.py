"""Knowledge-tracing backbone for the static and recurrent latent families.

A student answering items uses the same ``DecisionPoint`` schema as a chess
player making moves: ``state`` = item features, ``legal_actions`` =
``correct``/``incorrect``, ``recent_outcomes`` = the student's prior items,
``time_signal`` = response time. The injectors and the trainer are shared
with chess; only the encoder and head differ.

The backbone predicts P(correct) from item features + the latent with a
small MLP head, and response time with a latent-driven log-normal. It
implements the protocol the trainer and per-user scoring need
(``encode_batch`` -> batch with ``feats``/``player_ids``,
``train_eval_masks``, ``trajectory_loss``, ``per_traj_move_nll`` /
``per_traj_timing_nll``). torch is imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass

from attribution_trials.interface import DecisionPoint
from attribution_trials.latent.base import Injection, InjectionKind
from attribution_trials.latent.structured import DIMENSIONS, history_features
from attribution_trials.policy.base import PolicyBackbone
from attribution_trials.prediction import MoveDistribution, Prediction, TimingPrediction
from attribution_trials.train.base import Trajectory


@dataclass
class KTBatch:
    """A batch of student item-response trajectories as padded tensors."""

    feats: object  # FloatTensor [T, B, n_features]  (injector input)
    item: object  # FloatTensor [T, B, item_dim]    (item features)
    resp: object  # LongTensor  [T, B]   (0 = correct, 1 = incorrect)
    times: object  # FloatTensor [T, B]  (response time, seconds)
    step_mask: object  # BoolTensor  [T, B]
    lengths: object  # LongTensor  [B]
    n_steps: int
    n_traj: int
    player_ids: tuple[str, ...] = ()

    def to(self, device) -> KTBatch:
        f = lambda x: x.to(device)  # noqa: E731
        return KTBatch(
            feats=f(self.feats),
            item=f(self.item),
            resp=f(self.resp),
            times=f(self.times),
            step_mask=f(self.step_mask),
            lengths=f(self.lengths),
            n_steps=self.n_steps,
            n_traj=self.n_traj,
            player_ids=self.player_ids,
        )


class KTBackbone(PolicyBackbone):
    """Logistic correct/incorrect head over item features + latent."""

    accepts = (InjectionKind.HIDDEN,)

    def __init__(
        self,
        *,
        latent_dim: int = 16,
        item_dim: int = 1,
        hidden_dim: int = 32,
        seed: int = 0,
    ) -> None:
        self.latent_dim = latent_dim
        self.item_dim = item_dim
        self.hidden_dim = hidden_dim
        self.seed = seed
        self._net = None

    def _build(self):
        if self._net is not None:
            return self._net
        try:
            import torch
            from torch import nn
        except ImportError as e:  # pragma: no cover - env-dependent
            raise ImportError("torch required for KTBackbone; install the 'train' extra.") from e

        class _Net(nn.Module):
            def __init__(self, item_dim, latent_dim, hidden):
                super().__init__()
                self.trunk = nn.Sequential(
                    nn.Linear(item_dim + latent_dim, hidden),
                    nn.ReLU(),
                )
                self.correct = nn.Linear(hidden, 1)  # logit P(correct)
                self.mu = nn.Linear(latent_dim, 1)  # response-time log-mean
                self.log_sigma = nn.Parameter(torch.zeros(1))

            def forward(self, item, latent):
                h = self.trunk(torch.cat([item, latent], dim=-1))
                return self.correct(h).squeeze(-1)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self._net = _Net(self.item_dim, self.latent_dim, self.hidden_dim)
        return self._net

    def parameters(self):
        return self._build().parameters()

    def to(self, device):
        self._build().to(device)
        return self

    # --- encoding ------------------------------------------------------
    def encode_batch(self, trajectories: list[Trajectory]) -> KTBatch:
        import torch

        B = len(trajectories)
        lengths = [len(t.decisions) for t in trajectories]
        T = max(lengths) if lengths else 0
        cols = {k: [] for k in ("f", "it", "rp", "tm", "sm")}
        for t in trajectories:
            sub = {k: [] for k in cols}
            for i in range(T):
                if i < len(t.decisions):
                    dp, obs = t.decisions[i], t.observations[i]
                    h = history_features(dp)
                    sub["f"].append([h[d] for d in DIMENSIONS])
                    sub["it"].append([float(x) for x in dp.state])
                    sub["rp"].append(0 if obs.move == "correct" else 1)
                    sp = obs.time_spent
                    sub["tm"].append(float(sp) if sp is not None else 1e-3)
                    sub["sm"].append(True)
                else:
                    sub["f"].append([0.0] * len(DIMENSIONS))
                    sub["it"].append([0.0] * self.item_dim)
                    sub["rp"].append(0)
                    sub["tm"].append(1e-3)
                    sub["sm"].append(False)
            for k in cols:
                cols[k].append(sub[k])

        def tb(x, dtype):
            return torch.tensor(x, dtype=dtype).transpose(0, 1).contiguous()

        return KTBatch(
            feats=tb(cols["f"], torch.float32),
            item=tb(cols["it"], torch.float32),
            resp=tb(cols["rp"], torch.long),
            times=tb(cols["tm"], torch.float32),
            step_mask=tb(cols["sm"], torch.bool),
            lengths=torch.tensor(lengths, dtype=torch.long),
            n_steps=T,
            n_traj=B,
            player_ids=tuple(t.player_id for t in trajectories),
        )

    def train_eval_masks(self, batch: KTBatch, splits: list[int]):
        import torch

        device = batch.step_mask.device
        T, _ = batch.step_mask.shape
        tgrid = torch.arange(T, device=device).unsqueeze(1)
        sp = torch.tensor(splits, device=device).unsqueeze(0)
        train = batch.step_mask & (tgrid < sp)
        held = batch.step_mask & (tgrid >= sp)
        return train, held

    # --- loss + eval ---------------------------------------------------
    def _chosen_logp(self, latent_seq, batch: KTBatch):
        import torch

        net = self._build()
        logit_c = net(batch.item, latent_seq)  # [T,B]
        sp = torch.nn.functional.softplus
        logp_correct = -sp(-logit_c)  # log sigmoid(x)
        logp_incorrect = -sp(logit_c)  # log(1 - sigmoid(x))
        return torch.where(batch.resp == 0, logp_correct, logp_incorrect)

    def timing_mu_sigma(self, latent_seq):
        import torch

        net = self._build()
        mu = net.mu(latent_seq).squeeze(-1)
        sigma = torch.nn.functional.softplus(net.log_sigma) + 1e-3
        return mu, sigma

    def _timing_nll_steps(self, latent_seq, batch: KTBatch):
        import torch

        mu, sigma = self.timing_mu_sigma(latent_seq)
        logt = torch.log(batch.times.clamp_min(1e-3))
        z = (logt - mu) / sigma
        return 0.5 * z * z + torch.log(sigma) + 0.5 * 1.8378771 + logt

    def trajectory_loss(self, latent_seq, batch: KTBatch, lam, step_mask=None):
        chosen = self._chosen_logp(latent_seq, batch)
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

    def per_traj_move_nll(self, latent_seq, batch: KTBatch, step_mask=None):
        chosen = self._chosen_logp(latent_seq, batch)
        mask = batch.step_mask if step_mask is None else step_mask
        maskf = mask.to(chosen.dtype)
        denom = maskf.sum(dim=0).clamp_min(1.0)
        return -(chosen * maskf).sum(dim=0) / denom

    def per_traj_timing_nll(self, latent_seq, batch: KTBatch, step_mask=None):
        tnll = self._timing_nll_steps(latent_seq, batch)
        mask = batch.step_mask if step_mask is None else step_mask
        maskf = mask.to(tnll.dtype)
        denom = maskf.sum(dim=0).clamp_min(1.0)
        return (tnll * maskf).sum(dim=0) / denom

    def predict(self, dp: DecisionPoint, injection: Injection | None = None) -> Prediction:
        import torch

        net = self._build()
        latent = (
            torch.zeros(self.latent_dim)
            if injection is None or injection.vector is None
            else torch.tensor(injection.vector, dtype=torch.float32)
        )
        item = torch.tensor([float(x) for x in dp.state], dtype=torch.float32)
        with torch.no_grad():
            logit = net(item, latent)
            pc = float(torch.sigmoid(logit))
            mu, sigma = self.timing_mu_sigma(latent)
        return Prediction(
            moves=MoveDistribution(probs={"correct": pc, "incorrect": 1.0 - pc}),
            timing=TimingPrediction(mu=float(mu), sigma=float(sigma)),
        )

    @property
    def name(self) -> str:
        return f"KTBackbone(item_dim={self.item_dim})"
