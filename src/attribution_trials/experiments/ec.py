"""Per-user train/held-out split, arm training, and per-user held-out NLL.

Every latent-family condition is trained on each user's earlier sessions and
scored on the same user's later sessions. The unit of every bootstrap and
sign-flip test is the user, so scoring returns one held-out NLL per user.
"""

from __future__ import annotations

from attribution_trials.policy.board_native import BoardNativeBackbone
from attribution_trials.train.base import Trajectory
from attribution_trials.train.sft import EvalSpec, SFTTrainer


def _session_split_index(traj: Trajectory, train_frac: float) -> int:
    """Per-user boundary at a **session** edge: hold out the later sessions.

    Sessions are recovered from the persisted decisions: the game index is
    ``len(recent_outcomes.recent)`` (completed games so far) and the first game
    of each session has ``session_position == 0``, so the set of game indices
    with ``session_position == 0`` are the session starts. Falls back to a
    move-fraction split when a user has <2 sessions.
    """
    n = len(traj.decisions)
    if n < 2:
        return n
    game_idx = [len(d.recent_outcomes.recent) for d in traj.decisions]
    starts = sorted(
        {g for g, d in zip(game_idx, traj.decisions) if d.recent_outcomes.session_position == 0}
    )
    n_sess = len(starts)
    if n_sess < 2:  # one sitting -> no session split possible
        return min(n - 1, max(1, round(train_frac * n)))
    n_train = max(1, min(n_sess - 1, round(train_frac * n_sess)))
    boundary_game = starts[n_train]  # first game of the first held-out session
    for i, g in enumerate(game_idx):
        if g >= boundary_game:
            return max(1, min(n - 1, i))
    return n


def session_split_indices(trajectories: list[Trajectory], train_frac: float = 0.7) -> list[int]:
    """Per-user session-aware split boundaries (hold out later sessions)."""
    return [_session_split_index(t, train_frac) for t in trajectories]


def _per_player_nlls(injector, backbone, full, splits, batch_size=16):
    """Held-out per-user (move-NLL, timing-NLL) lists from a trained arm.

    Warms the latent over each user's whole trajectory (no grad) and scores
    only their held-out tail (``splits`` -> eval mask) -- one number per
    user, the unit the bootstrap resamples. Minibatched; results are returned
    in the original ``full`` user order.
    """
    import torch

    device = next(backbone.parameters()).device
    backbone._build().eval()
    injector._build().eval()
    n = len(full.trajectories)
    move = [0.0] * n
    timing = [0.0] * n
    # Length-sort for tight padding, but remember the original index so the
    # returned lists line up with full.trajectories (and thus the splits).
    indexed = sorted(
        enumerate(zip(full.trajectories, splits)),
        key=lambda kv: len(kv[1][0].decisions),
        reverse=True,
    )
    for i in range(0, len(indexed), batch_size):
        chunk = indexed[i : i + batch_size]
        idxs = [k for k, _ in chunk]
        trajs = [t for _, (t, _) in chunk]
        sp = [s for _, (_, s) in chunk]
        batch = backbone.encode_batch(trajs).to(device)
        _, eval_mask = backbone.train_eval_masks(batch, sp)
        with torch.no_grad():
            latent = injector.latent_trajectory(batch.feats, player_ids=batch.player_ids)
            mv = backbone.per_traj_move_nll(latent, batch, step_mask=eval_mask)
            tm = backbone.per_traj_timing_nll(latent, batch, step_mask=eval_mask)
        for j, idx in enumerate(idxs):
            move[idx] = float(mv[j].cpu())
            timing[idx] = float(tm[j].cpu())
    return move, timing


def _train_arm(injector, latent_dim, hidden_dim, full, splits, cfg):
    # Every arm uses the same seed -> identical backbone init, so a held-out
    # gap is the injector difference, not initialization luck (each injector
    # is likewise seeded with cfg.seed via its own forked RNG).
    backbone = BoardNativeBackbone(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        seed=cfg.seed,
        timing_model=cfg.extra.get("timing_model", "lognormal"),
        trunk=cfg.extra.get("trunk", "mlp"),
    )
    eval_spec = EvalSpec(dataset=full, splits=splits)
    trainer = SFTTrainer(injector, backbone, cfg)
    summary = trainer.fit(full, eval_spec=eval_spec)
    move_pp, timing_pp = _per_player_nlls(
        injector, backbone, full, splits, batch_size=cfg.batch_size
    )
    return injector, backbone, summary, move_pp, timing_pp
