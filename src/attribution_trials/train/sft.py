"""Supervised trainer for the static and recurrent latent families.

Maximises the likelihood of each user's observed actions (and response or
think times) w.r.t. the injector and backbone parameters. The latent is
teacher-forced along each trajectory and the loss is::

    sum_t  -log P(action_t | state_t, user, z_t)
           - lambda * log p(time_t | state_t, user, z_t)

The loop is generic over two protocols:

* the **injector** exposes ``latent_trajectory(feats_seq, player_ids)`` ->
  ``[T, B, L]`` (see :mod:`attribution_trials.latent`);
* the **backbone** exposes ``encode_batch``, ``train_eval_masks``,
  ``trajectory_loss`` and ``parameters()`` (see
  :mod:`attribution_trials.policy`).

torch is imported lazily so the module loads without it.
"""

from __future__ import annotations

from dataclasses import dataclass

from attribution_trials.train.base import Trainer, TrajectoryDataset


@dataclass
class EvalSpec:
    """Per-user held-out split for training and validation.

    ``splits`` holds one boundary index per trajectory of ``dataset``. The
    trainer builds train/eval **masks** from it
    (``backbone.train_eval_masks``), so training never sees a user's
    held-out tail and validation scores only that tail.
    """

    dataset: TrajectoryDataset
    splits: list[int]


class SFTTrainer(Trainer):
    """Maximum-likelihood imitation of real trajectories."""

    def fit(self, dataset: TrajectoryDataset, eval_spec: EvalSpec) -> dict:
        try:
            import torch
        except ImportError as e:  # pragma: no cover - env-dependent
            raise ImportError(
                "torch required for SFTTrainer; install the 'train' extra: pip install '.[train]'"
            ) from e

        handle, wandb_run = self.begin_run(dataset)

        params = list(self._trainable_parameters(torch))
        try:
            summary = self._train(torch, dataset, eval_spec, handle, wandb_run, params)
            wandb_run.summary(summary)
            wandb_run.finish(status="completed")
            handle.finalize(status="completed")
            return summary
        except Exception as e:  # noqa: BLE001 - re-raised after recording
            wandb_run.finish(status="failed")
            handle.finalize(status="failed", error=repr(e))
            raise

    def _train(self, torch, dataset, eval_spec, handle, wandb_run, params):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Place both modules on the device (params are moved in place, so the
        # optimizer below still references the right tensors).
        self.injector.to(device)
        self.backbone.to(device)

        lam = float(self.config.extra.get("timing_lambda", 0.5))
        opt = torch.optim.AdamW(params, lr=self.config.lr)
        return self._train_masked_minibatched(
            torch, dataset, eval_spec, opt, lam, handle, wandb_run, params
        )

    def _train_masked_minibatched(
        self, torch, dataset, eval_spec, opt, lam, handle, wandb_run, params
    ):
        """Masked per-user split, minibatched over users.

        Users are length-sorted and chunked into ``config.batch_size`` groups
        that are **encoded once** (so each minibatch pads only to *its* longest
        trajectory). Each epoch shuffles the minibatch visit order (seeded) and
        takes one optimizer step per minibatch. Every condition trained on the
        same data runs the identical schedule, so comparisons stay paired.
        """
        import random

        device = next(self.backbone.parameters()).device
        mbs = self._encode_masked_minibatches(dataset.trajectories, eval_spec.splits, device)
        rng = random.Random(self.config.seed)

        final: dict = {}
        for epoch in range(self.config.epochs):
            self._set_train(True)
            order = list(range(len(mbs)))
            rng.shuffle(order)
            loss_w = move_w = time_w = tr_steps = 0.0
            for i in order:
                mb = mbs[i]
                latent = self._latent(mb["batch"])
                out = self.backbone.trajectory_loss(
                    latent, mb["batch"], lam, step_mask=mb["train_mask"]
                )
                opt.zero_grad()
                out["loss"].backward()
                torch.nn.utils.clip_grad_norm_(params, self.config.grad_clip)
                opt.step()
                w = mb["n_train_steps"]
                loss_w += out["loss"].item() * w
                move_w += out["move_nll"].item() * w
                time_w += out["timing_nll"].item() * w
                tr_steps += w

            self._set_train(False)
            ev_w = ev_steps = 0.0
            with torch.no_grad():
                for mb in mbs:
                    latent = self._latent(mb["batch"])
                    o = self.backbone.trajectory_loss(
                        latent, mb["batch"], lam=0.0, step_mask=mb["eval_mask"]
                    )
                    ev_w += o["move_nll"].item() * mb["n_eval_steps"]
                    ev_steps += mb["n_eval_steps"]

            tr = max(tr_steps, 1.0)
            metrics = {
                "loss": loss_w / tr,
                "move_nll": move_w / tr,
                "timing_nll": time_w / tr,
                "val_move_nll": ev_w / max(ev_steps, 1.0),
            }
            wandb_run.log(metrics, step=epoch)
            final = metrics

        summary = {
            "status": "completed",
            "epochs": self.config.epochs,
            "n_minibatches": len(mbs),
            **final,
        }
        summary.update(self._param_report())
        self._save_checkpoint(torch, handle)
        return summary

    def _encode_masked_minibatches(self, trajectories, splits, device):
        """Length-sorted, pre-encoded minibatches of (batch, train/eval masks).

        Length-sorting keeps each minibatch's padding tight; encoding once and
        caching avoids re-tensorizing FENs every epoch.
        """
        bs = max(1, self.config.batch_size)
        pairs = sorted(
            zip(trajectories, splits),
            key=lambda ts: len(ts[0].decisions),
            reverse=True,
        )
        mbs = []
        for i in range(0, len(pairs), bs):
            chunk = pairs[i : i + bs]
            trajs = [t for t, _ in chunk]
            sp = [s for _, s in chunk]
            batch = self.backbone.encode_batch(trajs).to(device)
            tmask, emask = self.backbone.train_eval_masks(batch, sp)
            mbs.append(
                {
                    "batch": batch,
                    "train_mask": tmask,
                    "eval_mask": emask,
                    "n_train_steps": float(tmask.sum().item()),
                    "n_eval_steps": float(emask.sum().item()),
                }
            )
        return mbs

    # --- helpers -------------------------------------------------------
    def _latent(self, batch):
        """Injector latent for a batch, forwarding identity when present.

        Batches carry ``player_ids`` (column order); identity-keyed injectors
        (static embedding) use them, the GRU injector ignores them.
        """
        return self.injector.latent_trajectory(
            batch.feats, player_ids=getattr(batch, "player_ids", None) or None
        )

    def _set_train(self, mode: bool) -> None:
        for obj in (self.injector, self.backbone):
            net = getattr(obj, "_net", None)
            if net is not None and hasattr(net, "train"):
                net.train(mode)

    def _param_report(self) -> dict:
        report: dict = {}
        for label, obj in (
            ("injector", self.injector),
            ("backbone", self.backbone),
        ):
            fn = getattr(obj, "parameters", None)
            if callable(fn):
                report[f"{label}_params"] = int(sum(p.numel() for p in fn()))
        report["total_params"] = sum(v for k, v in report.items() if k.endswith("_params"))
        return report

    def _save_checkpoint(self, torch, handle) -> None:
        state = {}
        for label, obj in (
            ("injector", self.injector),
            ("backbone", self.backbone),
        ):
            net = getattr(obj, "_net", None)
            if net is not None:
                state[label] = net.state_dict()
        torch.save(state, handle.artifact_path("checkpoint.pt"))

    def _trainable_parameters(self, torch):
        """Yield torch parameters from injector/backbone if any exist."""
        for obj in (self.injector, self.backbone):
            params = getattr(obj, "parameters", None)
            if callable(params):
                yield from params()

    def save(self, path: str) -> None:
        import importlib.util

        if importlib.util.find_spec("torch") is None:  # pragma: no cover
            raise ImportError("torch required to save")
        import torch

        state = {}
        for label, obj in (
            ("injector", self.injector),
            ("backbone", self.backbone),
        ):
            net = getattr(obj, "_net", None)
            if net is not None:
                state[label] = net.state_dict()
        torch.save(state, path)
