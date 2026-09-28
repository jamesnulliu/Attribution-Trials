"""Optional Weights & Biases tracking for training runs.

:func:`start_run` logs a training run to W&B when ``WANDB_API_KEY`` is set
and otherwise returns a run that only writes to the local
:class:`~attribution_trials.results.RunHandle`. Tracking never changes what
is trained or scored.

``wandb`` is imported only when a key is set, so it is needed only on hosts
that track. Set ``WANDB_MODE=offline`` to keep a tracked run local.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from attribution_trials.results import RunHandle

WANDB_API_KEY_ENV = "WANDB_API_KEY"
#: Optional overrides; project/entity can also be passed explicitly.
WANDB_PROJECT_ENV = "WANDB_PROJECT"
WANDB_ENTITY_ENV = "WANDB_ENTITY"

DEFAULT_PROJECT = "attribution-trials"


@dataclass
class WandbRun:
    """Wrapper over a live ``wandb`` run (or none), mirroring to the local store.

    Use :meth:`log` for streaming metrics (W&B and the local
    ``metrics.jsonl``) and :meth:`summary` for final numbers (W&B run summary
    and the local ``metrics.json``). With ``run=None`` only the local store
    is written.
    """

    run: object | None  # the wandb.sdk.wandb_run.Run, or None when untracked
    handle: RunHandle | None = None

    @property
    def id(self) -> str | None:
        return self.run.id if self.run is not None else None

    @property
    def url(self) -> str | None:
        if self.run is None:
            return None
        getter = getattr(self.run, "get_url", None)
        return getter() if callable(getter) else getattr(self.run, "url", None)

    def log(self, metrics: dict, step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(metrics, step=step)
        if self.handle is not None:
            self.handle.log_metrics(metrics, step=step)

    def summary(self, metrics: dict) -> None:
        if self.run is not None:
            for k, v in metrics.items():
                self.run.summary[k] = v
        if self.handle is not None:
            self.handle.set_summary(metrics)

    def finish(self, status: str = "completed") -> None:
        if self.run is None:
            return
        exit_code = 0 if status == "completed" else 1
        try:
            self.run.finish(exit_code=exit_code)
        except TypeError:  # older wandb signatures
            self.run.finish()


def start_run(
    *,
    experiment: str,
    config: dict,
    handle: RunHandle | None = None,
    project: str | None = None,
    entity: str | None = None,
    tags: list[str] | None = None,
) -> WandbRun:
    """Start a W&B run if ``WANDB_API_KEY`` is set; otherwise track locally.

    When a local ``handle`` is supplied and W&B is used, the W&B run id/url
    is written into the run's ``run.json``.
    """
    key = os.environ.get(WANDB_API_KEY_ENV, "").strip()
    if not key:
        return WandbRun(run=None, handle=handle)

    try:
        import wandb
    except ImportError as e:  # pragma: no cover - env-dependent
        raise ImportError(
            f"{WANDB_API_KEY_ENV} is set but wandb is not installed: "
            "pip install wandb (or unset the key to train untracked)"
        ) from e

    # Non-interactive login with the env key (never prompts).
    wandb.login(key=key, relogin=False, verify=False)

    run = wandb.init(
        project=project or os.environ.get(WANDB_PROJECT_ENV, DEFAULT_PROJECT),
        entity=entity or os.environ.get(WANDB_ENTITY_ENV) or None,
        name=handle.run_id if handle is not None else None,
        group=experiment,
        tags=tags or [],
        config=config,
        # Reuse the local run dir so wandb's own files land beside ours.
        dir=str(handle.dir) if handle is not None else None,
    )
    wrapped = WandbRun(run=run, handle=handle)
    if handle is not None:
        handle.attach_wandb(run_id=wrapped.id, url=wrapped.url)
    return wrapped
