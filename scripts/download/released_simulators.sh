#!/usr/bin/env bash
# Download the released simulators at the revisions released_panel pins,
# one model at a time (resumable: rerun after an interruption).
#
#   scripts/download/released_simulators.sh
set -euo pipefail
python - <<'PY'
from huggingface_hub import snapshot_download

from attribution_trials.audit.released_panel import REVISIONS, SIMULATORS

for tag, repo in SIMULATORS.items():
    print(snapshot_download(repo_id=repo, revision=REVISIONS[tag]), flush=True)
PY
