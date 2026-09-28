"""Re-run the WildChat generation at a lower decoding temperature (default 0.7).

Runs ``generate.main`` unchanged for one simulator with ``generate.TEMPERATURE``
overridden and its outputs redirected to ``paths.WILDCHAT /
"temperature_<T>"``, so the temperature-1 files are never touched.  Frame,
panel, cards, imposter map, prefix contract, K=8, 96 new tokens, seed 0 and
both content metrics are the same as the temperature-1 run.

  CUDA_VISIBLE_DEVICES=0 GENERATION_BATCH=32 python -m attribution_trials.wildchat.temperature 8b
"""

from __future__ import annotations

import argparse
import json
import sys

from attribution_trials import paths
from attribution_trials.wildchat import generate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()
    out = paths.WILDCHAT / f"temperature_{args.temperature:g}"
    (out / "samples").mkdir(parents=True, exist_ok=True)

    generate.TEMPERATURE = args.temperature
    generate.RESULTS = out
    generate.SAMPLES = out / "samples"

    sys.argv = [sys.argv[0], args.candidate]
    generate.main()

    result = json.loads((out / f"content_{args.candidate}.json").read_text())
    if result["temperature"] != args.temperature:
        raise AssertionError("result did not record the overridden temperature")


if __name__ == "__main__":
    main()
