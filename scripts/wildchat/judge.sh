#!/usr/bin/env bash
# LLM judge of the generated WildChat turns: build requests -> send them -> parse -> readout.
# The send step is resumable: rerun this script to pick up where it stopped.
#
# Environment:
#   OPENAI_API_KEY    required; key for an OpenAI-compatible Responses API
#   OPENAI_BASE_URL   optional; base URL of that API (default: the OpenAI API)
#   MIN_INTERVAL      seconds between request starts (default 1.1)
#   CONCURRENCY       requests in flight (default 6)
#   REASONING         reasoning effort sent with each request (default low)
set -euo pipefail

PYTHON="${PYTHON:-python}"
: "${OPENAI_API_KEY:?set OPENAI_API_KEY}"
MIN_INTERVAL="${MIN_INTERVAL:-1.1}"
CONCURRENCY="${CONCURRENCY:-6}"
REASONING="${REASONING:-low}"

"$PYTHON" -u -m attribution_trials.wildchat.judge build --reasoning "$REASONING"
"$PYTHON" -u -m attribution_trials.wildchat.judge sync --min-interval "$MIN_INTERVAL" --concurrency "$CONCURRENCY"
"$PYTHON" -u -m attribution_trials.wildchat.judge collect
"$PYTHON" -u -m attribution_trials.wildchat.judge_readout
