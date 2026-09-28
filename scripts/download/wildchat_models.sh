#!/usr/bin/env bash
# Download the checkpoints used by the WildChat simulator selection into $AT_MODELS,
# each at its pinned Hub revision:
#   Qwen3-1.7B, Qwen3-8B, Qwen3-32B          simulators (Qwen3-1.7B is also the reference tokenizer)
#   humanlike-7b, coser-8b, osim-8b,
#   humanlm-opinion                          released user simulators
#   all-MiniLM-L6-v2                         content metric and panel clustering (no pinned revision)
# Needs the `hf` CLI (huggingface_hub). Gated repositories use HF_TOKEN from the environment.
set -euo pipefail

PYTHON="${PYTHON:-python}"
MODELS="$("$PYTHON" -c 'from attribution_trials import paths; print(paths.MODELS)')"
mkdir -p "$MODELS"

fetch() {
  local dir="$1" repo="$2" rev="${3:-}"
  echo ">>> $repo${rev:+ @ $rev} -> $MODELS/$dir"
  if [ -n "$rev" ]; then
    hf download "$repo" --revision "$rev" --local-dir "$MODELS/$dir"
  else
    hf download "$repo" --local-dir "$MODELS/$dir"
  fi
}

fetch Qwen3-1.7B       Qwen/Qwen3-1.7B                           70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
fetch Qwen3-8B         Qwen/Qwen3-8B                             b968826d9c46dd6066d109eabc6255188de91218
fetch Qwen3-32B        Qwen/Qwen3-32B                            9216db5781bf21249d130ec9da846c4624c16137
fetch humanlike-7b     HumanLLMs/Human-Like-Qwen2.5-7B-Instruct  7cab6062ab32fa984f51d4bf0254472b2b320362
fetch coser-8b         Neph0s/CoSER-Llama-3.1-8B                 80d5c88072599026bd0b0e2eb4369a87e6a5c960
fetch osim-8b          cmu-lti/osim-8b                           a0002c6686e70684250479d325dc5f14bd366f46
fetch humanlm-opinion  snap-stanford/humanlm-opinion             4065187f3747cd1730d8fdd6a0b065b687cf3d72
fetch all-MiniLM-L6-v2 sentence-transformers/all-MiniLM-L6-v2
echo "done"
