"""Shared WildChat simulator interfaces.

Qwen3 and Osim use a plain Qwen3 chat-markup text prompt; HumanLike and CoSER
use their native chat templates with the card in the system persona slot;
HumanLM uses the native persona/task contract from its model card and the
response wrapper.
"""

from __future__ import annotations

import math
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from attribution_trials import paths

PANEL_RUNTIME = paths.WILDCHAT / "panel_runtime.json"

SEED = 0
PREFIX_BUDGET = 2000
ARMS = ("A0", "A3", "A4")
REFERENCE_TOKENIZER_PATH = paths.MODELS / "Qwen3-1.7B"
_REF_TOK = None

SYSTEM_BASE = (
    "You are simulating a specific user who is chatting with an AI assistant. "
    "Write the user's messages exactly as that user would."
)
SYSTEM_CARD = SYSTEM_BASE + (" Here are examples of messages this user has written before:\n{card}")

HUMANLM_SYSTEM = (
    "You are a real human user. Your name is HUMAN. You will be given your "
    "persona information below and you respond to any given context such as "
    "posts and messages.\n\nYour persona:\n{card}\n"
    "<|The End of Persona|>\n\n## Task and Output format:<response>\n"
    "<HUMAN's actual written comment or reply text.>\n</response>"
)

CANDIDATES = {
    "1.7b": {
        "model_path": paths.MODELS / "Qwen3-1.7B",
        "model": "Qwen/Qwen3-1.7B",
        "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "interface": "qwen3_frozen",
    },
    "8b": {
        "model_path": paths.MODELS / "Qwen3-8B",
        "model": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "interface": "qwen3_frozen",
    },
    "dense": {
        "model_path": paths.MODELS / "Qwen3-32B",
        "model": "Qwen/Qwen3-32B",
        "revision": "9216db5781bf21249d130ec9da846c4624c16137",
        "interface": "qwen3_frozen",
    },
    "humanlike": {
        "model_path": paths.MODELS / "humanlike-7b",
        "model": "HumanLLMs/Human-Like-Qwen2.5-7B-Instruct",
        "revision": "7cab6062ab32fa984f51d4bf0254472b2b320362",
        "interface": "native_chat_persona",
    },
    "coser": {
        "model_path": paths.MODELS / "coser-8b",
        "model": "Neph0s/CoSER-Llama-3.1-8B",
        "revision": "80d5c88072599026bd0b0e2eb4369a87e6a5c960",
        "interface": "native_chat_persona",
    },
    "osim": {
        "model_path": paths.MODELS / "osim-8b",
        "model": "cmu-lti/osim-8b",
        "revision": "a0002c6686e70684250479d325dc5f14bd366f46",
        "interface": "qwen3_text_osim",
    },
    "humanlm": {
        "model_path": paths.MODELS / "humanlm-opinion",
        "model": "snap-stanford/humanlm-opinion",
        "revision": "4065187f3747cd1730d8fdd6a0b065b687cf3d72",
        "interface": "humanlm_native",
    },
}

# All seven simulators, in table order.
SIMULATORS = ("1.7b", "8b", "dense", "humanlike", "coser", "osim", "humanlm")
# Scored by score_trial.py; Qwen3-1.7B and Qwen3-8B are scored by score_trial_frozen.py.
TRIAL_CANDIDATES = ("dense", "humanlike", "coser", "osim", "humanlm")


def row_key(uid: str, decision: dict) -> str:
    return f"{uid}|{decision['conv_id']}|{int(decision['turn_index'])}"


def _seg(tok, role: str, text: str) -> list[int]:
    return tok.encode(
        f"<|im_start|>{role}\n{text}<|im_end|>\n",
        add_special_tokens=False,
    )


def _frozen_context(tok, card: str | None, prefix: list[dict]) -> list[int]:
    system = SYSTEM_CARD.format(card=card) if card else SYSTEM_BASE
    ids = _seg(tok, "system", system)
    pieces = []
    for turn in prefix:
        role = turn.get("role", "user")
        role = role if role in ("user", "assistant") else "user"
        content = " ".join(str(turn.get("content", "")).split())[:6000]
        pieces.append(_seg(tok, role, content))
    while pieces and sum(len(x) for x in pieces) > PREFIX_BUDGET:
        pieces.pop(0)
    for piece in pieces:
        ids.extend(piece)
    ids.extend(tok.encode("<|im_start|>user\n", add_special_tokens=False))
    return ids


def _reference_tokenizer():
    global _REF_TOK
    if _REF_TOK is None:
        _REF_TOK = AutoTokenizer.from_pretrained(
            str(REFERENCE_TOKENIZER_PATH), local_files_only=True
        )
    return _REF_TOK


def _normalized_turns(prefix: list[dict]) -> list[dict]:
    """Per-turn normalization identical to _frozen_context."""
    turns = []
    for turn in prefix:
        role = turn.get("role", "user")
        role = role if role in ("user", "assistant") else "user"
        content = " ".join(str(turn.get("content", "")).split())[:6000]
        turns.append({"role": role, "content": content})
    return turns


def select_prefix_turns(prefix: list[dict]) -> list[dict]:
    """Newest turns that fit PREFIX_BUDGET reference (Qwen3-1.7B) tokens.

    The normalization and token counting are the same as _frozen_context.
    All seven candidates see this same text; only the tokenization of the
    resulting native prompt is candidate-specific.
    """
    ref = _reference_tokenizer()
    turns = _normalized_turns(prefix)
    sizes = [len(_seg(ref, t["role"], t["content"])) for t in turns]
    while turns and sum(sizes) > PREFIX_BUDGET:
        turns.pop(0)
        sizes.pop(0)
    return turns


def _dialogue_messages(prefix: list[dict]) -> list[dict]:
    messages = []
    for turn in prefix:
        role = turn.get("role", "user")
        role = role if role in ("user", "assistant", "system") else "user"
        messages.append(
            {
                "role": role,
                "content": str(turn.get("content", "")),
            }
        )
    # Generic chat templates generate the next simulated message as an
    # assistant completion.  The instruction is a fixed adapter, independent
    # of arm and candidate, while the logged dialogue remains role-preserved.
    messages.append(
        {
            "role": "user",
            "content": "Write the next message exactly as this user would.",
        }
    )
    return messages


def _native_chat_context(
    tok, candidate: str, card: str | None, prefix: list[dict], for_generation: bool = False
) -> tuple[list[int], dict]:
    if tok.chat_template is None:
        raise RuntimeError(f"{candidate}: tokenizer has no native chat template")
    selected_prefix = select_prefix_turns(prefix)
    if candidate == "humanlm":
        # A0 has no card. HumanLM's native contract still needs its
        # persona/task wrapper, so A0 carries an empty persona field;
        # A3/A4 carry the imposter's or the target's card verbatim.
        card_text = card or ""
        system = HUMANLM_SYSTEM.format(card=card_text)
        messages = [{"role": "system", "content": system}]
        messages.extend(_dialogue_messages(selected_prefix)[:-1])
        # The native HumanLM template generates a user turn and opens <think>.
        text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(for_generation),
        )
        if not for_generation:
            # Teacher forcing measures only the realized response, not a
            # latent reasoning trace.
            text += "<response>\n"
        return tok(text, add_special_tokens=False)["input_ids"], {
            "template_mode": "native_humanlm_response_wrapper",
            "enable_thinking": bool(for_generation),
            "response_prefix": "<response>\\n" if not for_generation else None,
            "persona_present": bool(card),
            "prefix_budget_tokens": PREFIX_BUDGET,
            "prefix_turns_kept": len(selected_prefix),
        }
    system = (
        "You are this user. Reproduce the user's next message from the "
        "conversation below.\n\nPersona:\n" + (card or "")
    )
    messages = [{"role": "system", "content": system}]
    messages.extend(_dialogue_messages(selected_prefix))
    text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tok(text, add_special_tokens=False)["input_ids"], {
        "template_mode": "native_chat_persona",
        "enable_thinking": False,
        "system_persona_wrapper": "You are this user...Persona:",
        "prefix_budget_tokens": PREFIX_BUDGET,
        "prefix_turns_kept": len(selected_prefix),
    }


def build_context(
    tok, candidate: str, card: str | None, prefix: list[dict], for_generation: bool = False
) -> tuple[list[int], dict]:
    """Return prompt ids and interface metadata."""
    interface = CANDIDATES[candidate]["interface"]
    if interface in {"qwen3_frozen", "qwen3_text_osim"}:
        return _frozen_context(tok, card, prefix), {
            "template_mode": "qwen3_text",
            "prefix_budget_tokens": PREFIX_BUDGET,
            "for_generation": bool(for_generation),
        }
    return _native_chat_context(tok, candidate, card, prefix, for_generation)


def target_ids(tok, candidate: str, target: str) -> list[int]:
    # Only the realized text is the measured target for every interface; for
    # HumanLM the response wrapper is context and the closing tag is not scored.
    return tok.encode(str(target), add_special_tokens=False)


def generation_context(
    tok, candidate: str, card: str | None, prefix: list[dict]
) -> tuple[list[int], dict]:
    return build_context(tok, candidate, card, prefix, for_generation=True)


def parse_generated(candidate: str, tok, ids: list[int]) -> tuple[str, bool]:
    """Decode a sample and return (realized text, terminal flag)."""
    eos = tok.eos_token_id
    first_eos = bool(ids and eos is not None and ids[0] == eos)
    text = tok.decode(ids, skip_special_tokens=True)
    if candidate == "humanlm":
        # HumanLM may expose reasoning and response tags in the decoded sample.
        if "<response>" in text:
            text = text.split("<response>", 1)[1]
        if "</response>" in text:
            text = text.split("</response>", 1)[0]
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = text.strip()
    return text, bool(first_eos or not text)


def feature_code(text: str) -> bool:
    if "```" in text:
        return True
    lines = text.splitlines()
    matches = 0
    for line in lines:
        if re.search(r"^\s*(def|import|from|class|for|while|if)\b", line):
            matches += 1
        elif re.search(r"[{};]\s*$", line):
            matches += 1
    return matches >= 2


def card_for(panel: dict, uid: str, arm: str) -> str | None:
    if arm == "A0":
        return None
    imposter = panel["imposter"][uid]
    if arm == "A3":
        return panel["users"][imposter]["profile_card"]
    if arm == "A4":
        return panel["users"][uid]["profile_card"]
    raise ValueError(arm)


def load_local_model(candidate: str):
    spec = CANDIDATES[candidate]
    path = spec["model_path"]
    if not path.is_dir():
        raise FileNotFoundError(path)
    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cuda",
    ).eval()
    return tok, model


@torch.inference_mode()
def score_candidate_target(
    model, tok, candidate: str, prompt_ids: list[int], target: str, device: str = "cuda"
) -> tuple[float, int]:
    tids = target_ids(tok, candidate, target)
    if not tids:
        raise ValueError("empty target")
    ids = torch.tensor([prompt_ids + tids], dtype=torch.long, device=device)
    logits = model(ids).logits[0, len(prompt_ids) - 1 : -1, :].float()
    target_t = torch.tensor(tids, dtype=torch.long, device=device)
    loss = torch.nn.functional.cross_entropy(logits, target_t, reduction="mean")
    value = float(loss.cpu())
    if not math.isfinite(value):
        raise FloatingPointError("non-finite NLL")
    return value, len(tids)
