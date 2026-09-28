"""Job list for the OPeRA GPU panels (user-profile LoRA and frozen prompt).

A0 reuse: the `none` arm has no card, so it is identical across configs.
`none` is trained only for the `combined` config and serves as A0 for the
static/dynamic configs too, which therefore carry only {placebo, group, self}.

  LoRA 1.7B / 8B: combined with seeds 0 1 2; static/dynamic with seed 0,
                  plus seeds 1 2 for the 8B timing static/dynamic configs.
  frozen prompt:  1.7B, 8B and 30B on every channel x config.

Prints one job per line:
  SFT    <substrate> <cond> <seed> <size>
  FROZEN <substrate> <size>

  python -m attribution_trials.audit.opera_jobs
"""

CHAN = ["action", "timing"]
FULL = ["none", "placebo", "group", "self"]
NONONE = ["placebo", "group", "self"]
CONFIGS = ["combined", "static", "dynamic"]


def sft(sub, cond, seed, size):
    return f"SFT {sub} {cond} {seed} {size}"


def opera_sft(size, seeds_combined):
    out = []
    for ch in CHAN:
        for cfg in CONFIGS:
            conds = FULL if cfg == "combined" else NONONE
            seeds = seeds_combined if cfg == "combined" else [0]
            for cond in conds:
                for s in seeds:
                    out.append(sft(f"opera-{ch}-{cfg}", cond, s, size))
    return out


def opera_frozen(size):
    return [f"FROZEN opera-{ch}-{cfg} {size}" for ch in CHAN for cfg in CONFIGS]


def jobs():
    out = opera_sft("1.7b", [0, 1, 2]) + opera_frozen("1.7b")
    out += opera_sft("8b", [0, 1, 2]) + opera_frozen("8b")
    out += opera_frozen("30b")
    for cfg in ["static", "dynamic"]:
        for s in [1, 2]:
            for cond in NONONE:
                out.append(sft(f"opera-timing-{cfg}", cond, s, "8b"))
    return out


if __name__ == "__main__":
    print("\n".join(jobs()))
