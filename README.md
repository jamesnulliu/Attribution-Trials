# Attribution Trials

Code for *Is There an Imposter Among Us? Auditing User Simulators with Attribution Trials*.

Attribution Trials (AT) audits whether a user simulator represents the individual it is given.
Each simulator predicts a user's held-out next actions under the target user's information (treatment)
and under four controls: no user information, generic population information, matched-group information,
and a matched imposter's information.
AT reports the gain of the target's information over each control and credits individual fidelity only
when all four gains are significant.

## Installation

With [uv](https://docs.astral.sh/uv/) (uses a uv-managed Python, see `uv.toml`):

```bash
uv sync --extra all          # or pick extras: --extra gpu --extra chess --extra api
```

With pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"      # or pick extras: ".[gpu,chess,api]"
```

Extras: `gpu` (training and scoring), `chess` (Lichess ingest), `api` (OpenAI-compatible calls),
`wandb` (optional tracking), `dev` (ruff). Format with `ruff format` and lint with `ruff check`.

Every step is a module run from the repository root, e.g. `python -m attribution_trials.analysis.gains`
(with uv: `uv run python -m ...`).

## Paths and credentials

| Variable | Default | Content |
|---|---|---|
| `AT_DATA` | `data/` | raw and prepared datasets |
| `AT_MODELS` | `models/` | local model checkpoints |
| `AT_RESULTS` | `results/` | every output (`audit/`, `analysis/`, `comparison/`, `wildchat/`, `validation/`, `figures/`, `tables/`) |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | | API models, the known-signal controls and the LLM judge |
| `HF_TOKEN` | | gated Hugging Face downloads |
| `WANDB_API_KEY` | | optional; tracking is skipped without it |

## Data

- **Lichess** (chess): `bash scripts/data/lichess.sh` downloads three monthly archives of the Lichess open
  database and ingests the three blitz cohorts of the pooled chess panel.
- **ASSISTments 2009** (knowledge tracing): place `skill_builder_data_corrected.csv` in `$AT_DATA/kt/raw/`, then
  `python -m attribution_trials.data.prepare_kt`.
- **OPeRA** (online shopping): downloaded at a pinned revision on first use (`attribution_trials.data.opera`).
- **WildChat-1M**: `bash scripts/download/wildchat.sh`.
- **PersonaMem** and **HorizonBench**: downloaded at pinned revisions by the validation modules.

## Pipeline

### 1. Audit: per-user scores under the five conditions (Section 4)

```bash
bash scripts/audit/latent_families.sh             # static embedding, recurrent embedding, structured memory
bash scripts/audit/lora_matrix.sh [n_gpus]        # user-profile LoRA, chess and KT
bash scripts/audit/frozen_panel.sh                # frozen prompt, chess and KT
bash scripts/audit/run_opera_jobs.sh [n_gpus]     # user-profile LoRA and frozen prompt, OPeRA
bash scripts/download/released_simulators.sh
bash scripts/audit/released_panel.sh              # CoSER-8B, HumanLike-7B
bash scripts/audit/osim_panel.sh                  # Osim-4B/8B, with and without midtraining
bash scripts/audit/api_panel.sh                   # GPT-4.1, GPT-4.1-mini, DeepSeek-V3.2
```

### 2. Gains, significance and the main results (Section 5.1, Appendices A–C)

```bash
python -m attribution_trials.analysis.identity
python -m attribution_trials.analysis.gains
python -m attribution_trials.analysis.master_ladder
python -m attribution_trials.analysis.seed_inference
python -m attribution_trials.analysis.api_rows
python -m attribution_trials.figures.attribution_map
python -m attribution_trials.tables.combinations
python -m attribution_trials.tables.master_ladder
python -m attribution_trials.tables.seed_inference
```

`analysis.shares` and `tables.main_results` run after steps 3 and 4, whose outputs they read.

### 3. Existing benchmarks and per-user baselines (Section 5.2, Appendices C–D)

```bash
bash scripts/comparison/per_option_frozen.sh
bash scripts/comparison/per_option_olmo.sh
bash scripts/comparison/per_option_released.sh
bash scripts/comparison/per_option_cross.sh
bash scripts/comparison/cpu_readouts.sh           # readouts, marginal and stronger baselines, tables
python -m attribution_trials.analysis.shares
```

### 4. Simulator selection on WildChat (Section 5.3, Appendix E)

```bash
bash scripts/download/wildchat_models.sh
python -m attribution_trials.wildchat.build_panel
python -m attribution_trials.wildchat.build_frame
GPUS=0,1,2,3,4,5,6 bash scripts/wildchat/score_trial.sh
python -m attribution_trials.wildchat.metric_screen
GPUS=0,1,2,3,4,5,6 GENERATION_BATCH=4 bash scripts/wildchat/generate.sh
python -m attribution_trials.wildchat.readouts
python -m attribution_trials.wildchat.selectors
python -m attribution_trials.wildchat.frechet
bash scripts/wildchat/judge.sh
GPUS=0,1 TEMPERATURE=0.7 GENERATION_BATCH=32 bash scripts/wildchat/temperature.sh
python -m attribution_trials.tables.selector_comparison
python -m attribution_trials.tables.judge_stranger
python -m attribution_trials.tables.temperature_check
python -m attribution_trials.tables.main_results
```

### 5. Validation and ablations (Section 6, Appendix F)

```bash
# known-signal controls
python -m attribution_trials.validation.personamem --stage screen
python -m attribution_trials.validation.personamem --stage powered
python -m attribution_trials.validation.horizonbench --stage screen
python -m attribution_trials.validation.horizonbench --stage powered
python -m attribution_trials.validation.no_history_baseline
# synthetic populations
python -m attribution_trials.validation.lambda_sweep
python -m attribution_trials.validation.stress_nulls
python -m attribution_trials.validation.reference_nulls
# closer imposters
python -m attribution_trials.validation.static_cross_matrices
python -m attribution_trials.validation.hardest_imposter
python -m attribution_trials.validation.nearest_imposter
for s in 0 1 2; do
  python -m attribution_trials.validation.matcher_recurrent $s
  python -m attribution_trials.validation.train_self_adapters $s
  python -m attribution_trials.validation.matcher_lora $s
done
python -m attribution_trials.validation.matcher_frozen
python -m attribution_trials.validation.matcher_hierarchy
# target-information dose
bash scripts/validation/dose.sh
# tables
python -m attribution_trials.tables.ablation
python -m attribution_trials.tables.identity_recount
python -m attribution_trials.tables.matcher_hierarchy
python -m attribution_trials.tables.stress_nulls
```

### 6. Figures

```bash
python -m attribution_trials.figures.headline_data
python -m attribution_trials.figures.headline
python -m attribution_trials.figures.dose_ladder
```

The figures use the Times New Roman font.

## License

Apache-2.0.
