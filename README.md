# Constraint Satisfaction in Time-Series Foundation Models

Code accompanying a multi-model evaluation of whether time-series foundation models
satisfy the constraints real forecasting problems impose — hierarchical and temporal
coherence, non-negativity — and what it costs to restore them.

> **Supplementary material.** This repository accompanies a double-anonymized
> submission as its supplementary code and deliberately contains no author names or
> affiliations.

- **[FINDINGS.md](FINDINGS.md)** — plain-language summary of what we did and found.
- **[DATASETS.md](DATASETS.md)** — the datasets and exactly how they are prepared.
- **[paper/main.tex](paper/)** — the anonymized manuscript, with the generated tables
  and figures it inputs.

## What is and is not in this repository

Shipped: all the code, the manuscript source, and the generated tables and figures.

**Not shipped: the datasets (~2.5 GB) and the experiment outputs.** Both are
reproducible from what is here, and neither is ours to redistribute. Every dataset is
public and is downloaded automatically by the code on first use; the result CSVs are
produced by `experiments/run_all.py`. See below.

## Environment

One environment, managed by [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # create the environment from uv.lock
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`torch` is built against **CUDA 13.0**: the sm_121 accelerator used for these runs
needs nvrtc ≥ 13 to JIT-compile Moirai's kernels, and the cu128 build cannot. On other
hardware an ordinary CUDA build will work; the results are not accelerator-specific.

> The lockfile deliberately overrides two upstream pins (`uni2ts` wants `torch<2.5`,
> `gluonts` wants `pandas<2.2`). Those overrides are intentional — see the comment in
> `pyproject.toml`. Do not "fix" them.

## Getting the data

Nothing to do by hand. The loaders download and cache each dataset into `data/` on
first use:

| Hierarchy / dataset | Source | Fetched by |
|---|---|---|
| Tourism (small, large), Labour, Traffic, Wiki2 | Monash archive | `datasetsforecast` |
| M5 retail hierarchy | M5 competition | `datasetsforecast`, assembled by `m5_hierarchy.py` |
| Solar (137 PV stations) | GluonTS `solar_nips` | `gluonts`, assembled by `hierarchies_energy.py` |
| Electricity (40 feeders) | GluonTS `electricity_nips` | `gluonts`, assembled by `hierarchies_energy.py` |
| M3, M4 (monthly, quarterly), M5 daily | M-competitions | `datasetsforecast` |

Expect roughly 2.5 GB and a slow first run. Model weights are pulled from the Hugging
Face Hub on first use and cached likewise.

## Reproducing the paper

`run_all.py` is the single orchestrator. It writes each study's CSVs to `results/` as
soon as that study finishes, so it can be run detached and resumed study-by-study.

```bash
# 1. sanity-check the wiring on a tiny configuration (minutes)
uv run python experiments/run_all.py --quick

# 2. the full campaign (GPU; ~12 h on a single accelerator)
uv run python experiments/run_all.py

# 3. the three analyses run_all does not cover
uv run python experiments/run_accuracy_stats.py   # Wilcoxon + Benjamini-Hochberg
uv run python experiments/run_prob_law.py         # dependence regression + its figure
uv run python experiments/run_timing.py           # wall-clock cost of each step (GPU)

# 4. rebuild the paper's tables and figures from the CSVs just produced
uv run python experiments/make_tables.py          # -> paper/tables/*.tex
uv run python experiments/make_figures.py         # -> paper/figures/*

# 5. unit tests (no GPU, no data)
uv run pytest
```

Individual studies: `--studies coherence joint prob temporal nonneg frontier dm
ablation postproc` (`joint` is Chronos-2 in multivariate mode). Step 5 is the only one that needs neither GPU nor data; steps 3 and 4
require step 2 to have completed, since `results/` starts empty.

Every driver stamps the environment that produced its output into
`results/_provenance/`, so a rerun on different hardware stays traceable.

## Repository layout

```
src/tsfm_bench/        the library
  models.py            uniform wrappers: 6 foundation models + classical + CAA
  global_models.py     trained global baselines (LightGBM, native-PyTorch TCN)
  loaders.py           flat datasets (M3/M4/M5) with declared constraints
  hierarchies.py       registry of the 8 hierarchies, ascending dependence
  hierarchies_energy.py  Solar and Electricity, the two high-dependence hierarchies
  m5_hierarchy.py      the M5 retail hierarchy we construct
  coherence.py         cross-sectional coherence + reconciliation (BU / MinT / ERM)
  temporal.py          temporal-aggregation coherence (single series)
  prob_coherence.py    distributional coherence (sampling, Schaake, Gaussian copula)
  frontier.py          accuracy-vs-history sample-efficiency study
  _frontier_metrics.py MASE / scaling helpers used by the frontier
  paths.py             canonical absolute paths (keeps every driver cwd-independent)
  provenance.py        stamp environment + versions into results/_provenance/
  tables.py            auto-generate the paper's LaTeX tables from result CSVs
  figstyle.py          shared matplotlib style for the figures (text-block width, fonts)

experiments/           run_all.py (orchestrator) + the analyses and builders
tests/                 unit tests (window cap, summing-matrix identity, wrapper shapes)
paper/                 LaTeX source, generated tables and figures
```

`data/` and `results/` are created on first run and are not tracked.

## Building the paper

```bash
cd paper && latexmk -pdf main
```

`main.tex` is set to `review` (double-spaced, line-numbered) for submission. Switch the
`\documentclass` option to `preprint` for a single-spaced reading copy. The tables in
`paper/tables/` and figures in `paper/figures/` are committed as generated, so the
manuscript builds without rerunning any experiment.
