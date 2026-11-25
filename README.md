# An Optimization Problem for a Head-to-Head Competition of the Fantasy Football League

Repository: <https://github.com/ilyandho/FPL-H2H-Optimizer>

This repository holds the code, input data and notebooks behind the MSc dissertation of the same title. The
dissertation itself is submitted separately and is not stored here. The repository holds only the parts the final
thesis uses, and every path below is relative to the repository root.

## Layout

| Folder | What it holds |
|---|---|
| `structural/src/` | The pipeline: data build (`build_data.py`, `build_availability.py`), the structural points predictor (`structural.py`), the FantasyGo selection model (`selection.py`), the Haugh–Singal MVT for FantasyGo (`mvt.py`, `mvt_portfolio.py`), and the H2H problem (`h2h.py`, `h2h_data.py`, `h2h_opponents.py`, `h2h_chips.py`). Settings and every path: `config.py`. |
| `structural/data/` | Inputs for 2022-23 to 2024-25: players and features, availability, FantasyGo contests (2023-24), the H2H league (2024-25), and the raw download caches in `sources/`. |
| `structural/notebooks/` | The predictor, selection-model and MVT notebooks. |
| `opponent_model/` | The H2H opponent model: the league Dirichlet-multinomial notebooks (`beta estimators/league betas/`), the manager-fit and copula notebooks (`beta estimators/opponent betas/`), the fitting code (`beta estimators/utils.py`), and their inputs (`inputs/`). |
| `direct_predictor/` | Predictions of the direct machine-learning predictor (2023-24, 2024-25), for the structural-vs-direct comparison. Only the predictions are kept, not the model code or pickles. |
| `working/` | The thesis analyses: the H2H season simulator (`Optimizer/baseline.py`, `lineups.py`), FantasyGo evaluation (`FantasyGo/fantasygo.py`), `opponent_eval.py`, `calibration_check.py`, `calibration_check_fg.py`, the figures (`thesis_figures.py` → `figures/`), and the predictor notebooks. Text summaries of the thesis runs are in `Optimizer/outputs/` and `FantasyGo/outputs/`. |

The scripts write their fitted models and result caches to `structural/outputs/` and the `opponent_model/` folders
when they run. These are not in the repository; see "What is not in the repository" below.

## Setup

```bash
git clone https://github.com/ilyandho/FPL-H2H-Optimizer.git
cd FPL-H2H-Optimizer
python3.11 -m venv .venv && source .venv/bin/activate   # or a conda/mamba env
pip install -r requirements.txt
```

The optimizers use Gurobi, which needs a licence; a free academic licence works. On a shared machine, limit threads:
`export OMP_NUM_THREADS=2` and pass `--threads 2` to `baseline.py`.

## Reproducing the results

A fresh clone has the inputs but none of the fitted models or result caches, so the results are rebuilt from the
inputs. Run the steps roughly in pipeline order. All commands run from `structural/src/`, except `baseline.py`, which
runs from `working/`:

1. `python build_data.py --season 23-24` and `python build_availability.py --season 23-24` (and the same for 22-23 and
   24-25). These read the download caches in `structural/data/sources/`, so they do not need the network.
2. `python structural.py --season 23-24 --history 22-23` (and `--season 24-25 --history 23-24`): the points predictions.
3. `python selection.py`: the FantasyGo field model.
4. `python mvt.py ...`: the FantasyGo MVT. The script's help lists the objectives and variants (`--objective topheavy`,
   `--sigma-shrink`, `--track-k`).
5. The opponent-model notebooks in `opponent_model/beta estimators/` refit the league and manager models.
6. `python Optimizer/baseline.py --league --threads 2` from `working/`: the H2H season simulation for the 20 teams.

Each script's docstring lists its options and outputs. Steps 3 to 6 (the model fits and the season
simulations) are the slow part.

Once the fits and caches exist, the figures, the H2H comparison and the predictor notebooks are quick to produce,
because they read the caches:

```bash
cd working
python thesis_figures.py sigma sigmap h2h squads opponents field fgmethods fantasygo frontiers traces s1coefs
python Optimizer/baseline.py --compare-teams --league --threads 2   # the H2H comparison over the 20 league teams
jupyter nbconvert --to notebook --execute predictor_comparison.ipynb structural_components.ipynb
```

To check the reported numbers without rerunning anything, see the text summaries in `working/Optimizer/outputs/` and
`working/FantasyGo/outputs/`, and the figures in `working/figures/`.

## What is not in the repository

The repository keeps:
- the code, notebooks, docs and figures;
- the text summaries of the results;
- the input data: `structural/data/`, `opponent_model/inputs/` and `direct_predictor/`.

The five input CSVs over 40 MB are stored gzipped and read as `.csv.gz`:
- `players.csv.gz` for each season;
- `rolled_data_24-25_xp_fixed.csv.gz`;
- `FFS_manager_weekly_selections.csv.gz`.

pandas reads these directly, so no file in the repository is over 25 MB.

These are left out:
- the thesis (LaTeX source and PDF), which is submitted separately;
- the fitted models and generated caches: `structural/outputs/`, `league_models_*`, the manager fits
  (`opponent betas/estimates/`), every `*.pkl` and `*.joblib`;
- run logs.

