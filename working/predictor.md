# Direct vs structural points predictor

Which of our two player-points predictors should feed the optimiser? This note records how we compared them, what
we had to fix in the direct predictor along the way, what the comparison found, and how each of the structural
model's scoring components performs. The numbers come from [predictor_comparison.ipynb](predictor_comparison.ipynb)
(steps 1-3, Findings) and [structural_components.ipynb](structural_components.ipynb) (step 4), which re-run
everything from the saved predictions.

## The two predictors

Both predict every player's points for the next gameweek, walk-forward: a gameweek is predicted only from the
gameweeks before it.

| | Direct | Structural |
|---|---|---|
| Code | `playground/predictors/23-24/src`, `playground/predictors/24-25/predictor/src` | `playground/predictors/structural/src` |
| Idea | one regression per position straight onto the gameweek's points | one model per scoring component (minutes, goals conceded, goals, assists, bonus, saves, cards), combined with the FPL scoring rules |
| Models | 9 base models (ridge, dt, rf, hgb, lgbm, xgb, cat, svr, dummy) and 2 ensembles; at each of 7 selection points the one with the lowest validation RMSE is used for the next 5 gameweeks | GLMs per component |
| Training data | current season, players with minutes in the previous 3 gameweeks | current season plus the previous one (the season-only run is also shown) |
| Output | `models/predictions.csv`, one row per player and gameweek | `outputs/{season}/structural_predictions_history_{prev}.csv`, one row per player and fixture |

## How they are compared

- **Same rows.** Both files cover exactly the same player-gameweeks (25,384 in 2023-24, 23,708 in 2024-25), so every
  score is paired.
- **Gameweeks 6-38.** The direct model's GW5 predictions were used to choose that model, so GW5 is left out for
  both.
- **Actual score:** realised FPL points for the gameweek, summed over both fixtures in a double gameweek.
- **Two populations:** every player with a fixture (what the optimiser chooses from) and players who played.
- **Metrics:** RMSE (the optimiser needs expected points, and squared error is minimised by the mean), MAE, R², bias,
  and Spearman rank correlation within each gameweek (picking a team is a ranking).
- **Significance:** errors within a gameweek are correlated, so the gameweek is the unit. Per-gameweek differences
  in squared error are tested with a Diebold-Mariano test and a Wilcoxon signed-rank test.
- **Decision check:** each gameweek, the top-k predicted players per position (2 GK, 5 DEF, 5 MID, 3 FWD, no budget)
  and the top predicted player (a captain pick), counted at their actual points.

## Steps

### 1. First comparison: the original direct predictor

The first run used the direct predictor as it stood in the repo-root `23-24/` and `24-25/predictor/` (the Chapter 3
pipeline). The structural model won clearly: RMSE 1.90 vs 2.16 (2023-24) and 1.89 vs 2.02 (2024-25). But the
comparison wasn't like for like, for three reasons:

1. **Different targets.** The direct models were trained on FPL's `xP`, not on points. The source's same-gameweek
   `xP` was recorded after the match, so it isn't a pre-match expectation either.
2. **Different training rows in 2023-24.** The 2023-24 direct models were trained only on players who played (about
   20 goalkeepers a gameweek) but predicted every player with a fixture (about 88). Non-players were over-predicted:
   bias +0.51 points overall, +1.15 for goalkeepers.
3. **Double gameweeks.** The source data has one row per player and gameweek, and in a double gameweek its
   `total_points` and `minutes` are only the first fixture's.

`playground/predictors/23-24` also held a half-finished conversion of the direct predictor to a points target: the
code and fold files had been converted, but the models were never retrained.

### 2. Fix the direct predictor

All changes are in `playground/predictors/`. The repo-root `xP` pipelines, and the Chapter 3 numbers built from them,
are untouched.

- **Target = the gameweek's points** (`config.TARGET = "total_points"`). `prepare_data.py` takes the gameweek totals
  from vaastav's per-fixture `merged_gw.csv`. Outside double gameweeks they match the source exactly (one stray row in
  2024-25).
- **`n_fixtures` as an input,** so a double gameweek can be predicted as two matches. The other features still
  describe the first fixture.
- **Every player with a fixture in 2023-24,** as 2024-25 already had: the models are validated and predict on
  everyone, and train on players with minutes in the previous 3 gameweeks (`data.training_rows`).
- **One set of scripts for both seasons** (`train`, `ensemble`, `deploy`, `compare`, `predictor`, `performance`).
  Only `config.py` and `model_dispatcher.py` differ.
- The gameweeks without `xP` (26 in 2023-24; 22, 32, 34 in 2024-25) are now only `config.MISSING_XP_GWS`, for
  rebuilding the lagged `xP` features. They no longer drop gameweeks from training.

### 3. Choose the direct models by RMSE, not MAE

After step 2 the direct predictor was still chosen by validation MAE, and it got worse at the top of the ranking.
MAE is minimised by the median, and with every player in the data most scores are 0. So svr, whose
epsilon-insensitive loss behaves like absolute error, won almost every selection (20 of 28 in 2023-24, 24 of 28 in
2024-25). Its predictions were shrunk:

- The top decile was under-predicted by 0.6-1.1 points (2024-25: 2.6 predicted, 3.7 scored).
- Players who played were under-predicted by about 1.2-1.3 points.
- The top-15 picks scored about 62 points per gameweek, less than the original `xP` version (about 70).

RMSE now decides throughout: model selection, the top-3 ensemble's ranking, and every results file and report. MAE
is kept as an extra column. Every model except svr already trains on squared error. svr stays as a candidate and is
now picked 8 times out of 28 (2023-24) and 4 (2024-25).

| Direct predictor, every player | 2023-24 | | | 2024-25 | | |
|---|---|---|---|---|---|---|
| | original (xP) | points, MAE | points, RMSE | original (xP) | points, MAE | points, RMSE |
| RMSE | 2.16 | 2.10 | **2.04** | 2.02 | 2.09 | **2.00** |
| Spearman within gameweek | 0.51 | 0.59 | **0.62** | 0.62 | 0.62 | **0.66** |
| Bias (points) | +0.51 | −0.10 | +0.09 | +0.01 | −0.32 | +0.05 |
| Top-15 picks, points per gameweek | 69.7 | 61.6 | 67.3 | 69.5 | 62.1 | 69.0 |

## Findings

Final setting: the direct predictor from steps 2 and 3, the structural predictor trained with the previous season.

### Accuracy

| Every player with a fixture | 2023-24 direct | 2023-24 structural | 2024-25 direct | 2024-25 structural |
|---|---|---|---|---|
| RMSE | 2.04 | **1.90** | 2.00 | **1.89** |
| MAE | 1.13 | **0.90** | 1.11 | **0.96** |
| R² | 0.26 | **0.36** | 0.28 | **0.36** |
| Bias | +0.09 | 0.00 | +0.05 | +0.02 |
| Spearman within gameweek | 0.62 | **0.73** | 0.66 | **0.74** |

- **The structural model has the lower squared error in every gameweek of both seasons** (33/33; Diebold-Mariano
  p ≈ 2e-12 and 2e-10). Its within-gameweek Spearman is higher in every gameweek as well.
- **It wins in every position.** RMSE, direct vs structural:

  | | GK | DEF | MID | FWD |
  |---|---|---|---|---|
  | 2023-24 | 1.61 vs 1.40 | 2.09 vs 1.93 | 2.05 vs 1.93 | 2.24 vs 2.09 |
  | 2024-25 | 1.71 vs 1.55 | 1.91 vs 1.82 | 2.05 vs 1.95 | 2.33 vs 2.12 |

- **Among players who played** it is ahead on RMSE (2.97 vs 3.11, 2.80 vs 2.93; better in 29/33 gameweeks in both
  seasons). MAE there is level (1.95 vs 1.94, 1.87 vs 1.88).

### Calibration

- **Both are unbiased overall, and both get the top decile about right.** Direct: 3.8 predicted vs 3.9 scored
  (2023-24), 4.0 vs 3.7 (2024-25). Structural: 4.1 vs 4.2, 4.0 vs 4.0.
- **Who plays is where the direct model loses.** It has no separate model of whether a player plays. It
  over-predicts the players who don't (bottom half of predictions: about 0.3 predicted vs 0.16-0.19 scored) and
  under-predicts the players who do (bias −0.79 and −0.65, against −0.44 and −0.37 for the structural `pred`). The
  structural model predicts playing time explicitly (`p_play`, `p60`) from FPL's pre-deadline status.

### Picking players

| Per gameweek | 2023-24 direct | 2023-24 structural | 2024-25 direct | 2024-25 structural |
|---|---|---|---|---|
| Top-15 picks, actual points | 67.3 | **75.0** | 69.0 | **78.5** |
| Captain pick, actual points | 2.9 | **6.1** | 7.0 | **9.0** |

- The structural top-15 picks score 7.7 and 9.5 more points per gameweek, better in 22/33 (Wilcoxon p = 0.006) and
  18/33 (p = 0.04) gameweeks. The two predictors share only 38-40% of their top-15 picks.
- Week to week this is noisier than the accuracy result: a few hauls decide a gameweek's top picks.

### Other checks

- **The previous season adds little to the structural model:** RMSE 1.90 with it, 1.90 without (a 0.005 gain in
  both seasons). Its advantage comes from the model, not from the extra data.
- **Against FPL's `xP`,** which neither is trained on, the structural model is closer too (RMSE 1.05 vs 1.33, 1.02
  vs 1.22).
- **Model selection is noisy.** In the direct pipeline, always using rf would have beaten the selection procedure by
  0.01-0.04 RMSE for most positions, because a single validation gameweek can't separate close models.

## The structural predictor, component by component (step 4)

The structural prediction is a sum of expected points per scoring action. To see which parts work, each component was
scored against its own actual points ([structural_components.ipynb](structural_components.ipynb)). Actual component
points are rebuilt from the FPL counts with the scoring rules, and they add up to `total_points` exactly. Data: the
deployed predictions, per player-fixture, gameweeks 5-38, every player with a fixture.

| Component | Points | How it's predicted |
|---|---|---|
| Appearance | 1 for 1-59 min, 2 for 60+ | minutes state (0 / 1-59 / 60+) with FPL's pre-deadline status |
| Clean sheet | GK/DEF 4, MID 1, on for 60+ | P(60+) × e^(−λ), λ from a team goals-against model |
| Goals conceded | GK/DEF −1 per 2 | E[floor(GA/2)], GA ~ Poisson(λ) |
| Goals | 6 / 6 / 5 / 4 | Poisson GLM on xG, rescaled to realised goals, with set-piece duties |
| Assists | 3 | Poisson GLM on xA, rescaled to realised assists |
| Bonus | 0-3 | Poisson GLM per position, capped at 3 |
| Saves | GK 1 per 3 | E[floor(S/3)], S ~ Poisson |
| Other | penalty saves, cards, own goals, missed penalties | constant mean per appearance |

**Accuracy per component** (every player-fixture). Skill = 1 − MSE / MSE of the player's mean from that component in
the season so far. Share of error = the component's share of the total error variance (cov(error_k, total error) /
var(total error); the shares add to 100%).

| Component | R² 23-24 | R² 24-25 | Skill 23-24 | Skill 24-25 | Share of error 23-24 | Share of error 24-25 |
|---|---|---|---|---|---|---|
| Appearance | 0.71 | 0.70 | 0.42 | 0.36 | 10% | 11% |
| Clean sheet | 0.17 | 0.17 | 0.10 | 0.11 | 16% | 18% |
| Goals conceded | 0.32 | 0.29 | 0.17 | 0.15 | 2% | 2% |
| Goals | 0.11 | 0.11 | 0.06 | 0.07 | 38% | 36% |
| Assists | 0.09 | 0.07 | 0.07 | 0.06 | 13% | 13% |
| Bonus | 0.08 | 0.10 | 0.06 | 0.06 | 17% | 17% |
| Saves | 0.45 | 0.41 | 0.16 | 0.18 | 1% | 1% |
| Other | 0.07 | 0.05 | 0.09 | 0.08 | 2% | 2% |
| **Total** | **0.34** | **0.34** | **0.15** | **0.13** | | |

- **Every component is about right on average.** Appearance, clean sheet, goals conceded, goals, assists and bonus
  match their actual means within 5% (2023-24) and 7% (2024-25); saves and other, with few events, within 6-12%. The
  largest position-level gaps: MID clean sheets are 14-16% low (the model treats a player on for 60+ as on for the
  whole match), FWD assists in 2024-25 are 20% high, and FWD bonus is +16% in 2023-24 and −9% in 2024-25.
- **Playing time carries most of the skill.** P(plays) and P(60+) are calibrated, with AUC 0.95-0.96 and a third of
  the base-rate Brier score. This is also what the direct predictor lacks (Findings, Calibration).
- **The error is in the match events.** Goals are 36-38% of the total error variance (about half for MID and FWD),
  clean sheets 16-18% (45-47% for GK, 36-42% for DEF), bonus 17% and assists 13%. Appearance is only 10-11%.
  Every component beats the season-mean baseline, but goals, assists and bonus explain only about a tenth of their
  variation (R² 0.07-0.11): whether the goal happens is mostly luck at this level.
- **The team model separates games only weakly.** λ is calibrated (P(0) and P(4+) goals conceded match the observed
  rates) but explains 10-12% of the deviance; the clean-sheet probability has AUC 0.65-0.67 and a Brier score only
  5% below the base rate.
- **Given that the player plays, each component is calibrated,** top decile included (goals: 1.60 predicted vs 1.59
  scored, and 1.58 vs 1.49). The saves curve is noisy: its top decile is 1.10 vs 1.29 in 2023-24 and 1.20 vs 1.00 in
  2024-25.

- **On their own events, the goal and assist models are calibrated and beat a player's own record.** Given that he
  plays, P(at least one goal) is 0.093 predicted vs 0.094 observed (2023-24) and 0.091 vs 0.087 (2024-25); assists
  0.088 vs 0.086 and 0.085 vs 0.079. AUC is 0.75-0.77 for goals and 0.70-0.71 for assists, against 0.73-0.75 and
  0.66-0.67 for the player's season-to-date rate (shrunk to his position's), and the model has the lower Brier score,
  log loss and Poisson deviance than both that baseline and the overall rate in both seasons. The margins are small
  (goals Brier 0.078 vs 0.079 vs 0.085 in 2023-24).
- **The limit is mostly finishing, not the model.** The predicted goal rate explains 22% of the variation in the
  player's realised xG but only 9-10% of his realised goals, and realised xG itself explains only 37-40% of goals.
  For assists: 14-17% of realised xA, 4-5% of assists, and xA explains 13-15% of assists. Defenders' goal rate
  explains 2-3% of their xG: their goals (set pieces) are the least predictable.

**Figures** (`working/figures/`, PDF): `structural_level` (predicted / actual by component and position),
`structural_accuracy` (R² and skill), `structural_error_share` (share of the error variance), `structural_probabilities`
(calibration of P(plays), P(60+), the clean sheet and λ), `structural_conditional_calibration` (each component given
that the player plays), `structural_events` (calibration and ROC of P(≥1 goal) and P(≥1 assist)), `structural_xg_check`
(the predicted rate against realised xG/xA and goals/assists).

The places to improve the structural model are therefore the goal, clean-sheet and bonus models, not the minutes
model, and the MID clean-sheet rule (a 60+ appearance is not a full match).

## Conclusion

On equal terms (same target, same players, same metric) the structural predictor is the better input for the
optimiser in both seasons. It is more accurate, it ranks players better within each gameweek, its expected points are
calibrated where the optimiser looks, and its picks score more. Most of the direct model's remaining gap comes from
not modelling whether a player plays.

Within the structural model, every component is about right on average and calibrated for players who play. Playing
time is predicted very well (AUC 0.95-0.96) and accounts for only a tenth of the remaining error. Most of the rest comes
from goals (36-38%), clean sheets (16-18%) and bonus (17%), which the model predicts only weakly (R² 0.08-0.17).
On their own events the goal and assist models are calibrated and beat the player's own record (AUC 0.75-0.77 and
0.70-0.71), and most of what they miss is finishing: realised xG explains only 37-40% of goals. These
models, the team goals-against model behind clean sheets (AUC 0.65-0.67), and the MID clean-sheet rule are where
further work would pay off.

## Caveats

- The structural RMSEs here (1.89-1.90) are higher than in the structural README (1.85-1.88), because this comparison
  starts at GW6 and scores double-gameweek players on their gameweek total.
- In a double gameweek the direct model's features describe the first fixture only; `n_fixtures` tells it there is a
  second one.
- The top-15 check ignores budget and club limits, so it measures ranking, not a feasible squad.
- The `model_comparison.ipynb` notebooks in the two direct-predictor folders were written for the MAE version and
  haven't been re-run.

## Reproduce

```
# direct predictor, each season (about 15-20 minutes each)
cd playground/predictors/23-24/src && bash run.sh
cd playground/predictors/24-25/predictor/src && bash run.sh

# comparison and component evaluation (ds_env kernel), run from working/
jupyter nbconvert --to notebook --execute --inplace predictor_comparison.ipynb
jupyter nbconvert --to notebook --execute --inplace structural_components.ipynb
```

The structural predictions are read as saved (`playground/predictors/structural/outputs/`); to rebuild them, see
`playground/predictors/structural/README.md` (`python structural.py --season 23-24 --history 22-23`, and 24-25 with
23-24).
