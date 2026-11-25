# Structural xP model

Predicts each FPL player's points for every gameweek by predicting each scoring component (minutes, team goals
conceded, goals, assists, bonus, saves, cards) and combining them with the FPL scoring rules. Every model for
gameweek *gw* is fitted on gameweeks before *gw* only. Works for 2023-24 and 2024-25 (`config.SEASONS`), with
2022-23 built as training history for 2023-24; the scoring rules are the same in all three and reproduce
`total_points` exactly for every position.

## Layout

```
src/
  config.py               seasons, positions, paths(season)
  build_data.py           every player with a fixture, per gameweek, with lagged features   (once per season)
  build_availability.py   pre-deadline FPL status, chance of playing, set-piece orders       (once per season)
  structural.py           the model: load, fit, walk-forward prediction, scoring; also a CLI
  selection.py            2023-24 only: the FantasyGo field's selection model (Bayesian Dirichlet-multinomial)
  mvt.py                  2023-24 only: the MVT optimizer for the FantasyGo contests, on the model's predictions
  irl.py                  2023-24 only: an alternative optimizer, heuristic-guided inverse RL with an HGAT policy
notebooks/
  selection.ipynb         the selection model and its evaluation: convergence, out-of-sample log loss against
                          baselines, calibration of each player's pick count, posterior predictive checks
  mvt.ipynb               the MVT stage: one gameweek's frontier and team, every gameweek's MVT and base teams
                          scored in the real FantasyGo contests, the simulated field against the real one
  structural.ipynb        runs the model for one season (SEASON in the first code cell) and evaluates it:
                          scores by position, per-gameweek RMSE, calibration, expected points per component,
                          a sample of players' predicted vs actual points, one-gameweek example, xP data check
data/
  {season}/players.csv          build_data.py
  {season}/availability.csv     build_availability.py
  23-24/fantasygo/              FantasyGo contest entries (copied from 23-24/optimizer/data: fg_manager_picks.csv,
                                fg_managers.csv); gameweeks 22-25 are excluded, see "FantasyGo data" below
  sources/                      download caches: vaastav_{season}/ (merged_gw, fixtures, teams, players_raw,
                                understat/, id_dict), football-data/ (results and odds, shared by seasons),
                                players_raw_snapshots_{season}/ (players_raw.csv at each pre-deadline commit)
outputs/
  23-24/selection/                                          selection.py: league_models.pkl (the fits),
                                                            league_select_probs.csv, league_alpha_draws.pkl
                                                            checks_gw*.joblib (selection_checks.py)
  23-24/selection_production_priors/                        the same three outputs under the notebook's priors
  23-24/mvt/mvt_{field}_r{pct}[_history_...].pkl          mvt.py's solutions, one per setting
  23-24/mvt/field_check_r{pct}.pkl                          mvt.field_check: simulated fields against the real ones
  23-24/mvt/generator_checks_gw{a}-{b}.joblib               generator_checks.py: the team generator against the real field
  23-24/mvt/generator_oracle.joblib                         generator_checks.py --oracle: the MVT with the real field
  23-24/irl/irl_{expert}_{reward}_r{pct}[_history_...].pkl  irl.py's teams, learned reward weights, training logs,
                                                            and the last networks (warm start on resume)
  {season}/structural_predictions.csv                       structural.py and the notebook
  {season}/structural_predictions_history_{seasons}.csv     the same, trained with earlier seasons (--history)
```

The folder is self-contained: it reads nothing from `23-24/` or `24-25/`. `build_data.py` is adapted from
`23-24/src/build_rolled_data.py` and gives the same columns (checked: the rebuilt 2023-24 file matches that
pipeline's all-players file in every column the model uses, and the model's predictions are unchanged).

## Run

Python 3.11 with the packages in `23-24/requirements.txt`. Scripts run from `src/` and take `--season`
(default `23-24`).

```
cd src
python build_data.py --season 22-23           # history for 2023-24 (same for build_availability.py)
python build_data.py --season 24-25           # downloads once (about a minute), then uses data/sources/
python build_availability.py --season 24-25   # needs build_data.py's fixtures.csv; downloads ~35 snapshots once
python structural.py --season 24-25           # all positions, GW 5-38 -> ../outputs/24-25/structural_predictions.csv
```

`structural.py` options: `--history 23-24` (add earlier seasons, see below), `--pos GK DEF`, `--gws 21` or `--gws 20-38`, `--targets realised` (train goals, assists
and goals conceded on realised counts instead of xG, xA and xGA), `--no-availability` (leave the FPL status out of
the minutes model), `--no-set-pieces` (leave the set-piece duties out of the goal and assist models), `--out PATH`.

Pre-deadline inputs from the `players_raw.csv` snapshots (`build_availability.py`): the FPL status and chance of
playing feed the minutes model; the penalty, corner and direct free-kick taker orders feed the goal model
(first-choice and backup penalty taker, first-choice free-kick taker) and the assist model (first-choice corner and
free-kick taker).

### Training on earlier seasons too

`--history` (`walk_forward(..., history=[...])`, `HISTORY` in the notebook) adds the listed seasons to the training
data of each gameweek. Predictions then start at GW1 (there is history before it) and are written to
`structural_predictions_history_{seasons}.csv`, next to the single-season file. Each season is trained on the one
before it:

```
python structural.py --season 23-24 --history 22-23
python structural.py --season 24-25 --history 23-24
```

| GW5-38, RMSE / R² / Spearman | season only | with the previous season |
|---|---|---|
| 2023-24, every player | 1.853 / 0.341 / 0.722 | 1.846 / 0.346 / 0.724 |
| 2023-24, players who played | 2.849 / 0.109 / 0.403 | 2.839 / 0.115 / 0.413 |
| 2024-25, every player | 1.876 / 0.337 / 0.732 | 1.870 / 0.342 / 0.736 |
| 2024-25, players who played | 2.739 / 0.108 / 0.395 | 2.732 / 0.113 / 0.402 |
| GW1-4, every player (only possible with history) | 2023-24: 2.160 / 0.174 / 0.627 | 2024-25: 1.984 / 0.288 / 0.703 |

Most of the gain is early in the season (GW5-10 RMSE 1.872 to 1.849 in 2023-24, 1.895 to 1.873 in 2024-25), and it
holds for every position; the last 10 gameweeks of the previous season give nearly all of it.

2022-23 is history only. FPL published xG, xA and xGC only from its GW16, so before that the expected-stat targets and
features are zero; `config.SEASONS["22-23"]["history_from_gw"] = 26` uses it from GW26, once the 10-match team form
features are complete too. Using all of 2022-23 instead gave 23-24 less (every player RMSE 1.851) and lowered the
Spearman for players who played (0.396). 2022-23 has no GW7 (postponed after the Queen's death) and its GW17 snapshot
is 33 days old (the World Cup break).

From Python:

```python
import structural

data = structural.load("24-25")  # load once
preds = structural.walk_forward(gws=range(5, 39), data=data)
preds_h = structural.walk_forward(
    gws=range(1, 39), data=data, history=["23-24"]
)  # with 2023-24 as well
structural.score(
    preds, ("pred",), played_only=False
)  # RMSE, R2, Spearman, MAE by position
```

To add a season: add it to `config.SEASONS` (with any gameweeks that have no target), check that the scoring rules
in `structural.SCORING` still hold (`structural.points` must reproduce `total_points`), then run the three scripts.
2025-26 added defensive contribution points, so it needs a new component before it will reproduce.

## FantasyGo data (2023-24)

The FantasyGo contest data holds other rounds' contests in gameweeks 22-25: GW22, 23 and 25 are exact copies of
GW15, 16 and 20 (same entries, managers and points), and GW24 is an Arsenal-Liverpool single-match contest whose
player points match FPL GW18 (Liverpool v Arsenal). Every other gameweek's player points match its own FPL round
(86-97% exactly; GW34 and 37 are double gameweeks, lower because FantasyGo scores them differently). The four
gameweeks are listed in `config.SEASONS["23-24"]["fantasygo_excluded_gws"]` and treated as missing: `selection.py`
leaves them out of the fits, kappa and the evaluation (its "last round's FantasyGo share" is then the latest round with
picks, GW21 for GW26), and `mvt.py` does not solve or score them and leaves them out of the team generator's
formations, budget bound and field size (`Settings.excluded_fg_gws`).

## Selection model (2023-24 FantasyGo)

`src/selection.py` is the model of how the FantasyGo managers pick players that the MVT's team generator draws the
field from: per position and round, the pick counts are Dirichlet-multinomial with $\alpha = \exp(\beta_0 + x^T\beta)$,
fitted with NUTS on the rounds before each gameweek (6-38). It is ported from `23-24/optimizer/MVT/beta
estimators.ipynb` on `data/23-24/players.csv`, which has the same rows, order and features as that notebook's input
(`xP_rolling_3`, FPL's xP over the last three gameweeks the player played, is rebuilt here and matches exactly; the
other features differ by floating-point rounding, about 1e-15). With that notebook's priors ($\beta \sim N(0, 1)$,
$\beta_0 \sim N(0, 2)$) the posteriors agree with it to three decimals, not draw for draw.

The fits now use tighter priors, $\beta \sim N(0, 0.5)$ and $\beta_0 \sim N(0, 1)$ (`selection.BETA_SD`,
`INTERCEPT_SD`). `src/selection_checks.py` (notebooks `selection_checks.ipynb`, rounds 6-10, and
`selection_checks_gw11-15.ipynb`) chose them: they match the notebook's priors on leave-future-out ELPD at rounds 6-10 and
beat them at 11-15, and their prior predictive pick shares are more plausible. Each fit records its priors, and
`selection.py` refits cached fits made under other priors. The fits under the notebook's priors are kept in
`outputs/23-24/selection_production_priors/`.

```
cd src
python selection.py           # fits what is not cached (about an hour from scratch), writes outputs/23-24/selection/
```

`notebooks/selection.ipynb` reads the fits and evaluates them. `mvt.py` reads `league_alpha_draws.pkl`, so run
`selection.py` before `mvt.py`.

## MVT stage (2023-24 FantasyGo)

`src/mvt.py` is Haugh & Singal's mean-variance optimizer as in `23-24/optimizer/MVT/` (same FantasyGo rules,
structured Sigma, team generator, Monte Carlo and MIQP), with mu_delta = this model's `pred` and Sigma built from
its errors. Every player with a fixture is a candidate for Sigma and the field (the decision uses the 30 best per
position); the old `minutes_rolling_3 > 30` filter is not needed now that `pred` includes the chance of playing.

```
cd src
python mvt.py                               # every gameweek from 6, simulated field, top 20%, 22-23 history
python mvt.py --field previous --r-pct 0.5  # the previous gameweek's real teams as the field, beat the median
```

The MIQP is solved with Gurobi (gurobipy, academic licence; `pip install gurobipy`); results are cached per setting
after every gameweek in `mvt_*_gurobi.pkl` (and `g_gaps_r{pct}_gurobi.pkl`), so a run resumes where it stopped. The
earlier results were solved with SCIP and are the files without the `_gurobi` suffix; on GW20 Gurobi returns the same
teams wherever SCIP proved optimality and better objectives where SCIP stopped at its 30 s limit. `notebooks/mvt.ipynb` reads the cache and scores the MVT team
(lambda*) and the base team (lambda = 0, highest expected points) in the real contests.

The team generator draws one selection vector per position for the whole simulated field (`Settings.shared_p`), as
the selection model's likelihood does; one per simulated team averages the field out to the mean pick shares, far
flatter than a real field. `mvt.field_check` compares both generators' fields with the real ones (`mvt.ipynb`).
The budget's lower bound comes from the earlier gameweeks' real spending (`budget_lb_quantile`). In a blank gameweek
the real managers spend less, and the earlier bound can rule out almost every team the generator draws: in GW29, when 8
clubs play, none of them. So `simulate_field` draws a pilot of `budget_pilot` teams (on its own random stream, so the
field's draws are unchanged) and, if fewer than `budget_min_accept` (10%) of them meet the bound, lowers it to the cost
that 10% reach. In 2023-24 this only binds in GW29 (bound 74.7 -> about 70; the real field's 25% quantile was 65.7) and,
for some field draws, marginally in GW30. `field_check` records both bounds per simulated field.

Teams are scored on the contest's scale (`mvt.actual_points`): the contest's own points for every player someone picked
(their pick points over the captain / vice multiplier), else FPL's total over all his fixtures. Earlier versions
kept only the first fixture of a double gameweek, so scoring on `total_points` understated the MVT team's points in
GW7, 28, 34, 35 and 37 (GW34: the real top-20% cut is 137.5 contest points, 74.8 on first-fixture points) and ranked it
near the bottom.

Double gameweeks: `build_data.py` writes one row per player and fixture (`fixture_no`, `n_fixtures`); a second fixture
gets its own opponent, venue, odds, Elo and team form, and the gameweek's lagged features (computed from first
fixtures only, so the `fixture_no == 1` rows are exactly the one-row-per-gameweek table of before). `structural.py`
trains on and predicts every fixture; `structural.per_gameweek` sums a player's fixtures into his gameweek
(`pred`, `e_*`, `total_points`; `p_play` is the larger of the two). `mvt.Inputs` reads the per-fixture file through
it, and `player_std` measures errors per match (a double gameweek's error / sqrt 2) and scales a player's sd by
sqrt(`n_fixtures`). `selection.py` keeps the first-fixture rows, so its fits are unchanged. Rescored this way the real entries reproduce the contest's top-20% cut in every gameweek; the
remaining differences are in the data (GW9's picks file misses 5 of 256 entries; GW19, 29, 31, 32: some entries' pick
points do not add up to the manager's total; GW26: some players' points differ between entries).

`src/generator_checks.py` (notebook `generator_checks.ipynb`) tests the generator against each gameweek's real entries:
pick shares, spread, team structure, the top-20% cut, G and cov(delta, G), ablations of each piece of the generator,
and an oracle MVT solved with the real field.

The selection model knows nothing about availability, and the real managers' picks play more often than the simulated
ones. So the generator multiplies the selection model's (sharpened) alphas by p_play^gamma, the structural model's chance
of playing (from the pre-deadline injury and suspension flags), at the same total so the field's concentration is
unchanged (`Settings.play_tilt`, `mvt.fit_play_tilt`). gamma is the multinomial maximum likelihood of the real picks of
the contest rounds before the gameweek given the selection model's out-of-sample shares for them (0 at GW6, about 1.3
from GW8). Out of sample it raises the log likelihood of the real picks in 86% of gameweeks (+0.03 per pick); adding
pred or a further power on the shares gained little. GW6-15: the simulated field's cut is about 3.5 points too low because the
real managers pick better (higher-xP players, and players who turn out to play), but the MVT's team barely changes with
the real field (9 of 11 players in common, same points).

## IRL optimizer (2023-24 FantasyGo, alternative to the MVT)

`src/irl.py` adapts Zhang et al. (IJCAI-25), *Heuristic-Guided Inverse Reinforcement Learning with Multi-Objective
Reward and Graph-based Policy Learning* (`IRL.pdf` in the repo root), to the FantasyGo XI. The MVT fixes the objective
(Prob{w'delta > G} via a mean-variance frontier); here the objective is learned from teams that beat the benchmark,
and a graph policy learns to pick teams that score high under it. Same inputs as the MVT: mu = `pred`, Sigma from
`mvt.build_mu_sigma`, the same shortlist, rules and contest scoring.

- Experts: `--expert contest` (default), the top 20% entries of every earlier contest (from GW3); `--expert greedy`,
  the paper's Algorithm 1 (rank by mu, at most 3 per club, skip a player whose mean correlation with the picks is
  above a threshold), made feasible for the formation and budget. The contest experts are picked on their realised
  points, but only from gameweeks before the one solved.
- Contrast (`--contrast`, contest experts): `field` (default) contrasts the top 20% with the other 80% of the same
  contest, and z-scores every gameweek against the previous contest's real entries (known at the deadline). `random`
  contrasts them with random legal teams, as the paper's background does: the reward then learns what every real
  manager does and random teams don't (weight 0.5-0.8 on spending the budget, about 0 on expected points), and its
  teams ranked near the bottom in GW6-10. Against the rest of the field it learns +expected points, -sd,
  +correlation with G and -ownership.
- `--no-policy`: the reward alone (no HGAT/PPO), decided by swap search under it from the MIQP base team and the best
  random legal teams; separates whether the learned objective helps from whether the policy optimizes it well.
  `--mean-floor x` keeps the expected points' effective weight >= x (linear reward).
- Multi-objective reward: R = sum_k beta_k f_k over four modalities, beta a softmax. *base*: expected points, sd,
  correlation with the field's cut G, mean field ownership, field captaincy of the captain, cost. *ind*: entropy of the
  club shares. *pos* / *neg*: the paper's correlation penalty / incentive with momentum (mean points over the last 3
  gameweeks). Each feature is z-scored per gameweek against random legal teams, so the weights read as reward per sd
  and compare across gameweeks. f_k linear (`--reward linear`, default, gives `coefficients` per feature) or an MLP.
- MaxEnt IRL: -E_expert[R] + log E_background[exp R] per gameweek, background = policy samples, random legal teams
  and the experts (Finn et al. 2016).
- HGAT policy: one graph per gameweek over the shortlist with edges for same club, rho > 0.03 and rho < -0.01
  (Sigma's pooled correlations are all within +-0.1, so the paper's 0.2 would leave the correlation graphs empty);
  masked multi-head attention per edge type, per-node softmax fusion, a pick and a captain logit per player. Teams are
  drawn player by player with the formation, club cap and a budget bound masked, so every sample is a legal XI.
- PPO on the learned reward, alternating with the IRL steps (150 alternations for the first gameweek, 40 for every
  later one, warm-started; 8 sampled demonstration weeks per alternation). The field of every gameweek is the latest
  earlier contest's real entries (its ownership, captaincy and the Monte Carlo cut G).
- Decision: the best of 4000 policy samples and the greedy decode under the learned reward, polished by swaps.

```
cd src
python irl.py                    # every contest gameweek from 6, contest experts, linear reward (~1 h, ds_env + torch)
python irl.py --expert greedy    # the paper's heuristic experts
```

It prints the season summary next to the cached MVT run (`irl.evaluate`, `irl.summary`).

## Head-to-head stage (2024-25 FPL H2H league)

Our team (205) in Fantasy Football Scout H2H League 1, Division 1 (20 managers), GW6-38, on the structural
predictions trained with 2023-24 (`--history 23-24`). Ported from `24-25/optimizer/notebooks/optimization.ipynb`.

```
src/
  h2h_data.py        builds data/24-25/league/ (player features for the selection model, league squads) and
                     data/24-25/ffs/ (FFS managers' weekly picks with XI, captain, bench order; managers' league tier)
  h2h_opponents.py   the opponent model: league betas + manager fits, keep/buy (eq. 3.6), sparse C-vine; fixtures
  h2h.py             base and MVT: mu = pred, Sigma = team factor on points errors, 15-man MIQP (budget, free
                     transfers, 3 per club, XI and captain), rolling season, scoring on real line-ups
  irl_h2h.py         IRL for the whole weekly decision, experts = FFS Leagues 1-3, contrast = Leagues 7-8
```

```
cd src
python h2h_data.py                 # once (needs data/sources/ffs_24-25/, from FPL-Optimal-Transfer's "FPL Data")
python h2h.py --methods base mvt   # Gurobi: a whole season in about 2 minutes
python irl_h2h.py                  # ~3 min per gameweek; --standardize rescales every feature (irl_std)
```

Inputs: the league selection model's fits stay in `24-25/optimizer/beta estimators` (`config.SEASONS["24-25"]
["league_models"]`, ~150 MB). `data/sources/ffs_24-25/FFS_manager_weekly_selections.csv` is 96 MB: do not commit it.
Real scores (ours, the opponents') come from the FFS picks: actual XI, captain, vice, bench order, auto-subs, Triple
Captain and Bench Boost; transfer hits are not in the data and are not deducted. The H2H draw is the notebook's seeded
round robin. The MIQP is solved with Gurobi (academic licence): the variance is exact (Sigma - diag has rank 20), one
model per gameweek is re-optimized for each lambda, and on GW20 it returns the same squads as the notebook's SCIP
formulation (one tie aside) in 1 s instead of 7-15 s. Result files: `h2h_{method}_r50_history_23-24_gurobi.pkl`; the
files without the suffix are the first runs, solved with SCIP (which also corrupted the heap on some models and hung).

IRL: experts are chosen before the season (the FFS pyramid's top leagues), not on the week's result as in 2023-24.
They differ from the lower leagues by a higher XI expected points (+0.4 sd), more template squads (ownership 0.44 vs
0.39), half as many hits (0.12 vs 0.25 a week) and transfers that add expected points. The decision is a search over
hold, single and double transfers and the MIQP's squads (with one hit) under the learned reward; the HGAT / PPO policy
is not used (it optimized its reward worse than search in 2023-24). The reward has no opponent term.

The XI expected points and the transfer gain are net of hits (4 points each), for the demonstrations and our
candidates alike (`FEATURES_VERSION = 2`); without it the reward put no weight on hits and the search took one in 29
of 33 gameweeks.

Season, GW6-38 (33 gameweeks) against the opponents' real scores (2082 points), Gurobi:

| | points | margin | W-D-L | H2H points | league rank | transfers (hits) |
|---|---|---|---|---|---|---|
| real (205) | 2191 | +109 | 20-0-13 | 60 | 3 | |
| base | 1801 | -281 | 11-0-22 | 33 | 19 | 33 (0) |
| mvt (r = 0.5) | 1847 | -235 | 12-0-21 | 36 | 19 | 33 (0) |
| irl | 1907 | -175 | 15-1-17 | 46 | 13 | 52 (19) |
| irl --standardize | 1844 | -238 | 11-2-20 | 35 | 19 | 51 (18) |

Hits are deducted from the IRL's points. The first runs (SCIP, features before the net-of-hits terms): base 1805, mvt
1889 (SCIP often stopped at its time limit on the nonconvex branch; the squads it returned scored better than
Gurobi's optimal ones, same 12-0-21), irl 1846 (29 hits), irl --standardize 1825 (24 hits). The real squads scored with the rule-based XI give 2180, so the real
manager's edge is the squads (transfers and chips), not line-ups; the rollouts make at most their free transfers (IRL:
plus one hit) and use no chips. IRL is ahead of base and MVT to GW21 (910 points against 811 and 832), level in
GW22-29 and behind in GW30-38 (464 against 456 and 487).

## Output columns

- `pred`: expected points, including the chance of not playing. Pick teams on this.
- `pred_if_plays`: expected points if the player plays.
- `p_play`, `p60`: probability of playing at all, and of playing 60+ minutes.
- `lam_ga`, `p_cs`: expected goals conceded by the player's team, and the clean-sheet probability.
- `e_app`, `e_cs`, `e_gc`, `e_goals`, `e_assists`, `e_bonus`, `e_saves`, `e_other`: expected points per
  component; they sum to `pred`.
- `season`, `has_target` (False in a gameweek without a target), and `total_points`, `minutes`, `played`: what
  actually happened, for scoring.

## Results (gameweeks 5-38, walk-forward)

| | 2023-24 | 2024-25 |
|---|---|---|
| Every player with a fixture: RMSE / R² / Spearman | 1.853 / 0.341 / 0.722 | 1.876 / 0.337 / 0.732 |
| Players who played: RMSE / R² / Spearman | 2.849 / 0.109 / 0.403 | 2.739 / 0.108 / 0.395 |

Goal and assist calibration: the GLMs learn xG and xA, but FPL counts more assists than xA measures (a shot saved or
deflected in, a penalty or free kick won): realised assists are about 2.1x the xA of forwards and 1.35x that of
midfielders, and defenders score about 0.85x their xG, stably over 2022-23 to 2024-25. Each goal and assist model is
rescaled by realised / predicted over its training appearances (`structural.Scaled`). Without it assists were
predicted at half (FWD) to three quarters (MID) of what was scored, and the top decile of predictions was 0.2 points
too low; now every component is within a few hundredths of a point per player-gameweek.

With the previous season in training (the deployed setting), dropping one input at a time (every player RMSE,
2023-24 / 2024-25; full model 1.846 / 1.870): without FPL status 1.858 / 1.879, without set pieces 1.850 / 1.873,
realised-count targets 1.848 / 1.872. Set pieces move first-choice penalty takers' predicted goal points per
appearance from 1.16 to 1.48 in 2023-24 (realised 1.59) and from 1.08 to 1.41 in 2024-25 (realised 1.18, an
over-correction). Pooling: with history, the pooled minutes model has 1-9% lower log loss than one per position, so
it stays pooled; a pooled bonus model was 0.5-2% worse in deviance than one per position (it helped goalkeepers only
with a single season of training), so bonus is fitted per position. Its prediction is capped at 3 (the most a match
gives): trained on one season, the goalkeeper GLM once extrapolated to 2677 for one row.

## Metric

RMSE. Squared error is minimised by the mean, which is what the optimiser needs; MAE is minimised by the median
and favours predictions that ignore hauls. Spearman is also reported, since picking a team depends on ranking
players.

## Assumptions and known gaps

- The minutes state is independent of the match result, and a player on for 60+ minutes is treated as on for the
  whole match (team clean sheet = player clean sheet). Midfield and forward clean sheets are under-predicted as a
  result (about 4-6 points of probability): a player subbed off after 60 minutes before a goal still gets one.
- Availability snapshots are a median of 44 hours (2023-24) and 39 hours (2024-25) old at the deadline, and over a
  week for GW5, 21, 22, 27, 30 (2023-24) and GW4, 8, 21, 23, 30, 33 (2024-25), so late team news there is missed.
- 2024-25: 39 players (315 appearances) have no Understat match, so their Understat features (shots, key passes,
  xG, xA) are 0; FPL's own xG and xA features still cover them.
- Set-piece orders are FPL's listed takers at the snapshot, not who actually took the kick; about 18 first-choice
  penalty takers are listed per gameweek, so a few teams list none.
- FPL's same-gameweek `xP` in the source data was recorded after the match (see the notebook's data check), so it
  is not used. It is missing in GW26 (2023-24) and GW22, 32, 34 (2024-25), where it is left empty.
