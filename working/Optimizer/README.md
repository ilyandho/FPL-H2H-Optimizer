# Blind vs MVT in the 2024-25 H2H league

Six managers, and the MVTs' r-quantile variants, play the 2024-25 FPL head-to-head season (Fantasy Football Scout H2H League 1, Division 1, our team
205) on the structural model's predictions:

- **blind**: maximizes expected points and ignores the opponent. It learns the result only after the gameweek.
- **blind_h1**, **blind_h2**: blind allowed at most 1 or 2 hits (transfers beyond the free ones, 4 points each) per
  gameweek.
- **mvt_mix**: Haugh & Singal's mean-variance optimizer, maximizing the probability of beating this week's opponent,
  with the opponent's score modelled as a mixture over the teams he could field. Free transfers only.
- **mvt_mix_h1**, **mvt_mix_h2**: the same MVT, allowed at most 1 or 2 hits per gameweek.
- **mvt_mix_r{50..90}**, **mvt_mix_h1_r{50..90}**, **mvt_mix_h2_r{50..90}**: the three MVTs with the benchmark set to
  the r-quantile (r = 0.5, 0.6, 0.7, 0.8, 0.9) of the opponent's plausible line-ups' scores instead of the mixture.
- **mvt_mix_h1_m{0,1,5,10}**, **mvt_mix_h1_r50_m{0,1,5,10}**: mvt_mix_h1 and mvt_mix_h1_r50 with the transfer margin
  (the Prob(beat) a transfer must add over holding) set to 0, 0.01, 0.05 or 0.10 instead of 0.03.

All are scored against the opponent's real score each week, reported with and without the gameweeks in which
chips were played.

## Setup

Every manager:

- starts from team 205's real GW5 squad, and plays GW6-38 (33 gameweeks) in the league's seeded round robin;
- each week keeps last week's squad and may make up to its free transfers (unused ones roll over, at most 5), plus
  up to 1 or 2 hits for the _h1 and _h2 managers (after a hit, the next week has 1 free transfer); no chips;
- pick an XI in a valid formation and a captain, within the budget (squad value at current prices plus the bank),
  at most 3 players per club;
- use the same inputs: mu = the structural model's `pred` (trained with 2023-24 history), Sigma =
  `h2h.build_mu_sigma` (a team factor on the earlier gameweeks' errors);
- are scored on real minutes and points with auto-subs and the vice rule (`h2h.lineup_points`), against the
  opponent's real score from his FFS picks (`h2h.real_points`).

The MVT knows who the opponent is at the deadline, and his squad of the previous gameweek.

**blind** solves the 15-man MIQP with lambda = 0 (`h2h.build_gurobi`) and transfers when that adds more than
`xp_margin` (0.01) expected points.

**blind_h1** and **blind_h2** solve the same MIQP once per transfer limit (free transfers + 0, 1, ... up to the maximum
hits) and play the squad with the most expected points net of its hits (4 points each), if that beats holding by
`xp_margin`.

**mvt_mix** solves Algorithm 1 of the thesis: max mu'v - sense * lambda (v'Sigma v - 2 v' cov(delta, G)) over the
lambda grid. Sense is +1 when holding's expected score beats mu_G, else -1. It plays the lambda with the highest
Prob(beat), and transfers when that beats holding by `transfer_margin` (0.03).

**mvt_mix_h1** and **mvt_mix_h2** solve the same frontier once per transfer limit (free transfers + 0, 1, ... up to
the maximum hits). Each solution's Prob(beat) is net of the hits its squad takes (4 points each, subtracted from our
mean), and the best over every limit and lambda is played. Their points are reported net of hits.

## The benchmark

The opponent fields one team, and the benchmark is his score G = w_o' delta with w_o drawn from a mixture over
(squad, XI, captain), built in `lineups.py`:

| level | candidates | weight |
|---|---|---|
| squad | his plausible squads from the opponent model (`League.opponent_squads`: league and manager selection fits, keep/buy, sparse C-vine, 20,000 samples) | share of the sampler's feasible draws; his previous squad when the model has no fit |
| XI | all 550 valid XIs of the squad | proportional to exp(XI expected points / tau_xi) |
| captain | each of the 11 starters (x2) | proportional to exp(mu_captain / tau_cap) |

tau_xi and tau_cap are fitted by maximum likelihood on the league managers' real XIs and captains in the gameweeks
before the decision, each on that gameweek's mu (Bench Boost weeks left out). Over the season tau_xi is 0.53-0.67
and tau_cap 0.50-1.01.

With row weights p_k and line-up vectors w_k:

- mu_G = sum_k p_k mu'w_k
- sigma_G^2 = sum_k p_k w_k' Sigma w_k + Var_k(mu'w_k)
- cov(delta, G) = Sigma wbar, where wbar = sum_k p_k w_k
- Prob(beat) = sum_k p_k Phi(mu'(v - w_k) / sd_k), where sd_k^2 = (v - w_k)' Sigma (v - w_k)

The opponent model gives a median of about 2,800 distinct squads a week; the most likely one holds a median 45% of
the weight. Squad s keeps its K_s = clip(p_s x 300,000, 50, 6,050) most likely (XI, captain) rows, renormalized.
The closed forms match a 20,000-draw Monte Carlo (mu_G and sd_G to 0.3 points, Prob(beat) to 0.006).

### The r-quantile benchmark

Following Lemma 3.2.2 of the thesis, the r-variants replace G by the r-quantile of the opponent's possible scores:
2,000 line-ups are drawn from the mixture rows (by p_k) and 2,000 draws of delta ~ N(mu, Sigma); in each draw G is
the r-quantile of the 2,000 line-ups' scores (`lineups.quantile_benchmark`). mu_G and cov(delta, G) are the Monte
Carlo mean and covariance, and Prob(beat) is the share of draws in which our score (net of hits) beats G
(`lineups.prob_beat_mc`). The frontier and the transfer rule are otherwise those of the mixture MVTs.

## Results (GW6-38)

The main comparison leaves out the gameweeks in which the opponent played a chip, the same weeks for the automated
managers and the real team. The season is still played through every gameweek; results with chips are given at the end of the
section.

Chips are read from the league's picks (`h2h_chips.chip_histories`): Triple Captain is a captain multiplier of 3,
Bench Boost bench multipliers of 1, Free Hit and Wildcard more than 8 new players (a Free Hit's squad reverts the next
week).

The managers are run for six league teams (`--team`), each from that team's real GW5 squad against its own
opponents, and the thesis managers for all 20 teams in the division (below); "real" is that team's real season. The opponent model uses each opponent's own manager fit where there
is one (every league manager but 205) and the league-average manager otherwise.

### Without chips (main comparison)

Totals over the 6 teams: each row adds up one manager over all six teams (the "real" row is the six teams' real
managers together). Each team has 33 gameweeks (GW6-38), 26-32 of them without an opponent's chip, 171 in all
(6 x 33 minus 27); a single team's season is in the per-team table below.

| 6 teams | gameweeks | points | points / week | W-D-L | win rate | H2H points | hits |
|---|---|---|---|---|---|---|---|
| real (6 managers) | 171 | 10755 | 62.9 | 84-7-80 | 0.49 | 259 | |
| blind | 171 | 9703 | 56.7 | 67-5-99 | 0.39 | 206 | 0 |
| blind_h1 | 171 | 10124 | 59.2 | 72-6-93 | 0.42 | 222 | 23 |
| blind_h2 | 171 | 10102 | 59.1 | 69-5-97 | 0.40 | 212 | 31 |
| mvt_mix | 171 | 9545 | 55.8 | 69-2-100 | 0.40 | 209 | 0 |
| mvt_mix_h1 | 171 | 9973 | 58.3 | 75-2-94 | 0.44 | 227 | 23 |
| mvt_mix_h2 | 171 | 9923 | 58.0 | 75-2-94 | 0.44 | 227 | 30 |

Points per gameweek between the automated managers, pooled over the teams. The difference is the mean of the
paired per-week differences (same team, same opponent, same gameweek) and t = mean / (sd / sqrt(n)) over the n team-weeks;
p is the block bootstrap's (below); "teams ahead" counts the teams in which the first manager has the higher mean:

| | difference | t | bootstrap p | teams ahead |
|---|---|---|---|---|
| mvt_mix - blind | -0.92 | -0.88 | 0.33 | 3/6 |
| mvt_mix_h1 - blind | +1.58 | +1.51 | 0.17 | 3/6 |
| mvt_mix_h2 - blind | +1.29 | +1.21 | 0.19 | 2/6 |
| mvt_mix_h1 - mvt_mix | +2.50 | +2.20 | 0.08 | 4/6 |
| mvt_mix_h2 - mvt_mix | +2.21 | +1.97 | 0.10 | 4/6 |
| mvt_mix_h2 - mvt_mix_h1 | -0.29 | -0.64 | 0.30 | 3/6 |
| blind_h1 - blind | +2.46 | +2.57 | 0.015 | 5/6 |
| blind_h2 - blind | +2.33 | +2.52 | 0.018 | 5/6 |
| blind_h2 - blind_h1 | -0.13 | -0.31 | 0.80 | 3/6 |
| mvt_mix_h1 - blind_h1 | -0.88 | -1.02 | 0.26 | 4/6 |
| mvt_mix_h2 - blind_h2 | -1.05 | -1.30 | 0.10 | 3/6 |

Hits are worth about 2.5 points a week to either manager: blind_h1 is 2.46 ahead of blind (p = 0.015), mvt_mix_h1
2.50 ahead of mvt_mix (p = 0.08). A second hit adds nothing. At the same number of hits the MVT is not ahead of blind
(mvt_mix_h1 - blind_h1 -0.88, p = 0.26); its lead over blind comes from the hits.

Per team (points / H2H points / wins):

| team | weeks | blind | blind_h1 | blind_h2 | mvt_mix | mvt_mix_h1 | mvt_mix_h2 | real |
|---|---|---|---|---|---|---|---|---|
| 205 | 26 | 1399 / 27 / 9 | 1599 / 41 / 13 | 1550 / 37 / 12 | 1440 / 27 / 9 | 1616 / 42 / 14 | 1610 / 45 / 15 | 1711 / 45 / 15 |
| 38351 | 30 | 1543 / 29 / 9 | 1735 / 31 / 10 | 1759 / 31 / 10 | 1616 / 28 / 9 | 1777 / 42 / 14 | 1799 / 42 / 14 | 1909 / 52 / 17 |
| 69718 | 32 | 1960 / 49 / 16 | 1869 / 36 / 12 | 1886 / 36 / 12 | 2019 / 57 / 19 | 1877 / 42 / 14 | 1909 / 45 / 15 | 2033 / 52 / 16 |
| 593532 | 29 | 1624 / 40 / 13 | 1629 / 40 / 13 | 1660 / 40 / 13 | 1602 / 39 / 13 | 1575 / 36 / 12 | 1563 / 39 / 13 | 1891 / 55 / 18 |
| 696127 | 27 | 1634 / 28 / 9 | 1698 / 37 / 12 | 1674 / 37 / 12 | 1388 / 19 / 6 | 1529 / 25 / 8 | 1545 / 25 / 8 | 1601 / 19 / 6 |
| 749244 | 27 | 1543 / 33 / 11 | 1594 / 37 / 12 | 1573 / 31 / 10 | 1480 / 39 / 13 | 1599 / 40 / 13 | 1497 / 31 / 10 | 1610 / 36 / 12 |

Managers ranked by points within each team (points / W-D-L / H2H points; every manager in a team plays the same
gameweeks):

| team (weeks) | 1st | 2nd | 3rd | 4th | 5th | 6th | 7th |
|---|---|---|---|---|---|---|---|
| 205 (26) | real 1711 / 15-0-11 / 45 | mvt_mix_h1 1616 / 14-0-12 / 42 | mvt_mix_h2 1610 / 15-0-11 / 45 | blind_h1 1599 / 13-2-11 / 41 | blind_h2 1550 / 12-1-13 / 37 | mvt_mix 1440 / 9-0-17 / 27 | blind 1399 / 9-0-17 / 27 |
| 38351 (30) | real 1909 / 17-1-12 / 52 | mvt_mix_h2 1799 / 14-0-16 / 42 | mvt_mix_h1 1777 / 14-0-16 / 42 | blind_h2 1759 / 10-1-19 / 31 | blind_h1 1735 / 10-1-19 / 31 | mvt_mix 1616 / 9-1-20 / 28 | blind 1543 / 9-2-19 / 29 |
| 69718 (32) | real 2033 / 16-4-12 / 52 | mvt_mix 2019 / 19-0-13 / 57 | blind 1960 / 16-1-15 / 49 | mvt_mix_h2 1909 / 15-0-17 / 45 | blind_h2 1886 / 12-0-20 / 36 | mvt_mix_h1 1877 / 14-0-18 / 42 | blind_h1 1869 / 12-0-20 / 36 |
| 593532 (29) | real 1891 / 18-1-10 / 55 | blind_h2 1660 / 13-1-15 / 40 | blind_h1 1629 / 13-1-15 / 40 | blind 1624 / 13-1-15 / 40 | mvt_mix 1602 / 13-0-16 / 39 | mvt_mix_h1 1575 / 12-0-17 / 36 | mvt_mix_h2 1563 / 13-0-16 / 39 |
| 696127 (27) | blind_h1 1698 / 12-1-14 / 37 | blind_h2 1674 / 12-1-14 / 37 | blind 1634 / 9-1-17 / 28 | real 1601 / 6-1-20 / 19 | mvt_mix_h2 1545 / 8-1-18 / 25 | mvt_mix_h1 1529 / 8-1-18 / 25 | mvt_mix 1388 / 6-1-20 / 19 |
| 749244 (27) | real 1610 / 12-0-15 / 36 | mvt_mix_h1 1599 / 13-1-13 / 40 | blind_h1 1594 / 12-1-14 / 37 | blind_h2 1573 / 10-1-16 / 31 | blind 1543 / 11-0-16 / 33 | mvt_mix_h2 1497 / 10-1-16 / 31 | mvt_mix 1480 / 13-0-14 / 39 |

Rank by points in each team, and the mean over the six teams:

| | 205 | 38351 | 69718 | 593532 | 696127 | 749244 | mean rank |
|---|---|---|---|---|---|---|---|
| real | 1 | 1 | 1 | 1 | 4 | 1 | 1.50 |
| blind_h2 | 5 | 4 | 5 | 2 | 2 | 4 | 3.67 |
| blind_h1 | 4 | 5 | 7 | 3 | 1 | 3 | 3.83 |
| mvt_mix_h1 | 2 | 3 | 6 | 6 | 6 | 2 | 4.17 |
| mvt_mix_h2 | 3 | 2 | 4 | 7 | 5 | 6 | 4.50 |
| blind | 7 | 7 | 3 | 4 | 3 | 5 | 4.83 |
| mvt_mix | 6 | 6 | 2 | 5 | 7 | 7 | 5.50 |

Team 205 in detail (26 gameweeks, opponents' real points 1612; the chips are in the table below):

| | points | margin | W-D-L | H2H points | hits | win rate | mean P(beat) |
|---|---|---|---|---|---|---|---|
| real (205) | 1711 | +99 | 15-0-11 | 45 | | 0.58 | |
| blind | 1399 | -213 | 9-0-17 | 27 | 0 | 0.35 | 0.52 |
| blind_h1 | 1599 | -13 | 13-2-11 | 41 | 2 | 0.50 | 0.57 |
| blind_h2 | 1550 | -62 | 12-1-13 | 37 | 5 | 0.46 | 0.56 |
| mvt_mix | 1440 | -172 | 9-0-17 | 27 | 0 | 0.35 | 0.51 |
| mvt_mix_h1 | 1616 | +4 | 14-0-12 | 42 | 3 | 0.54 | 0.58 |
| mvt_mix_h2 | 1610 | -2 | 15-0-11 | 45 | 4 | 0.58 | 0.59 |

| GW | opponent | team 205 |
|---|---|---|
| 6 | Wildcard | Wildcard |
| 30 | Wildcard | Wildcard |
| 31 | Wildcard | |
| 32 | Bench Boost | Triple Captain |
| 33 | Bench Boost | Bench Boost |
| 34 | Free Hit | Free Hit |
| 36 | Triple Captain | |

P(beat) is measured against the mixture benchmark for every manager, net of hits.

### Hits (team 205, without chips)

The hit-taking MVTs' gameweeks by the number of hits taken (points net of hits; the last column against mvt_mix in
the same gameweeks):

| | hits | gameweeks | mean points (gross) | mean points (net) | win rate | per week vs mvt_mix |
|---|---|---|---|---|---|---|
| mvt_mix_h1 | 0 | 23 | 63 | 63 | 0.57 | +7.0 |
| | 1 | 3 | 61 | 57 | 0.33 | +5.0 |
| mvt_mix_h2 | 0 | 23 | 63 | 63 | 0.61 | +6.8 |
| | 1 | 2 | 69 | 65 | 0.50 | +16.0 |
| | 2 | 1 | 46 | 38 | 0.00 | -19.0 |

Over the whole season mvt_mix_h1 took its hits in GW7, 18, 29, 31, 32, 33 and 34; mvt_mix_h2 took one in GW7, 18, 31
and 34 and two in GW29, 32 and 33. The two are identical up to GW28. Their lead over mvt_mix is 30 points at GW10, 147
at GW15 and 196 at GW20; by GW20 their squads share 8 of 15 players with mvt_mix's.

### Benchmark quantile r (without chips)

The three MVTs with the mixture benchmark ("mix") and the r-quantile benchmarks, over the same 171 team-weeks. Mean
rank is by points within each team among all 22 rows (21 automated managers, the blind ones included, and real),
averaged over the six teams.
The comparisons are per gameweek, paired (block bootstrap p in brackets):

| hits | benchmark | points / week | W-D-L | H2H points | mean rank | vs real | vs blind | vs mix |
|---|---|---|---|---|---|---|---|---|
| 0 | mix | 55.82 | 69-2-100 | 209 | 15.50 | -7.08 (0.023) | -0.92 (0.33) |  |
| 0 | r = 0.5 | 56.15 | 67-3-101 | 204 | 13.67 | -6.74 (0.015) | -0.59 (0.54) | +0.33 (0.71) |
| 0 | r = 0.6 | 55.40 | 63-4-104 | 193 | 18.33 | -7.50 (0.007) | -1.35 (0.16) | -0.42 (0.67) |
| 0 | r = 0.7 | 55.47 | 68-4-99 | 208 | 16.33 | -7.42 (0.020) | -1.27 (0.17) | -0.35 (0.77) |
| 0 | r = 0.8 | 55.05 | 65-2-104 | 197 | 16.83 | -7.85 (0.013) | -1.70 (0.051) | -0.77 (0.36) |
| 0 | r = 0.9 | 55.70 | 66-3-102 | 201 | 13.50 | -7.20 (0.012) | -1.05 (0.28) | -0.12 (0.93) |
| 1 | mix | 58.32 | 75-2-94 | 227 | 9.67 | -4.57 (0.11) | +1.58 (0.17) |  |
| 1 | r = 0.5 | 59.50 | 73-6-92 | 225 | 7.50 | -3.40 (0.18) | +2.75 (0.011) | +1.18 (0.070) |
| 1 | r = 0.6 | 58.16 | 72-8-91 | 224 | 9.33 | -4.74 (0.11) | +1.42 (0.083) | -0.16 (0.83) |
| 1 | r = 0.7 | 57.20 | 67-6-98 | 207 | 11.33 | -5.69 (0.087) | +0.46 (0.62) | -1.12 (0.30) |
| 1 | r = 0.8 | 58.39 | 74-4-93 | 226 | 9.17 | -4.50 (0.13) | +1.65 (0.13) | +0.07 (0.96) |
| 1 | r = 0.9 | 58.08 | 74-3-94 | 225 | 9.17 | -4.81 (0.13) | +1.34 (0.33) | -0.24 (0.82) |
| 2 | mix | 58.03 | 75-2-94 | 227 | 10.83 | -4.87 (0.086) | +1.29 (0.19) |  |
| 2 | r = 0.5 | 58.81 | 73-6-92 | 225 | 8.50 | -4.08 (0.11) | +2.07 (0.034) | +0.78 (0.18) |
| 2 | r = 0.6 | 56.95 | 69-5-97 | 212 | 12.83 | -5.94 (0.042) | +0.21 (0.83) | -1.08 (0.016) |
| 2 | r = 0.7 | 56.19 | 65-8-98 | 203 | 16.33 | -6.71 (0.038) | -0.56 (0.60) | -1.84 (0.028) |
| 2 | r = 0.8 | 57.56 | 71-4-96 | 217 | 12.83 | -5.33 (0.070) | +0.82 (0.47) | -0.47 (0.67) |
| 2 | r = 0.9 | 57.78 | 74-2-95 | 224 | 10.00 | -5.11 (0.12) | +1.04 (0.50) | -0.25 (0.84) |

The real managers score 62.89 a week (mean rank 2.00); blind_h1 and blind_h2 score 59.20 and 59.08 (mean rank 8.50
and 8.33). mvt_mix_h1_r50 is the best automated manager by points and by mean rank, but only 0.29 a week ahead of
blind_h1 (p = 0.75, below). It scores more than the real manager in 82 of the 171 weeks (89 fewer) and more than blind in 96 (68
fewer, 7 equal), and it finishes the season ahead of team 69718's real manager (2111 to 2033 points, 58 to 52 H2H
points). The r = 0.5 benchmark is the only one ahead of the mixture at every hit level.

#### Block bootstrap

Rolling squads make neighbouring gameweeks dependent, and the automated managers share the predictions, so a week
they miss, they miss in every team. The paired differences are therefore tested with a circular block bootstrap over
gameweeks (`block_bootstrap` in `baseline.py`): blocks of 4 consecutive gameweeks are drawn with wrap-around, the same
gameweeks for every team, and the pooled mean is taken over the team-weeks drawn; 10,000 draws. The bootstrap t is the
mean over the bootstrap standard error, p the two-sided share of centred bootstrap means at least as far from the
mean as the mean is from 0. The lag-1 autocorrelation of the differences is -0.17 to 0.21. Between teams, the
differences against the real managers are correlated 0.34-0.58 in the same gameweek, the differences between
automated managers -0.11 to 0.25. Blocks of 2 or 8 gameweeks move the bootstrap t by up to about 0.5.

| comparison | per week | t | bootstrap t | 95% interval | p |
|---|---|---|---|---|---|
| mvt_mix_h1_r50 - blind | +2.75 | 2.57 | 2.56 | [+0.69, +4.87] | 0.011 |
| mvt_mix_h1_r50 - mvt_mix_h1 | +1.18 | 1.67 | 1.82 | [-0.10, +2.47] | 0.070 |
| mvt_mix_h1_r50 - real | -3.40 | -2.28 | -1.35 | [-8.34, +1.40] | 0.18 |
| mvt_mix_h2_r50 - blind | +2.07 | 1.98 | 2.13 | [+0.21, +4.02] | 0.034 |
| mvt_mix_h1 - blind | +1.58 | 1.51 | 1.38 | [-0.70, +3.77] | 0.17 |
| mvt_mix_h1 - real | -4.57 | -3.06 | -1.61 | [-10.17, +0.83] | 0.11 |
| mvt_mix_h2_r70 - mvt_mix_h2 | -1.84 | -2.29 | -2.15 | [-3.51, -0.17] | 0.028 |
| blind - real | -6.15 | -3.89 | -2.03 | [-12.18, -0.32] | 0.042 |
| 0-hit MVTs - real | -6.7 to -7.9 | -4.4 to -5.2 | -2.3 to -2.7 | all below 0 | 0.007-0.023 |
| 0-hit MVTs - blind | -0.6 to -1.7 | -0.6 to -1.6 | -0.6 to -2.0 | all contain 0 | 0.05-0.54 |
| blind_h1 - blind | +2.46 | 2.57 | 2.48 | [+0.33, +4.27] | 0.015 |
| blind_h2 - blind | +2.33 | 2.52 | 2.34 | [+0.35, +4.20] | 0.018 |
| mvt_mix_h1_r50 - mvt_mix_r50 | +3.35 | 3.33 | 3.49 | [+1.35, +5.08] | 0.001 |
| mvt_mix_h2_r50 - mvt_mix_r50 | +2.66 | 2.64 | 2.54 | [+0.38, +4.51] | 0.012 |
| mvt_mix_r50 - blind | -0.59 | -0.56 | -0.62 | [-2.29, +1.43] | 0.54 |
| mvt_mix_h1_r50 - blind_h1 | +0.29 | 0.30 | 0.32 | [-1.36, +2.21] | 0.75 |
| mvt_mix_h2_r50 - blind_h2 | -0.26 | -0.28 | -0.27 | [-2.03, +1.78] | 0.80 |
| mvt_mix_h1 - blind_h1 | -0.88 | -1.02 | -1.11 | [-2.43, +0.66] | 0.26 |
| blind_h1 - real | -3.69 | -2.46 | -1.19 | [-9.89, +2.22] | 0.24 |
| blind_h2 - real | -3.82 | -2.62 | -1.24 | [-9.86, +2.13] | 0.22 |

Hits are the one clear effect: allowing them raises blind by 2.3-2.5 points a week (p 0.015-0.018) and the r = 0.5
MVT by 2.7-3.4 (p 0.001-0.012). At the same number of hits the MVT and blind are level (mvt_mix_r50 - blind -0.59,
mvt_mix_h1_r50 - blind_h1 +0.29, mvt_mix_h2_r50 - blind_h2 -0.26; p 0.54-0.80), so the MVT's leads over blind in the
tables above (mvt_mix_h1_r50 +2.75, p = 0.011) come from its hits, not from the benchmark. Blind and the 0-hit MVTs are
significantly behind the real managers (p 0.007-0.042); the managers with hits are not (blind_h1, blind_h2 and the
one-hit MVTs p 0.09-0.24; the two-hit MVTs too, except r = 0.6 and 0.7, p 0.04).

### All 20 league teams

The thesis managers (blind, blind_h1, blind_h2 and the r = 0.5 MVTs mvt_mix_r50, mvt_mix_h1_r50, mvt_mix_h2_r50) for
every team in the division, each from that team's real GW5 squad against its own opponents (`--compare-teams
--league`). The managers are built to win the week, so they are compared on wins and H2H points (3 a win, 1 a draw).

Totals over the 20 teams, without chips (568 team-weeks) and with chips (660):

| 20 teams | W-D-L | win rate | H2H points | points / week | W-D-L with chips | H2H points with chips |
|---|---|---|---|---|---|---|
| real (20 managers) | 284-16-268 | 0.50 | 868 | 63.85 | 320-20-320 | 980 |
| blind | 202-14-352 | 0.36 | 620 | 55.35 | 237-14-409 | 725 |
| blind_h1 | 228-16-324 | 0.40 | 700 | 59.04 | 257-16-387 | 787 |
| blind_h2 | 224-11-333 | 0.39 | 683 | 58.66 | 253-11-396 | 770 |
| mvt_mix_r50 | 208-9-351 | 0.37 | 633 | 55.95 | 251-11-398 | 764 |
| mvt_mix_h1_r50 | 248-13-307 | 0.44 | 757 | 59.02 | 282-16-362 | 862 |
| mvt_mix_h2_r50 | 235-15-318 | 0.41 | 720 | 58.34 | 270-17-373 | 827 |

H2H points per gameweek, paired (same team, opponent and gameweek), with the block bootstrap over gameweeks jointly
for the 20 teams. "Per team-season" is the total over the teams divided by 20; "weeks better" counts the team-weeks
with a better result, the same and a worse one:

| comparison | per week | per team-season | weeks better | teams ahead | p | with chips: per week | p |
|---|---|---|---|---|---|---|---|
| mvt_mix_r50 - blind | +0.023 | +0.7 | 67-434-67 | 11/20 | 0.63 | +0.059 | 0.22 |
| mvt_mix_h1_r50 - blind_h1 | +0.100 | +2.9 | 66-451-51 | 11/20 | 0.041 | +0.114 | 0.006 |
| mvt_mix_h2_r50 - blind_h2 | +0.065 | +1.9 | 64-452-52 | 12/20 | 0.093 | +0.086 | 0.015 |
| blind_h1 - blind | +0.141 | +4.0 | 82-437-49 | 13/20 | 0.13 | +0.094 | 0.31 |
| blind_h2 - blind | +0.111 | +3.2 | 79-435-54 | 10/20 | 0.24 | +0.068 | 0.45 |
| mvt_mix_h1_r50 - mvt_mix_r50 | +0.218 | +6.2 | 94-423-51 | 14/20 | 0.018 | +0.148 | 0.12 |
| mvt_mix_h2_r50 - mvt_mix_r50 | +0.153 | +4.4 | 86-426-56 | 12/20 | 0.080 | +0.095 | 0.29 |
| blind - real | -0.437 | -12.4 | 76-331-161 | 2/20 | 0.001 | -0.386 | 0.003 |
| blind_h1 - real | -0.296 | -8.4 | 88-335-145 | 4/20 | 0.088 | -0.292 | 0.086 |
| blind_h2 - real | -0.326 | -9.3 | 82-342-144 | 3/20 | 0.059 | -0.318 | 0.061 |
| mvt_mix_r50 - real | -0.414 | -11.8 | 73-340-155 | 1/20 | 0.001 | -0.327 | 0.008 |
| mvt_mix_h1_r50 - real | -0.195 | -5.6 | 93-343-132 | 6/20 | 0.22 | -0.179 | 0.26 |
| mvt_mix_h2_r50 - real | -0.261 | -7.4 | 89-340-139 | 4/20 | 0.11 | -0.232 | 0.15 |

The same comparisons in points per gameweek:

| comparison | per week | p | with chips: per week | p |
|---|---|---|---|---|
| mvt_mix_r50 - blind | +0.59 | 0.23 | +0.75 | 0.11 |
| mvt_mix_h1_r50 - blind_h1 | -0.01 | 0.98 | +0.38 | 0.52 |
| mvt_mix_h2_r50 - blind_h2 | -0.32 | 0.57 | +0.03 | 0.96 |
| blind_h1 - blind | +3.68 | 0.002 | +3.04 | 0.012 |
| blind_h2 - blind | +3.31 | 0.002 | +2.89 | 0.006 |
| mvt_mix_h1_r50 - mvt_mix_r50 | +3.08 | < 0.001 | +2.66 | 0.001 |
| mvt_mix_h2_r50 - mvt_mix_r50 | +2.40 | 0.004 | +2.16 | 0.005 |
| blind - real | -8.49 | 0.001 | -8.32 | 0.001 |
| blind_h1 - real | -4.81 | 0.070 | -5.29 | 0.044 |
| blind_h2 - real | -5.19 | 0.050 | -5.44 | 0.036 |
| mvt_mix_r50 - real | -7.90 | 0.001 | -7.57 | 0.001 |
| mvt_mix_h1_r50 - real | -4.82 | 0.049 | -4.91 | 0.042 |
| mvt_mix_h2_r50 - real | -5.50 | 0.025 | -5.41 | 0.025 |

In wins, the MVT with hits beats blind with the same hits: mvt_mix_h1_r50 takes 57 more H2H points than blind_h1 over
the 20 teams (+2.9 a team-season, p = 0.041; with chips 75, p = 0.006), and mvt_mix_h2_r50 37 more than blind_h2
(p = 0.093; with chips p = 0.015), while scoring the same points (p 0.57-0.98). The MVT turns the same points into
more wins. Without hits it is level with blind in wins too (p = 0.63). Hits raise every manager's points by 2.4-3.7 a
week (p ≤ 0.012) and the MVT's wins by 4-6 H2H points a team-season (mvt_mix_h1_r50 - mvt_mix_r50 p = 0.018); blind's
extra points from hits do not turn into significantly more wins (p 0.13-0.24). Against the real managers, blind and
mvt_mix_r50 win significantly less (p ≤ 0.008); the managers with hits are not significantly behind in wins
(p 0.06-0.26), mvt_mix_h1_r50 closest at 5.6 H2H points a team-season. Between teams the H2H differences against
the real managers are correlated 0.10-0.19 in the same gameweek, those between automated managers -0.01 to 0.06.

### Transfer margin (without chips)

mvt_mix_h1 and mvt_mix_h1_r50 with the minimum Prob(beat) gain over holding at 0, 0.01, 0.03 (the managers above),
0.05 and 0.10, over the same 171 team-weeks. Transfers and hits are summed over those weeks; the differences are per
gameweek, with the block bootstrap p in brackets:

| manager | margin | points / week | W-D-L | H2H points | transfers | hits | vs real (p) | vs blind (p) | vs 0.03 (p) | teams ahead of 0.03 |
|---|---|---|---|---|---|---|---|---|---|---|
| mvt_mix_h1 | 0, 0.01 | 57.94 | 72-1-98 | 217 | 196 | 21 | -4.95 (0.09) | +1.20 (0.31) | -0.38 (0.24) | 0/6 |
| mvt_mix_h1 | 0.03 | 58.32 | 75-2-94 | 227 | 198 | 23 | -4.57 (0.11) | +1.58 (0.17) |  |  |
| mvt_mix_h1 | 0.05 | 57.50 | 69-2-100 | 209 | 194 | 19 | -5.40 (0.08) | +0.75 (0.55) | -0.82 (0.27) | 3/6 |
| mvt_mix_h1 | 0.10 | 58.19 | 70-7-94 | 217 | 185 | 11 | -4.71 (0.11) | +1.44 (0.39) | -0.13 (0.93) | 4/6 |
| mvt_mix_h1_r50 | 0, 0.01 | 59.50 | 76-6-89 | 234 | 195 | 20 | -3.39 (0.17) | +2.76 (0.01) | +0.01 (0.99) | 2/6 |
| mvt_mix_h1_r50 | 0.03 | 59.50 | 73-6-92 | 225 | 199 | 24 | -3.40 (0.18) | +2.75 (0.01) |  |  |
| mvt_mix_h1_r50 | 0.05 | 58.67 | 74-4-93 | 226 | 188 | 17 | -4.23 (0.11) | +1.92 (0.17) | -0.83 (0.23) | 3/6 |
| mvt_mix_h1_r50 | 0.10 | 58.39 | 72-4-95 | 220 | 185 | 13 | -4.51 (0.09) | +1.64 (0.32) | -1.11 (0.34) | 2/6 |

Margins 0 and 0.01 play the same season: no transfer the MVT proposed added between 0 and 0.01. A higher margin
mainly cuts hits (at 0.10 about half), and no margin is significantly better than 0.03.

### With chips

Totals over the 6 teams, all 33 gameweeks per team (198 in all):

| 6 teams | points | H2H points | wins | per week vs mvt_mix | t | teams ahead |
|---|---|---|---|---|---|---|
| real (6 managers) | 12701 | 299 | 97 | | | |
| blind | 11565 | 239 | 78 | +1.24 | +1.26 | 3/6 |
| blind_h1 | 11964 | 249 | 81 | +3.25 | +2.94 | 5/6 |
| blind_h2 | 11925 | 239 | 78 | +3.06 | +2.78 | 5/6 |
| mvt_mix | 11320 | 242 | 80 | | | |
| mvt_mix_h1 | 11764 | 251 | 83 | +2.24 | +2.10 | 5/6 |
| mvt_mix_h2 | 11778 | 254 | 84 | +2.31 | +2.15 | 5/6 |

Team 205 (opponents' real points 2082):

| | points | margin | W-D-L | H2H points | hits | win rate |
|---|---|---|---|---|---|---|
| real (205) | 2191 | +109 | 20-0-13 | 60 | | 0.61 |
| blind | 1801 | -281 | 11-0-22 | 33 | 0 | 0.33 |
| blind_h1 | 2023 | -59 | 15-2-16 | 47 | 5 | 0.45 |
| blind_h2 | 1973 | -109 | 14-1-18 | 43 | 12 | 0.42 |
| mvt_mix | 1878 | -204 | 12-0-21 | 36 | 0 | 0.36 |
| mvt_mix_h1 | 2049 | -33 | 17-0-16 | 51 | 7 | 0.52 |
| mvt_mix_h2 | 2031 | -51 | 17-0-16 | 51 | 10 | 0.52 |

## Files and run

```
baseline.py     the season: the managers (blind with hits and the r variants included), the MIQP decisions, hits, scoring, the summaries,
                the r table and the block bootstrap (--compare-teams)
lineups.py      the benchmark: XI and captain enumeration, the temperature fit (LineupModel),
                benchmark() (moments and rows), prob_beat() (mixture probability),
                quantile_benchmark() and prob_beat_mc() (the r-quantile benchmark)
outputs/
  rolling_h2h.pkl       per-manager, per-gameweek rows (squad, line-up, transfers, pred, P(beat), points,
                        opponent points, the mixture's mu_G and sd_G and fit details)
  rolling_h2h_team{id}.pkl   the same for the other teams (--team)
  compare_teams.txt          --compare-teams output (four managers)
  compare_teams_r.txt        --compare-teams output with the r variants, the r table, the block bootstrap and the
                             transfer-margin table
  compare_teams_league.txt   --compare-teams --league output: the thesis managers over the 20 league teams, in
                             points and in H2H points
  run_*.log                  the runs
```

The scripts use `playground/predictors/structural/src` (`h2h.py`, `h2h_opponents.py`, `config.py`) with its data
and outputs. They need the `ds_env` environment (Gurobi, pyvinecopulib) and the structural predictions
`structural_predictions_history_23-24.csv`.

```
python baseline.py                              # GW6-38, resumes from outputs/rolling_h2h.pkl
python baseline.py --methods mvt_mix --recompute
python baseline.py --gws 6-10
python baseline.py --team 38351                 # another league team, outputs/rolling_h2h_team38351.pkl
python baseline.py --team 38351 --methods mvt_mix_h1_r50 --threads 2   # one r variant
python baseline.py --compare-teams              # every team's results from the caches, no solving
python baseline.py --compare-teams --league     # the thesis managers over the 20 league teams
```

A full season takes about 10 minutes.
