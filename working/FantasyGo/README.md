# Base vs MVT in the 2023-24 FantasyGo contests

Haugh & Singal's mean-variance optimizer (MVT) on the 2023-24 FantasyGo round contests, against the team with the
highest expected points, scored in the real contests. Three kinds of entry:

- **base**: the team with the highest expected points (lambda = 0).
- **mvt_r{50,80,90,99,100}**: the MVT, maximizing the probability of beating G^(r'), the simulated field's score at
  rank r (r = 0.5, 0.8, 0.9, 0.99; r = 1.0 is the field's best score).
- **portfolio**: Haugh & Singal's multiple-entry portfolio (Algorithm 7) with up to 40 entries a week: the
  "strategic" chain (the MVT) and the "benchmark" chain (the highest expected points, diversified the same way),
  scored for N = 1, 5, 10, 20 and 40 entries.

## Setup

FantasyGo: 11 players (1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD), budget 80, at most 3 per club, captain x2 and vice-captain
x1.5, unlimited changes each gameweek, entry fee R20.

The teams are solved by the structural pipeline (`playground/predictors/structural/src`):

- mu = the structural model's expected points (`pred`, with 2022-23 history), summed over a double gameweek's fixtures;
- Sigma = diag(sigma) rho diag(sigma): sigma from each player's past errors, rho one correlation per (same club /
  opponents, position pair), pooled over earlier gameweeks;
- W_o = a simulated field from the team generator and the league selection model (Dirichlet-multinomial fits on the
  earlier rounds' picks), tilted towards players likely to play and towards double-gameweek players;
- G^(r') by Monte Carlo (2,000 draws), shifted and widened by the earlier gameweeks' gaps between the simulated and
  the real cut;
- per lambda in the grid, the MIQP max w'mu -/+ lambda (w'Sigma w - 2 w'cov(delta, G)) (Gurobi) on the 30 best-predicted
  players per position; lambda* is the lambda whose team beats G^(r') in most draws (`mvt.py`).

The portfolio adds entries one at a time: per lambda, the MIQP with the risk-seeking sign, G at r = 0.99, and at most 8
players shared with each entry chosen so far; lambda* is the lambda that adds most to the portfolio's expected prize
(`mvt_portfolio.py`).

Every gameweek with a usable contest from GW6: GW6-21 and 26-38, 29 in all. GW22-25 hold other rounds' contests and
are left out.

## Scoring

Each entry is scored on the contest's points (`mvt.actual_points`) and inserted into the real contest: rank = 1 + the
real entries with at least as many points (a tie goes to the real entry) + our other entries with more points.

- **rank pct**: rank / the contest's entries (ours included).
- **beat r**: the share of weeks with more points than the real field's score at rank r; r = 1.0 is winning outright.
- **prize**: the week's real prize schedule at our rank. **prize top10**: FantasyGo's top-10 payout (40, 25, 12.5,
  7.5, 5, 3, 2.5, 2, 1.5, 1 % of the pool), our entries' R17.8125 each added to the pool in GW2-21. **profit** = prize -
  R20 per entry.
- **avg entrant**: the pool per real entry, what an average entry got back (about R45 a week); "prize / avg entrant"
  is our prize over what the same number of average entries got.

## Results (29 gameweeks)

### Single entries

| | base | mvt_r50 | mvt_r80 | mvt_r90 | mvt_r99 | mvt_r100 |
|---|---|---|---|---|---|---|
| points / week | 65.7 | 64.6 | 63.7 | 60.7 | 61.6 | 59.6 |
| median rank pct | 0.47 | 0.41 | 0.51 | 0.58 | 0.42 | 0.48 |
| mean rank pct | 0.44 | 0.46 | 0.46 | 0.54 | 0.48 | 0.53 |
| beat r = 0.5 | 0.52 | 0.59 | 0.48 | 0.38 | 0.59 | 0.55 |
| beat r = 0.8 | 0.28 | 0.14 | 0.24 | 0.17 | 0.17 | 0.14 |
| beat r = 0.9 | 0.14 | 0.03 | 0.14 | 0.10 | 0.14 | 0.07 |
| beat r = 0.99 | 0.03 | 0.03 | 0 | 0.03 | 0 | 0 |
| top 10 / wins | 2 / 1 | 1 / 1 | 1 / 0 | 2 / 1 | 0 / 0 | 0 / 0 |
| prize | 9734 | 9680 | 127 | 9734 | 98 | 49 |
| profit (cost 580) | +9154 | +9100 | -453 | +9154 | -482 | -531 |
| profit top10 | +5363 | +5058 | -439 | +5275 | -580 | -580 |
| prize / avg entrant | 7.42 | 7.38 | 0.10 | 7.42 | 0.07 | 0.04 |

MVT - base per gameweek (paired t; circular block bootstrap over the 29 weeks, blocks of 4, 10,000 draws):

| | points / week | t | bootstrap p | rank pct | bootstrap p | weeks better / same |
|---|---|---|---|---|---|---|
| mvt_r50 | -1.12 | -0.75 | 0.23 | +0.022 | 0.45 | 10 / 5 |
| mvt_r80 | -2.00 | -1.85 | 0.011 | +0.029 | 0.15 | 5 / 14 |
| mvt_r90 | -5.00 | -2.81 | < 0.001 | +0.103 | 0.002 | 5 / 9 |
| mvt_r99 | -4.09 | -1.32 | 0.10 | +0.049 | 0.30 | 13 / 1 |
| mvt_r100 | -6.10 | -1.77 | 0.035 | +0.094 | 0.07 | 13 / 1 |

(Rank pct: a positive difference is a worse rank. "Weeks better" counts the weeks with more points.)

The single-entry MVT gives up expected points for variance, more the higher r, and finishes no better than the base
team; only mvt_r50 beats its own cut (the median entry) more often than base (17 weeks against 15). Each positive
profit is one win of about R9,700 (base and mvt_r90 with the same team in the same week, mvt_r50 in another); without
it every single entry loses money.

### Portfolio

The best of the first N entries; prizes and cost over all N:

| chain | N | best points / week | mean points / week | median best rank pct | beat r = 0.8 | beat r = 0.9 | top 10 | wins | prize | cost | profit | profit top10 | prize / avg entrant |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| benchmark | 1 | 65.7 | 65.7 | 0.47 | 0.28 | 0.14 | 2 | 1 | 9734 | 580 | +9154 | +5363 | 7.42 |
| benchmark | 5 | 70.9 | 63.9 | 0.32 | 0.31 | 0.21 | 4 | 1 | 13167 | 2900 | +10267 | +8749 | 2.01 |
| benchmark | 10 | 75.5 | 64.4 | 0.21 | 0.48 | 0.31 | 5 | 1 | 16824 | 5800 | +11024 | +10904 | 1.28 |
| benchmark | 20 | 77.9 | 64.0 | 0.11 | 0.59 | 0.41 | 6 | 2 | 24109 | 11600 | +12509 | +14251 | 0.92 |
| benchmark | 40 | 80.4 | 63.7 | 0.10 | 0.66 | 0.45 | 8 | 2 | 27376 | 23200 | +4176 | +7152 | 0.52 |
| strategic | 1 | 59.9 | 59.9 | 0.51 | 0.24 | 0.14 | 0 | 0 | 103 | 580 | -477 | -580 | 0.08 |
| strategic | 5 | 73.7 | 60.9 | 0.13 | 0.55 | 0.41 | 4 | 0 | 798 | 2900 | -2102 | -1346 | 0.12 |
| strategic | 10 | 76.8 | 60.6 | 0.10 | 0.59 | 0.52 | 6 | 1 | 13904 | 5800 | +8104 | +6815 | 1.06 |
| strategic | 20 | 80.4 | 61.1 | 0.08 | 0.69 | 0.55 | 9 | 2 | 24070 | 11600 | +12470 | +12304 | 0.92 |
| strategic | 40 | 82.2 | 60.4 | 0.06 | 0.69 | 0.59 | 10 | 2 | 30592 | 23200 | +7392 | +8240 | 0.58 |

With many entries the MVT's risk-seeking chain pays: its best entry finishes higher (top 6% against top 10% at
N = 40), with more top-10 finishes and a higher profit at N = 40 on both schedules, while its entries score fewer
points on average. It needs about 10 entries or more; at N = 1 and 5 the benchmark portfolio is ahead. Against the
average entrant the larger portfolios win back about half of what the same number of average entries did.

## Files and run

```
fantasygo.py    reads the cached seasons, scores every entry in the real contests, the tables above
outputs/
  fantasygo.txt          the tables
  weeks.csv              single entries, one row per gameweek and manager (points, rank, prizes, cuts beaten)
  portfolio_weeks.csv    the portfolio, one row per gameweek, chain and N
```

The teams are read from the structural pipeline's caches: `outputs/23-24/mvt/mvt_simulated_r{50,80,90,99,100}_history_22-23_nfix_gurobi.pkl`
(`mvt.py --r-pct ...`) and `outputs/23-24/mvt/portfolio/N40_g8_r99/gw*.pkl` (`mvt_portfolio.py`). It needs the
`ds_env` environment.

```
python fantasygo.py      # about a minute; no solving
```
