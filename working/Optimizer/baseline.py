"""Blind points maximizer against the MVT, both managing a squad through the 2024-25 FPL H2H season.

Every manager plays a normal FPL head-to-head season from our real GW5 squad (team 205, as h2h.py): each gameweek
they keep last week's squad, may make up to their free transfers (unused ones roll over, at most 5; no hits, no
chips), pick an XI in a valid formation and a captain, within the budget (squad value at current prices + money in the
bank) and at most 3 players per club. mu = the structural model's `pred`, Sigma = h2h.build_mu_sigma (team factor on
the earlier gameweeks' errors). After the gameweek each is scored on real minutes and points (h2h.lineup_points:
auto-subs, vice rule) against the week's opponent in the league's seeded round robin (his real FFS score,
h2h.real_points). At the deadline the MVT knows who the opponent is and his squad of the previous gameweek, nothing
of this week's.

  blind    does not use the opponent; learns the result only after the gameweek. Maximizes expected points (the
           MIQP with lambda = 0); transfers when they add more than xp_margin expected points.
  blind_h1, blind_h2  the same, allowed at most 1 or 2 hits (4 points each): one lambda = 0 MIQP per transfer limit
           (free transfers + 0, 1, ...); the squad with the most expected points net of its hits is played if that
           beats holding by xp_margin.
  mvt_mix  benchmark = the opponent as a mixture (lineups.py): his plausible squads from the opponent model (selection
           fits, keep/buy, sparse C-vine; weighted by the sampler's counts; his previous squad without a fit), every
           XI of each squad and every captain, weighted by softmaxes on expected points whose temperatures are fitted
           on the league's real line-ups before the gameweek. Exact Prob(beat) = sum_k p_k Phi(mu'(v - w_k) / sd_k).
           Haugh & Singal's frontier: max mu'v - sense * lambda (v'Sigma v - 2 v' cov(delta, G)) over the lambda grid,
           sense +1 when the best feasible squad's expected score (the lambda = 0 solutions, net of hits) reaches mu_G,
           else -1 (Theorem 3.2.3.1; before the fix of 2026-11-24 it was the held squad's: outputs/pre_sense_fix/);
           the lambda with the highest Prob(beat) is played
           if it beats holding by transfer_margin (as h2h.run_gameweek).

`prob_mix` is each manager's Prob(beat) under the mixture benchmark (the blind manager's too, for calibration).
The evaluation is reported twice: over every gameweek, and without the gameweeks in which the opponent played a chip
(Wildcard, Free Hit, Bench Boost, Triple Captain), the same weeks for every manager and the real team. The season is played through every gameweek either
way.

Usage (ds_env):
  python baseline.py                      # GW6-38, cached per gameweek in outputs/rolling_h2h.pkl (resumes)
  python baseline.py --methods mvt_mix --recompute
"""

import argparse
import os
import re
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "structural/src"
OUT = HERE / "outputs"
OWN_TEAM = 205
TEAMS = (205, 38351, 69718, 593532, 696127, 749244)  # the league teams every manager is run for (h2h.py's set)
# the whole league (205 and its 19 opponents), run for the thesis managers (--league)
LEAGUE = (205, 901, 4213, 7497, 13603, 23881, 38054, 38351, 69718, 152245, 198971, 577273, 593532, 696127, 749244,
          1222335, 1441046, 1546929, 2634457, 4131447)
LEAGUE_MANAGERS = ("blind", "mvt_mix_r50", "mvt_mix_h1_r50", "mvt_mix_h2_r50",
                   "mvt_cal_r50", "mvt_cal_h1_r50", "mvt_cal_h2_r50")


CACHE_TAG = ""  # --cache-tag: a separate cache file per team (runs that must not share a file with a running job)


def cache_path(team, tag=None):
    tag = CACHE_TAG if tag is None else tag
    return OUT / (f"rolling_h2h{tag}.pkl" if team == OWN_TEAM else f"rolling_h2h{tag}_team{team}.pkl")
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HERE))
os.chdir(SRC)  # config's paths are relative to src/

import h2h  # noqa: E402
import h2h_chips  # noqa: E402
import lineups  # noqa: E402

_solve_gurobi = h2h.solve_gurobi


def _solve_with_retry(model, lam, tries=5):
    """h2h.solve_gurobi, retried when Gurobi's licence token server cannot be reached (a network outage)."""
    import time

    for k in range(tries):
        try:
            return _solve_gurobi(model, lam)
        except h2h.gp.GurobiError as e:
            if k == tries - 1 or "token" not in str(e).lower() and getattr(e, "errno", None) != 28:
                raise
            print(f"Gurobi licence error ({e}); retrying in {30 * (k + 1)} s", flush=True)
            time.sleep(30 * (k + 1))


h2h.solve_gurobi = _solve_with_retry

R_LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9)
MAX_HITS = {"mvt_mix": 0, "mvt_mix_h1": 1, "mvt_mix_h2": 2}  # transfers allowed beyond the free ones per gameweek
# the r variants: benchmark G = the r-quantile of the opponent's plausible line-ups' scores (lineups.quantile_benchmark)
R_OF = {f"{m}_r{round(r * 100)}": r for m in MAX_HITS for r in R_LEVELS}
MAX_HITS |= {f"{m}_r{round(r * 100)}": h for m, h in list(MAX_HITS.items()) for r in R_LEVELS}
# the margin variants: transfer only if Prob(beat) rises by at least the margin over holding (the others: 0.03)
MARGIN_BASES, MARGINS = ("mvt_mix_h1", "mvt_mix_h1_r50"), (0.0, 0.01, 0.05, 0.10)
MARGIN_OF = {f"{m}_m{round(x * 100)}": x for m in MARGIN_BASES for x in MARGINS}
BASE_OF = {f"{m}_m{round(x * 100)}": m for m in MARGIN_BASES for x in MARGINS}
MAX_HITS |= {k: MAX_HITS[b] for k, b in BASE_OF.items()}
R_OF |= {k: R_OF[b] for k, b in BASE_OF.items() if b in R_OF}
# the blind manager with hits: up to 0, 1 or 2 transfers beyond the free ones, played when expected points net of hits
# beat holding by xp_margin
BLIND_HITS = {"blind": 0, "blind_h1": 1, "blind_h2": 2}
MAX_HITS |= {m: h for m, h in BLIND_HITS.items() if h}
# the calibrated MVT (Chapter 4.7, calibration_check.py): G shifted by the walk-forward net gap (cal_shift), and
# Haugh & Singal's Algorithm 3: both frontiers solved every week (mu_Y - lambda sigma^2 over the squads with mu_Y >= 0,
# and mu_Y + lambda sigma^2), the squad with the highest Prob(beat) played. G^(0.5) as the r50 managers
CAL = {"mvt_cal_r50": 0, "mvt_cal_h1_r50": 1, "mvt_cal_h2_r50": 2}
MAX_HITS |= CAL
R_OF |= dict.fromkeys(CAL, 0.5)
CAL_SHRINK = 5.0  # as mvt.Settings.g_shrink
MANAGERS = ("blind", "blind_h1", "blind_h2", "mvt_mix", "mvt_mix_h1", "mvt_mix_h2", *dict.fromkeys(R_OF),
            *(k for k in BASE_OF if k not in R_OF))
# the thesis comparisons: every MVT against the blind manager WITHOUT hits (the fixed baseline: maximize expected points
# with the free transfers only), the calibrated against the uncalibrated MVT, what hits add to the MVT, and real.
# (blind_h1 / blind_h2 stay in the caches but are not a baseline.)
THESIS_PAIRS = [*((m, "blind") for m in ("mvt_mix_r50", "mvt_mix_h1_r50", "mvt_mix_h2_r50",
                                         "mvt_cal_r50", "mvt_cal_h1_r50", "mvt_cal_h2_r50")),
                ("mvt_cal_r50", "mvt_mix_r50"), ("mvt_cal_h1_r50", "mvt_mix_h1_r50"), ("mvt_cal_h2_r50", "mvt_mix_h2_r50"),
                ("mvt_mix_h1_r50", "mvt_mix_r50"), ("mvt_mix_h2_r50", "mvt_mix_r50"),
                ("mvt_cal_h1_r50", "mvt_cal_r50"), ("mvt_cal_h2_r50", "mvt_cal_r50"),
                *((m, "real") for m in ("blind", "mvt_mix_r50", "mvt_mix_h1_r50", "mvt_mix_h2_r50", *CAL))]
HIT_COST = 4.0


def problem(gw, inp, s, squad, bank):
    """The week's universe (players with a fixture and a price, plus our squad's players without a fixture), mu,
    Sigma, prices and what we own."""
    players, mu_all, sigma_all, diag_all = h2h.build_mu_sigma(gw, inp, s)
    latest = inp.latest(gw)
    predicted = [e for e in players.index if e in latest.index]
    extra = sorted(set(squad) - set(predicted))
    universe = predicted + extra
    idx = np.array([players.index.get_loc(e) for e in predicted])
    n, n_x = len(universe), len(extra)
    mu = np.concatenate([mu_all[idx], np.zeros(n_x)])
    sigma = np.zeros((n, n))
    sigma[:len(predicted), :len(predicted)] = sigma_all[np.ix_(idx, idx)]
    sigma[len(predicted):, len(predicted):] = 1e-4 * np.eye(n_x)
    diag = np.concatenate([diag_all[idx], np.full(n_x, 1e-4)])
    return {"gw": gw, "universe": universe, "mu": mu, "sigma": sigma, "diag": diag,
            "price": latest.loc[universe, "value"].to_numpy(dtype=float),
            "pos": np.array([inp.position[e] for e in universe]), "club": latest.loc[universe, "team"].to_numpy(),
            "owned": np.array([float(e in squad) for e in universe]),
            "budget": float(latest.loc[sorted(squad), "value"].sum()) + bank,
            "mu_map": dict(zip(universe, mu)), "at": {e: i for i, e in enumerate(universe)},
            "cov_g": np.zeros(n)}


def cal_shift(cache, gw, inp):
    """Fix 1 (calibration_check.py), walk-forward: shrink * mean over the team's earlier gameweeks without the
    opponent's chips of [(opponent's points - mixture mu_G) - (blind's points - blind's predicted)], shrink = n /
    (n + CAL_SHRINK). The same for every manager of the team (it uses the blind manager's rows)."""
    rows = cache.get("blind", {})
    past = {g: r for g, r in rows.items() if g < gw and r.get("opponent_points") is not None}
    chips = set(chip_weeks(inp, {g: r["opponent"] for g, r in past.items()})) if past else set()
    net = [(r["opponent_points"] - r["mix_mu_G"]) - (r.get("points_before_hits", r["points"]) - r["pred"])
           for g, r in past.items() if g not in chips]
    return len(net) / (len(net) + CAL_SHRINK) * float(np.mean(net)) if net else 0.0


def decide(method, gw, inp, s, squad, ft, bank, lm, shift=0.0):
    """One manager's gameweek: hold or the MIQP's squad, its line-up, transfers and the result against the opponent.
    shift: added to G (the calibrated managers, cal_shift)."""
    pr = problem(gw, inp, s, squad, bank)
    opp = inp.opponent_of[gw]
    bm = lineups.benchmark(gw, opp, pr, inp, s, lm)
    if method in R_OF:  # G = the r-quantile of his line-ups' scores, by Monte Carlo
        qb = lineups.quantile_benchmark(pr, bm, R_OF[method], seed=s.seed * 1000 + gw)
        if shift:
            qb = {**qb, "G": qb["G"] + shift, "mu_G": qb["mu_G"] + shift}
        prob, mu_g, cov_g = (lambda v, cost=0.0: lineups.prob_beat_mc(v, qb, cost)), qb["mu_G"], qb["cov_g"]
    else:  # G = his score, one draw from the mixture
        prob, mu_g, cov_g = (lambda v, cost=0.0: lineups.prob_beat(v, pr, bm, cost)), bm["mu_G"], pr["sigma"] @ bm["wbar"]
    hold_lineup = h2h.pick_lineup(sorted(squad), pr["mu_map"], inp.position)
    hold_v = h2h.lineup_vector(hold_lineup, pr)
    p_hold = prob(hold_v)
    lam_star, sense = 0.0, 1
    if method in BLIND_HITS:
        # one lambda = 0 MIQP per transfer limit ft + h; the squad with the most expected points net of the hits it
        # takes (ties: fewer hits)
        best = None
        for h in range(BLIND_HITS[method] + 1):
            sol = h2h.solve_gurobi(h2h.build_gurobi(pr, min(ft + h, 15), 1, s), 0.0)
            if sol is not None:
                hits = max(int(((1 - pr["owned"]) @ sol[0]).round()) - ft, 0)
                net = float(pr["mu"] @ sol[1]) - HIT_COST * hits
                if best is None or net > best[0]:
                    best = (net, hits, sol)
        net, hits, (w, v) = best
        p = prob(v, HIT_COST * hits)
        adopt = net - float(pr["mu"] @ hold_v) > s.xp_margin
    else:
        # up to MAX_HITS[method] transfers beyond the free ones, HIT_COST points each: one MIQP per transfer limit
        # ft + h, its Prob(beat) net of the hits its squad actually takes
        # Theorem 3.2.3.1: the sign comes from the best expected margin any feasible squad can reach, i.e. the
        # lambda = 0 solutions (one per transfer limit, net of their hits): if one reaches mu_G, the optimum has
        # mu_Y >= 0 and lies on mu_Y - lambda sigma_Y^2 (cut variance), otherwise on mu_Y + lambda sigma_Y^2
        pr["cov_g"] = cov_g
        frontier, zero = [], {}
        for h in range(MAX_HITS[method] + 1):  # lambda = 0: the objective is mu'v whatever the sign (convex)
            sol = h2h.solve_gurobi(h2h.build_gurobi(pr, min(ft + h, 15), 1, s), 0.0)
            if sol is not None:
                hits = max(int(((1 - pr["owned"]) @ sol[0]).round()) - ft, 0)
                zero[h] = float(pr["mu"] @ sol[1]) - HIT_COST * hits
                frontier.append((prob(sol[1], HIT_COST * hits), 0.0, sol))
        best_net = max(zero.values(), default=float(pr["mu"] @ hold_v))
        sense = 1 if max(best_net, float(pr["mu"] @ hold_v)) - mu_g >= 0 else -1
        calibrated = method in CAL
        # Algorithm 3 (calibrated managers): both signs, the cutting one over squads with mu_Y >= 0 only
        senses = (1, -1) if calibrated else (sense,)
        signs = {}
        for h in zero:
            for sg in senses:
                model = h2h.build_gurobi(pr, min(ft + h, 15), sg, s,
                                         **({"min_mean": mu_g, "ft": ft, "hit_cost": HIT_COST}
                                            if calibrated and sg > 0 else {}))
                for lam in s.lambda_grid:
                    if lam == 0:
                        continue
                    sol = h2h.solve_gurobi(model, lam)
                    if sol is not None:
                        hits = max(int(((1 - pr["owned"]) @ sol[0]).round()) - ft, 0)
                        frontier.append((prob(sol[1], HIT_COST * hits), lam, sol))
                        signs[id(sol)] = sg
        p, lam_star, (w, v) = max(frontier, key=lambda f: f[0])
        if calibrated:
            best_sol = max(frontier, key=lambda f: f[0])[2]
            sense = signs.get(id(best_sol), 0)  # 0: a lambda = 0 squad
        margin = MARGIN_OF.get(method, s.transfer_margin)
        adopt = p > p_hold + margin if margin > 0 else p > p_hold
    if adopt:
        new_squad, lineup = h2h.lineup_from_solution(w, v, pr, inp)
    else:
        new_squad, lineup, p, v = sorted(squad), hold_lineup, p_hold, hold_v
    n_tr = len(set(new_squad) - set(squad))
    hits = max(n_tr - ft, 0)
    _, pts = h2h.lineup_points(lineup, inp.week(gw), inp.position)
    real = h2h.real_lineup(opp, gw, inp)
    return {"gw": gw, "opponent": opp, "squad_before": sorted(squad), "squad": sorted(new_squad), "lineup": lineup,
            "n_transfers": n_tr, "hits": hits, "points_before_hits": pts, "ft_before": ft,
            "ft_after": min(h2h.FT_CAP, max(ft - n_tr, 0) + 1),
            "bank_after": pr["budget"] - float(pr["price"] @ np.array([e in new_squad for e in pr["universe"]])),
            "pred": float(pr["mu"] @ v), "sd": float(np.sqrt(v @ pr["sigma"] @ v)), "prob": p,
            "prob_mix": lineups.prob_beat(v, pr, bm, HIT_COST * hits), "mu_G": mu_g, "mix_mu_G": bm["mu_G"],
            "mix_sd_G": float(np.sqrt(bm["var_G"])), "lambda_star": lam_star if adopt else None, "sense": sense,
            "shift": shift,
            "points": pts - HIT_COST * hits, "opponent_points": h2h.real_points(opp, gw, inp),
            "real_points": h2h.real_points(inp.own, gw, inp),
            "opp_overlap": len(set(lineup[0]) & set(real[0][0] if real else ())),
            "mix": {k: bm[k] for k in ("n_squads", "top_squad_share", "mass_kept", "fallback", "tau_xi", "tau_cap",
                                        "n_obs")}}


def season(method, inp, s, gws, cache, lm, path):
    rows = cache.setdefault(method, {})
    squad = set(inp.league.squad_players(inp.own, s.start_gw - 1))
    ft, bank = inp.league.free_transfers(inp.own, s.start_gw), 0.0
    for gw in range(s.start_gw, max(gws) + 1):
        if gw not in rows:
            shift = cal_shift(cache, gw, inp) if method in CAL else 0.0
            rows[gw] = decide(method, gw, inp, s, squad, ft, bank, lm, shift)
            path.write_bytes(pickle.dumps(cache))
            r = rows[gw]
            print(f"[{method}] GW{gw}: {r['n_transfers']} transfers, pred {r['pred']:.1f} vs mu_G {r['mu_G']:.1f}, "
                  f"P(beat) {r['prob']:.2f} (mixture {r['prob_mix']:.2f}), points {r['points']:.0f} vs opponent "
                  f"{r['opponent_points']:.0f}", flush=True)
        squad, ft, bank = set(rows[gw]["squad"]), rows[gw]["ft_after"], rows[gw]["bank_after"]
    return pd.DataFrame([rows[g] for g in gws]).set_index("gw")


def chip_weeks(inp, team_of):
    """{gw: chip} for the gameweeks in which the team playing that week (team_of[gw]) played a chip: Wildcard, Free
    Hit, Bench Boost or Triple Captain (h2h_chips.chip_histories on the league's picks)."""
    played = h2h_chips.chip_histories(inp.picks).set_index(["team_id", "gw"])["chip"]
    return {gw: played[(team, gw)] for gw, team in team_of.items() if (team, gw) in played.index}


def zero_hit_of(m):
    """The same manager without hits: blind_h1 -> blind, mvt_mix_h1_r50 -> mvt_mix_r50, mvt_cal_h2_r50 -> mvt_cal_r50."""
    return re.sub(r"_h\d", "", m)


def by_hits(rollouts, opp, managers=None):
    """The hit-taking managers' gameweeks grouped by the number of hits taken: how often, the points net of hits, the
    result against the opponent, and the difference from the same manager without hits (zero_hit_of) in the same
    gameweeks. rollouts and opp may be pooled over teams (a (team, gw) index)."""
    rows = []
    for m, r in rollouts.items():
        if MAX_HITS.get(m, 0) == 0 or (managers is not None and m not in managers):
            continue
        ref = zero_hit_of(m)
        for h, g in r.groupby(r["hits"].fillna(0).astype(int)):
            o = opp.loc[g.index]
            rows.append({"manager": m, "hits": h, "gameweeks": len(g), "points (net)": g["points"].sum(),
                         "mean net points": g["points"].mean(),
                         "mean gross points": g["points_before_hits"].mean(), "W": int((g["points"] > o).sum()),
                         "win rate": (g["points"] > o).mean(), "mean P(beat)": g["prob"].mean(),
                         "vs 0-hit, per week": (g["points"] - rollouts[ref].loc[g.index, "points"]).mean()
                         if ref in rollouts else np.nan})
    return pd.DataFrame(rows).set_index(["manager", "hits"]) if rows else pd.DataFrame()


def summary(rollouts, opp, real, real_opp, team=OWN_TEAM):
    """The season of each manager against opp (the opponents' real scores) over the rollouts' gameweeks, and of the
    real team (real) against real_opp over its own gameweeks."""
    out = {}
    for name, own, against in ([(f"real ({team})", real, real_opp)]
                               + [(m, r["points"], opp) for m, r in rollouts.items()]):
        own, o = own.to_numpy(dtype=float), against.to_numpy(dtype=float)
        win = own > o
        row = {"gameweeks": len(own), "points": own.sum(), "margin": own.sum() - o.sum(), "W": int(win.sum()),
               "D": int((own == o).sum()), "L": int((own < o).sum()),
               "H2H points": int(h2h.h2h_result(own, o).sum()), "win rate": win.mean()}
        if name in rollouts:
            r = rollouts[name]
            hits = r["hits"].fillna(0) if "hits" in r else pd.Series(0, index=r.index)
            row |= {"transfers": int(r["n_transfers"].sum()), "hits": int(hits.sum()),
                    "hit points": -HIT_COST * hits.sum(), "mean pred": r["pred"].mean(),
                    "mean P(beat)": r["prob"].mean(), "mean P mixture": r["prob_mix"].mean(),
                    "Brier mixture": ((r["prob_mix"] - win) ** 2).mean(), "XI overlap": r["opp_overlap"].mean()}
        out[name] = row
    return pd.DataFrame(out).T


def block_bootstrap(per, block=4, n_boot=10_000, seed=0):
    """Circular block bootstrap of the pooled mean of per-team difference series (pd.Series indexed by gameweek),
    resampling gameweeks jointly: blocks of `block` consecutive gameweeks are drawn with wrap-around (keeping the
    serial correlation of the rolling squads), the same gameweeks for every team (keeping the correlation between
    teams in a gameweek: the automated managers share the predictions, so a week they miss, they miss in every team).
    A team without the gameweek (an opponent's chip week) adds nothing to that draw. Returns the bootstrap SE, the
    t = mean / SE, the 95% percentile interval and the two-sided p (share of centred bootstrap means at least as
    far from the mean as the mean is from 0)."""
    W = pd.concat([pd.Series(d) for d in per], axis=1).sort_index().to_numpy(dtype=float)
    ok = ~np.isnan(W)
    val, cnt = np.where(ok, W, 0.0).sum(1), ok.sum(1).astype(float)  # per gameweek: sum and count over teams
    n = len(W)
    rng = np.random.default_rng(seed)
    k = -(-n // block)
    idx = (rng.integers(0, n, size=(n_boot, k))[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n] % n
    boot, mean = val[idx].sum(1) / cnt[idx].sum(1), val.sum() / cnt.sum()
    se = boot.std(ddof=1)
    return {"boot SE": se, "boot t": mean / se, "CI low": np.quantile(boot, 0.025),
            "CI high": np.quantile(boot, 0.975), "p": float((np.abs(boot - mean) >= abs(mean)).mean())}


def lag1(per):
    """Pooled lag-1 autocorrelation of the per-team difference series (deviations from each team's mean)."""
    x = [np.asarray(d, dtype=float) - np.mean(d) for d in per]
    return sum((a[1:] * a[:-1]).sum() for a in x) / sum((a ** 2).sum() for a in x)


def cross_corr(per):
    """Mean correlation between teams of the difference series, over the gameweeks each pair shares."""
    W = pd.concat([pd.Series(d) for d in per], axis=1)
    c = W.corr().to_numpy()
    return float(c[np.triu_indices_from(c, 1)].mean())


def paired(a, b):
    """Per-team differences a - b (dicts team -> pd.Series by gameweek), over the teams both were run for."""
    return [a[t] - b[t] for t in a if t in b]


def compare_teams(gws, teams=TEAMS, managers=MANAGERS):
    """Every manager over every team in TEAMS (from the caches), with and without chip weeks: per team the points,
    wins and H2H points, then pooled over the teams each manager's points per gameweek minus mvt_mix's."""
    inp = h2h.Inputs(h2h.Settings())  # only the league's picks (chips) are needed
    per_team, diffs, wins, rolls = [], {}, {}, {}
    for team in teams:
        path = cache_path(team)
        if not path.exists():
            print(f"team {team}: no cache ({path.name})")
            continue
        cache = pickle.loads(path.read_bytes())
        full = {m: pd.DataFrame(cache[m].values()).set_index("gw").sort_index().loc[gws]
                for m in managers if m in cache and set(gws) <= set(cache[m])}
        if "blind" not in full:
            continue
        base = full["blind"]
        opp_chips = set(chip_weeks(inp, base["opponent"].to_dict()))
        no_opp = [g for g in gws if g not in opp_chips]  # every row, the real team included
        for view, weeks, real_weeks in [("with chips", gws, gws), ("without chips", no_opp, no_opp)]:
            rolls.setdefault(view, {})[team] = ({m: r.loc[weeks] for m, r in full.items()},
                                                base.loc[weeks, "opponent_points"])
            for m, wk, own in ([("real", real_weeks, base.loc[real_weeks, "real_points"])]
                               + [(m, weeks, r.loc[weeks, "points"]) for m, r in full.items()]):
                own, opp = own.to_numpy(dtype=float), base.loc[wk, "opponent_points"].to_numpy(dtype=float)
                per_team.append({"view": view, "team": team, "manager": m, "gameweeks": len(wk),
                                 "points": own.sum(), "W": int((own > opp).sum()), "D": int((own == opp).sum()),
                                 "L": int((own < opp).sum()), "H2H points": int(h2h.h2h_result(own, opp).sum()),
                                 "transfers": int(full[m].loc[wk, "n_transfers"].sum()) if m in full else np.nan,
                                 "hits": int(full[m].loc[wk].get("hits", pd.Series(0)).sum()) if m in full else np.nan})
                diffs.setdefault((view, m), {})[team] = pd.Series(own, index=wk)
                wins.setdefault((view, m), {})[team] = pd.Series(h2h.h2h_result(own, opp).astype(float), index=wk)
    t = pd.DataFrame(per_team)
    pairs = [("mvt_mix", "blind"), ("mvt_mix_h1", "blind"), ("mvt_mix_h2", "blind"), ("mvt_mix_h1", "mvt_mix"),
             ("mvt_mix_h2", "mvt_mix"), ("mvt_mix_h2", "mvt_mix_h1")]
    for view in ("with chips", "without chips"):
        print(f"\n{view}" + (" (the weeks in which the opponent played none)"
                              if view == "without chips" else "") + ":")
        x_all = t[t["view"] == view]
        x = x_all[~x_all["manager"].isin(BASE_OF)]  # the margin variants have their own table below
        print(x.pivot_table(index="team", columns="manager", values=["gameweeks", "points", "W", "H2H points"],
                            aggfunc="first").to_string())
        x = x.assign(rank=x.groupby("team")["points"].rank(ascending=False, method="min").astype(int))
        print("ranked by points within each team:")
        print(x.sort_values(["team", "rank"])[["team", "rank", "manager", "gameweeks", "points", "W", "D", "L",
                                               "H2H points"]].to_string(index=False))
        ranks = x.pivot(index="manager", columns="team", values="rank")
        print(ranks.assign(mean=ranks.mean(axis=1)).sort_values("mean").to_string(float_format=lambda v: f"{v:.2f}"))
        tot = x.groupby("manager")[["gameweeks", "points", "W", "D", "L", "H2H points"]].sum()
        tot["points / week"], tot["win rate"] = tot["points"] / tot["gameweeks"], tot["W"] / tot["gameweeks"]
        print(tot.loc[[m for m in ["real", *MANAGERS] if m in tot.index]].to_string(float_format=lambda v: f"{v:.2f}"))
        rows = []
        for a, b in pairs:
            if (view, a) in diffs and (view, b) in diffs:
                per = paired(diffs[(view, a)], diffs[(view, b)])
                d = np.concatenate(per)
                rows.append({"pair": f"{a} - {b}", "team-weeks": len(d), "per week": d.mean(),
                             "t": d.mean() / (d.std(ddof=1) / np.sqrt(len(d))),
                             "boot p": block_bootstrap(per)["p"],
                             "teams ahead": f"{sum(p.mean() > 0 for p in per)}/{len(per)}"})
        if rows:
            print(pd.DataFrame(rows).set_index("pair").to_string(float_format=lambda v: f"{v:+.2f}"))
        # the r levels: per hit level, the mixture benchmark (r = "mix") and the r-quantile benchmarks
        rows = []
        for base_m in ("mvt_mix", "mvt_mix_h1", "mvt_mix_h2"):
            for r in ("mix", *R_LEVELS):
                m = base_m if r == "mix" else f"{base_m}_r{round(r * 100)}"
                if (view, m) not in diffs or m not in tot.index:
                    continue
                row = {"hits": MAX_HITS[base_m], "r": r, "manager": m, "points / week": tot.at[m, "points / week"],
                       "W-D-L": f"{tot.at[m, 'W']}-{tot.at[m, 'D']}-{tot.at[m, 'L']}",
                       "H2H points": tot.at[m, "H2H points"], "mean rank": ranks.loc[m].mean()}
                for ref in ("real", "blind", base_m):
                    if ref != m and (view, ref) in diffs:
                        per = paired(diffs[(view, m)], diffs[(view, ref)])
                        d = np.concatenate(per)
                        name = "mix" if ref == base_m else ref
                        row[f"vs {name}"] = d.mean()
                        row[f"t vs {name}"] = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
                        row[f"teams ahead of {name}"] = f"{sum(p.mean() > 0 for p in per)}/{len(per)}"
                        if ref != base_m:
                            row[f"weeks above {name}"] = f"{int((d > 0).sum())}-{int((d == 0).sum())}-{int((d < 0).sum())}"
                rows.append(row)
        if rows:
            print("r levels:")
            print(pd.DataFrame(rows).set_index(["hits", "r"]).to_string(float_format=lambda v: f"{v:.2f}"))
        # block bootstrap of the same paired differences (blocks of 4 weeks; 2 and 8 as a check on the block length)
        rows = []
        for base_m in ("mvt_mix", "mvt_mix_h1", "mvt_mix_h2"):
            for r in ("mix", *R_LEVELS):
                m = base_m if r == "mix" else f"{base_m}_r{round(r * 100)}"
                for ref in ("real", "blind", base_m):
                    if ref == m or (view, m) not in diffs or (view, ref) not in diffs:
                        continue
                    per = paired(diffs[(view, m)], diffs[(view, ref)])
                    d = np.concatenate(per)
                    b = block_bootstrap(per)
                    rows.append({"hits": MAX_HITS[base_m], "r": r, "vs": "mix" if ref == base_m else ref,
                                 "per week": d.mean(), "t": d.mean() / (d.std(ddof=1) / np.sqrt(len(d))),
                                 "lag-1 corr": lag1(per), "corr between teams": cross_corr(per), **b,
                                 "boot t (block 2)": d.mean() / block_bootstrap(per, block=2)["boot SE"],
                                 "boot t (block 8)": d.mean() / block_bootstrap(per, block=8)["boot SE"]})
        if rows:
            print("block bootstrap (circular, gameweeks resampled jointly over the teams, 10000 draws, blocks of 4):")
            print(pd.DataFrame(rows).set_index(["hits", "r", "vs"]).to_string(float_format=lambda v: f"{v:.3f}"))
        # the transfer margin: per base manager, the minimum Prob(beat) gain over holding (0.03 = the base manager)
        rows = []
        for base_m in MARGIN_BASES:
            for x_m in sorted({0.03, *MARGINS}):
                m = base_m if x_m == 0.03 else f"{base_m}_m{round(x_m * 100)}"
                if (view, m) not in diffs:
                    continue
                y = x_all[x_all["manager"] == m]
                row = {"manager": base_m, "margin": x_m, "points / week": y["points"].sum() / y["gameweeks"].sum(),
                       "W-D-L": f"{y['W'].sum()}-{y['D'].sum()}-{y['L'].sum()}", "H2H points": y["H2H points"].sum(),
                       "transfers": y["transfers"].sum(), "hits": y["hits"].sum()}
                for ref in ("real", "blind", base_m):
                    if ref == m:
                        continue
                    per = paired(diffs[(view, m)], diffs[(view, ref)])
                    d = np.concatenate(per)
                    name = "0.03" if ref == base_m else ref
                    row[f"vs {name}"] = d.mean()
                    row[f"t vs {name}"] = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
                    row[f"p vs {name}"] = block_bootstrap(per)["p"]
                    if ref == base_m:
                        b = block_bootstrap(per)
                        row["boot t vs 0.03"] = b["boot t"]
                        row["teams ahead of 0.03"] = f"{sum(p.mean() > 0 for p in per)}/{len(per)}"
                rows.append(row)
        if rows:
            print("transfer margin (minimum Prob(beat) gain over holding):")
            print(pd.DataFrame(rows).set_index(["manager", "margin"]).to_string(float_format=lambda v: f"{v:.2f}"))
        # the thesis comparisons: paired per-week differences with the block bootstrap
        rows = []
        for a, b in THESIS_PAIRS:
            if (view, a) not in diffs or (view, b) not in diffs:
                continue
            per = paired(diffs[(view, a)], diffs[(view, b)])
            d = np.concatenate(per)
            rows.append({"pair": f"{a} - {b}", "team-weeks": len(d), "per week": d.mean(),
                         "t": d.mean() / (d.std(ddof=1) / np.sqrt(len(d))),
                         "weeks above": f"{int((d > 0).sum())}-{int((d == 0).sum())}-{int((d < 0).sum())}",
                         "teams ahead": f"{sum(p.mean() > 0 for p in per)}/{len(per)}",
                         "corr between teams": cross_corr(per), **block_bootstrap(per)})
        if rows:
            print("thesis comparisons (block bootstrap, blocks of 4 weeks):")
            print(pd.DataFrame(rows).set_index("pair").to_string(float_format=lambda v: f"{v:+.3f}"))
        # the same comparisons in H2H points per week (3 a win, 1 a draw): what the managers are built to maximize
        rows = []
        for a, b in THESIS_PAIRS:
            if (view, a) not in wins or (view, b) not in wins:
                continue
            per = paired(wins[(view, a)], wins[(view, b)])
            d = np.concatenate(per)
            rows.append({"pair": f"{a} - {b}", "team-weeks": len(d), "H2H points / week": d.mean(),
                         "per season": d.mean() * len(gws), "total": d.sum(),
                         "weeks won-same-lost": f"{int((d > 0).sum())}-{int((d == 0).sum())}-{int((d < 0).sum())}",
                         "teams ahead": f"{sum(p.mean() > 0 for p in per)}/{len(per)}",
                         "corr between teams": cross_corr(per), **block_bootstrap(per)})
        if rows:
            print("thesis comparisons in H2H points (3 a win, 1 a draw; block bootstrap, blocks of 4 weeks):")
            print(pd.DataFrame(rows).set_index("pair").to_string(float_format=lambda v: f"{v:+.3f}"))
        # the hit-taking thesis managers pooled over the teams, by the number of hits taken in the gameweek
        tm = rolls.get(view, {})
        hit_ms = [m for m in ("mvt_mix_h1_r50", "mvt_mix_h2_r50", "mvt_cal_h1_r50", "mvt_cal_h2_r50")
                  if all(m in r for r, _ in tm.values())]
        if tm and hit_ms:
            pooled = {m: pd.concat({team: r[m] for team, (r, _) in tm.items()}, names=["team", "gw"])
                      for m in {*hit_ms, *map(zero_hit_of, hit_ms)} if all(m in r for r, _ in tm.values())}
            opp_all = pd.concat({team: o for team, (_, o) in tm.items()}, names=["team", "gw"])
            print("gameweeks by hits taken, pooled over the teams (vs 0-hit: the same manager without hits):")
            print(by_hits(pooled, opp_all, hit_ms).to_string(float_format=lambda v: f"{v:.2f}"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gws", default="6-38")
    parser.add_argument("--methods", nargs="*", default=list(MANAGERS), choices=MANAGERS)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--team", type=int, default=OWN_TEAM, help="the league team whose squad is managed")
    parser.add_argument("--threads", type=int, default=2, help="Gurobi threads per solve")
    parser.add_argument("--cache-tag", default="", help="write the team's cache to rolling_h2h{tag}..., not the shared one")
    parser.add_argument("--compare-teams", action="store_true",
                        help="print the managers' results over every team in TEAMS from the caches (no solving)")
    parser.add_argument("--league", action="store_true",
                        help="with --compare-teams: the thesis managers (LEAGUE_MANAGERS) over all 20 league teams")
    args = parser.parse_args()
    global CACHE_TAG
    CACHE_TAG = args.cache_tag
    lo, _, hi = args.gws.partition("-")
    gws = list(range(int(lo), int(hi or lo) + 1))
    h2h._ENV = h2h.gp.Env(params={"OutputFlag": 0, "Threads": args.threads})
    if args.compare_teams:
        if args.league:
            return compare_teams(gws, LEAGUE, LEAGUE_MANAGERS)
        return compare_teams(gws)

    s = h2h.Settings(own_team_id=0 if args.team == OWN_TEAM else args.team)
    inp = h2h.Inputs(s)
    lm = lineups.LineupModel(inp)
    OUT.mkdir(exist_ok=True)
    path = cache_path(args.team)
    cache = pickle.loads(path.read_bytes()) if path.exists() else {}
    for m in args.methods if args.recompute else ():
        cache.pop(m, None)
    full = {m: season(m, inp, s, gws, cache, lm, path) for m in MANAGERS if m in cache or m in args.methods}
    if "blind" not in full:  # a tagged cache (--cache-tag) without the blind manager: nothing to compare
        return
    base = full["blind"]
    opp, real = base["opponent_points"], base["real_points"]
    opp_chips = chip_weeks(inp, base["opponent"].to_dict())
    own_chips = chip_weeks(inp, dict.fromkeys(gws, inp.own))
    print(f"\nOpponent's chips: {', '.join(f'GW{g} {c}' for g, c in opp_chips.items()) or 'none'}")
    print(f"Team {inp.own}'s chips: {', '.join(f'GW{g} {c}' for g, c in own_chips.items()) or 'none'}")
    fmt = lambda x: f"{x:.3f}" if abs(x) < 10 else f"{x:.0f}"  # noqa: E731
    no_opp = [g for g in gws if g not in opp_chips]  # the managers: without the opponent's chip weeks
    for title, weeks, real_weeks in [("with chips", gws, gws),
                                     ("without chips (the opponent played none)", no_opp, no_opp)]:
        rollouts = {m: r.loc[weeks] for m, r in full.items()}
        print(f"\nGW{gws[0]}-{gws[-1]} {title}: {len(weeks)} gameweeks for the managers (opponents' real points "
              f"{opp.loc[weeks].sum():.0f}), {len(real_weeks)} for the real team")
        print(summary(rollouts, opp.loc[weeks], real.loc[real_weeks], opp.loc[real_weeks], inp.own)
              .to_string(float_format=fmt))
        for m in rollouts:
            for ref in ("blind", "mvt_mix"):
                if m != ref and ref in rollouts and not (ref == "mvt_mix" and m == "blind"):
                    d = rollouts[m]["points"] - rollouts[ref]["points"]
                    print(f"{m} - {ref} per gameweek: mean {d.mean():+.2f}, sd {d.std():.2f}, "
                          f"t {d.mean() / (d.std() / np.sqrt(len(d))):+.2f}")
        print(by_hits(rollouts, opp.loc[weeks]).to_string(float_format=fmt))
    fits = pd.DataFrame([r["mix"] for r in full[next(iter(full))].to_dict("records")], index=gws)
    print(f"\nLine-up model: tau_xi {fits.tau_xi.min():.2f}-{fits.tau_xi.max():.2f}, tau_cap "
          f"{fits.tau_cap.min():.2f}-{fits.tau_cap.max():.2f}; squads per week median {fits.n_squads.median():.0f}, "
          f"top squad share median {fits.top_squad_share.median():.2f}, mass kept min {fits.mass_kept.min():.2f}, "
          f"fallback weeks {int(fits.fallback.sum())}")


if __name__ == "__main__":
    main()
