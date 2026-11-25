"""Step 1 of the calibration work (Chapter 4.7): do the fixes make the H2H forecasts honest? No optimizer is run; the
decisions already made (baseline.py's caches: blind, the MVT with 0 and 1 hit) are re-scored under
each variant of the model and compared with what happened.

Variants (cumulative where named so):
  base     delta ~ N(mu, Sigma), G = the opponent mixture (lineups.benchmark), as in the season
  shift    + fix 1: G shifted by the net gap of earlier weeks, mean[(opponent's points - mu_G) - (blind's points -
           blind's predicted)], shrunk by n / (n + 5) (as mvt.g_correction's shift and team_shift)
  factor   + fix 3: a weekly shock shared by every player, delta_p += s mu_p eta (eta ~ N(0, 1) per draw), s^2 =
           Var_t(sum e / sum mu) - mean_t(sum sigma^2 / (sum mu)^2) over every player of the earlier weeks (floor 0)
  twopart  + fix 4: each player plays with p = mu / mu_if_plays, then scores mu_if_plays + s_c z (z the correlated
           standard normal of the base draw), else 0; s_c keeps his mean and variance, so only the shape changes
  all      shift + factor + twopart

Per team-week (the 20 league teams, gameweeks 6-38 without the opponent's chips): the opponent's PIT under G, and P(beat)
for each manager's actual line-up (net of its hits) against the realised result (a draw counts 1/2).
Writes Optimizer/outputs/calibration_check.pkl (resumable).

Usage (ds_env, from working/):  OMP_NUM_THREADS=2 python calibration_check.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "Optimizer"))
import baseline  # noqa: E402  (chdirs to structural/src)
import h2h  # noqa: E402
import lineups  # noqa: E402

OUT = HERE / "Optimizer" / "outputs" / "calibration_check.pkl"
MANAGERS = ("blind", "mvt_mix_r50", "mvt_mix_h1_r50")
VARIANTS = ("base", "shift", "factor", "twopart", "all")
N_MC = 20_000
SHRINK = 5.0


def week_stats(gw, inp, s):
    """Sum of errors, of mu and of sigma^2 over every player with a fixture (for the weekly shock's scale)."""
    players, mu, sigma, _ = h2h.build_mu_sigma(gw, inp, s)
    e = players["total_points"].to_numpy(dtype=float) - mu
    return {"sum_e": e.sum(), "sum_mu": mu.sum(), "sum_var": np.diag(sigma).sum()}


def shock_scale(stats, gw):
    past = [v for g, v in stats.items() if g < gw]
    if len(past) < 3:
        return 0.0
    r = np.array([v["sum_e"] / v["sum_mu"] for v in past])
    idio = np.mean([v["sum_var"] / v["sum_mu"] ** 2 for v in past])
    return float(np.sqrt(max(r.var(ddof=1) - idio, 0.0)))


def draws(pr, inp, gw, rng, factor_s=0.0, twopart=False):
    mu, sigma = pr["mu"], pr["sigma"]
    z_full = rng.multivariate_normal(np.zeros(len(mu)), sigma, size=N_MC, method="eigh")
    if not twopart:
        d = mu + z_full
    else:
        sd = np.sqrt(np.maximum(np.diag(sigma), 1e-12))
        z = z_full / sd
        week = inp.week(gw)
        m_if = week["pred_if_plays"].reindex(pr["universe"]).fillna(0.0).to_numpy(dtype=float)
        m_if = np.maximum(m_if, mu)
        p = np.where(m_if > 1e-6, np.clip(mu / np.maximum(m_if, 1e-6), 0.0, 1.0), 0.0)
        var_c = np.where(p > 1e-6, (sd ** 2 - p * (1 - p) * m_if ** 2) / np.maximum(p, 1e-6), 0.0)
        s_c = np.sqrt(np.maximum(var_c, 0.0))
        plays = rng.uniform(size=(N_MC, len(mu))) < p
        d = np.where(plays, m_if + s_c * z, 0.0)
    if factor_s > 0:
        d = d + factor_s * mu * rng.normal(size=(N_MC, 1))
    return d


def g_draws(d, bm, rng):
    dpad = np.hstack([d, np.zeros((len(d), 1))])
    k = rng.choice(len(bm["p"]), size=len(d), p=bm["p"] / bm["p"].sum())
    return (dpad[np.arange(len(d))[:, None], bm["idx"][k]] * bm["W"][k]).sum(axis=1)


def main():
    h2h._ENV = h2h.gp.Env(params={"OutputFlag": 0, "Threads": 2})
    store = pickle.loads(OUT.read_bytes()) if OUT.exists() else {"rows": {}, "stats": {}}
    gws = list(range(6, 39))
    s0 = h2h.Settings()
    inp0 = h2h.Inputs(s0)
    for gw in gws:  # the league-wide weekly errors, the same for every team
        if gw not in store["stats"]:
            store["stats"][gw] = week_stats(gw, inp0, s0)
            OUT.write_bytes(pickle.dumps(store))
    for team in baseline.LEAGUE:
        s = h2h.Settings(own_team_id=0 if team == baseline.OWN_TEAM else team)
        inp = h2h.Inputs(s)
        lm = lineups.LineupModel(inp)
        cache = pickle.loads(baseline.cache_path(team).read_bytes())
        rows = {m: cache[m] for m in MANAGERS}
        chips = set(baseline.chip_weeks(inp, {g: rows["blind"][g]["opponent"] for g in gws}))
        for gw in gws:
            if gw in chips or (team, gw) in store["rows"]:
                continue
            # fix 1, walk-forward: the net gap of this team's earlier scored weeks
            past = [(r["opp_points"] - r["mu_G"]) - (r["blind_points"] - r["blind_pred"])
                    for (t_, g_), r in store["rows"].items() if t_ == team and g_ < gw]
            shift = (len(past) / (len(past) + SHRINK)) * float(np.mean(past)) if past else 0.0
            b = rows["blind"][gw]
            bank = rows["blind"][gw - 1]["bank_after"] if gw - 1 in rows["blind"] else 0.0
            pr = baseline.problem(gw, inp, s, set(b["squad_before"]) | set(b["squad"]), bank)
            bm = lineups.benchmark(gw, b["opponent"], pr, inp, s, lm)
            x = float(b["opponent_points"])
            vs = {m: (h2h.lineup_vector(rows[m][gw]["lineup"], pr), float(rows[m][gw]["points"]),
                      4.0 * (rows[m][gw].get("hits") or 0)) for m in MANAGERS}
            rng = np.random.default_rng(1000 * team + gw)
            fs = shock_scale(store["stats"], gw)
            out = {"opp_points": x, "mu_G": float(bm["mu_G"]), "blind_pred": float(pr["mu"] @ vs["blind"][0]),
                   "blind_points": float(b.get("points_before_hits", b["points"])), "factor_s": fs, "shift": shift}
            shapes = {}
            for shape in ("base", "factor", "twopart", "allshape"):
                d = draws(pr, inp, gw, rng, factor_s=fs if shape in ("factor", "allshape") else 0.0,
                          twopart=shape in ("twopart", "allshape"))
                shapes[shape] = (d, g_draws(d, bm, rng))
            for v in VARIANTS:
                shape = {"base": "base", "shift": "base", "factor": "factor", "twopart": "twopart",
                         "all": "allshape"}[v]
                d, G = shapes[shape]
                G = G + (shift if v in ("shift", "all") else 0.0)
                rec = {"pit": float((G < x).mean() + 0.5 * (G == x).mean()), "sd_G": float(G.std())}
                for m, (vec, pts, cost) in vs.items():
                    own = d @ vec - cost
                    rec[f"p_{m}"] = float((own > G).mean() + 0.5 * (own == G).mean())
                    rec[f"win_{m}"] = 1.0 if pts > x else (0.5 if pts == x else 0.0)
                out[v] = rec
            store["rows"][(team, gw)] = out
            OUT.write_bytes(pickle.dumps(store))
            print(f"team {team} GW{gw}: mu_G {out['mu_G']:.1f}, opponent {x:.0f}, shift {shift:+.1f}, "
                  f"shock s {fs:.3f}, PIT base {out['base']['pit']:.2f} all {out['all']['pit']:.2f}", flush=True)
    print(summarize(store))


def summarize(store):
    """Per variant: the opponent's PIT and, per manager, mean P(beat) against the share won and the Brier score."""
    recs = [{"team": team, "gw": gw, "variant": v, **r[v]} for (team, gw), r in store["rows"].items() for v in VARIANTS]
    d = pd.DataFrame(recs)
    rows = []
    for v, g in d.groupby("variant", sort=False):
        row = {"variant": v, "PIT mean": g["pit"].mean(), "PIT > 0.5": (g["pit"] > 0.5).mean(),
               "PIT > 0.9": (g["pit"] > 0.9).mean(), "PIT < 0.1": (g["pit"] < 0.1).mean(), "sd_G": g["sd_G"].mean()}
        for m in MANAGERS:
            p, w = g[f"p_{m}"], g[f"win_{m}"]
            row[f"{m}: P"], row[f"{m}: won"], row[f"{m}: Brier"] = p.mean(), w.mean(), ((p - w) ** 2).mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("variant").T.round(3).to_string()


if __name__ == "__main__":
    main()
