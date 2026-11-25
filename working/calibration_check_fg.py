"""Step 1 of the calibration work, FantasyGo 2023-24: do the fixes make the top-of-the-field forecasts honest? No
optimizer is run; the teams already chosen (blind, the top-heavy MVT's lambda* team, the r = 0.6 MVT's team) are
re-scored on the same simulated field (seed + gw, as mvt.run_gameweek) under each variant and compared with the
real contest.

Variants:
  raw      no correction of the field's scores
  prod     the correction used now: mvt.g_correction at r = 0.96 (one shift and one widening for every place)
  perrank  fix 2: place d's own walk-forward shift, shrink * mean over earlier contests of [(real d-th best - mean
           simulated d-th best) - (blind's actual - blind's predicted)] (place 10's beyond the 10th)
  factor   perrank + fix 3, the weekly shock delta_p += s mu_p eta (s as calibration_check.shock_scale, from every
           player of the earlier contest weeks)
  twopart  perrank + fix 4, plays-or-not draws with each player's mean and variance kept
  all      perrank + factor + twopart

Per contest and team: P(top 10), expected prize (the previous contest's payout table) against the realised rank and
prize; and the PIT of the real 1st and 10th best scores. Writes Optimizer/outputs/calibration_check_fg.pkl.

Usage (ds_env, from working/):  OMP_NUM_THREADS=2 python calibration_check_fg.py
"""

import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "Optimizer"))
import baseline  # noqa: E402,F401  (puts structural/src on the path and chdirs there)
import mvt  # noqa: E402

OUT = HERE / "Optimizer" / "outputs" / "calibration_check_fg.pkl"
N_MC = 20_000
VARIANTS = ("raw", "prod", "perrank", "factor", "twopart", "all")
TEAMS = ("blind", "topheavy", "r60")


def draws(mu, sigma, players, rng, factor_s=0.0, twopart=False):
    z_full = rng.multivariate_normal(np.zeros(len(mu)), sigma, size=N_MC, method="eigh")
    if not twopart:
        d = mu + z_full
    else:
        sd = np.sqrt(np.maximum(np.diag(sigma), 1e-12))
        m_if = np.maximum(players["pred_if_plays"].fillna(0.0).to_numpy(dtype=float), mu)
        p = np.where(m_if > 1e-6, np.clip(mu / np.maximum(m_if, 1e-6), 0.0, 1.0), 0.0)
        var_c = np.where(p > 1e-6, (sd ** 2 - p * (1 - p) * m_if ** 2) / np.maximum(p, 1e-6), 0.0)
        plays = rng.uniform(size=(N_MC, len(mu))) < p
        d = np.where(plays, m_if + np.sqrt(np.maximum(var_c, 0.0)) * z_full / sd, 0.0)
    if factor_s > 0:
        d = d + factor_s * mu * rng.normal(size=(N_MC, 1))
    return d


def vector(team, players):
    w = np.zeros(len(players))
    at = players.index.get_indexer(team["fpl_id"])
    w[at[at >= 0]] = team["w"].to_numpy()[at >= 0]
    return w


def main():
    s = replace(mvt.Settings(), r_pct=0.96, objective="topheavy")
    s60 = replace(mvt.Settings(), r_pct=0.6)
    inp = mvt.Inputs(s)
    th = pickle.loads(s.cache().read_bytes())["results"]
    r60 = pickle.loads(s60.cache().read_bytes())["results"]
    ev = {"th": mvt.evaluate(th, inp, s), "r60": mvt.evaluate(r60, inp, s60)}
    store = pickle.loads(OUT.read_bytes()) if OUT.exists() else {"rows": {}, "stats": {}}
    gws = list(ev["th"].index)
    for gw in gws:  # league-wide weekly errors for the shock scale (every player with a fixture)
        if gw not in store["stats"]:
            players, mu, sigma = mvt.build_mu_sigma(gw, inp, s)
            e = players["total_points"].to_numpy(dtype=float) - mu
            store["stats"][gw] = {"sum_e": e.sum(), "sum_mu": mu.sum(), "sum_var": np.diag(sigma).sum()}
    for gw in gws:
        if gw in store["rows"]:
            continue
        rng = np.random.default_rng(s.seed + gw)
        players, mu, sigma = mvt.build_mu_sigma(gw, inp, s)
        W_o = mvt.field_matrix(gw, players, inp, s, rng)
        prizes = mvt.topheavy_payout(gw, inp)
        K = max(len(prizes), 10)
        real = inp.managers.loc[inp.managers["gameweek"] == gw, "points"].sort_values(ascending=False).to_numpy()
        teams = {"blind": th[gw]["teams"][0.0], "topheavy": th[gw]["teams"][th[gw]["lambda_star"]],
                 "r60": r60[gw]["teams"][r60[gw]["lambda_star"]]}
        vecs = {k: vector(t, players) for k, t in teams.items()}
        e_th, e60 = ev["th"].loc[gw], ev["r60"].loc[gw]
        realised = {"blind": (e_th["base_rank"], e_th["base_prize"]), "topheavy": (e_th["mvt_rank"], e_th["mvt_prize"]),
                    "r60": (e60["mvt_rank"], e60["mvt_prize"])}
        past = [store["stats"][g] for g in gws if g < gw]
        if len(past) >= 3:
            r = np.array([v["sum_e"] / v["sum_mu"] for v in past])
            idio = np.mean([v["sum_var"] / v["sum_mu"] ** 2 for v in past])
            fs = float(np.sqrt(max(r.var(ddof=1) - idio, 0.0)))
        else:
            fs = 0.0
        # fix 2: per-place net gaps of earlier contests (each stored with its own week's row)
        prev = [store["rows"][g]["net_gap"] for g in gws if g < gw and g in store["rows"]]
        trust = len(prev) / (len(prev) + s.g_shrink)
        per = trust * np.mean(prev, axis=0) if prev else np.zeros(10)
        per_k = np.concatenate([per, np.full(K - 10, per[-1])]) if K > 10 else per[:K]
        prod_shift, prod_widen = mvt.g_correction(gw, inp, s)
        out = {"factor_s": fs, "per_shift_1": float(per[0]), "per_shift_10": float(per[9]),
               "prod_shift": prod_shift, "prod_widen": prod_widen, "real_1": float(real[0]), "real_10": float(real[9])}
        shapes = {}
        for shape in ("base", "factor", "twopart", "allshape"):
            d = draws(mu, sigma, players, rng, fs if shape in ("factor", "allshape") else 0.0,
                      shape in ("twopart", "allshape"))
            T = -np.sort(-(d @ W_o.T), axis=1)[:, :K]
            shapes[shape] = (d, T)
        d0, T0 = shapes["base"]
        own_gap = float(e_th["base_points"] - e_th["base_mean"])
        out["net_gap"] = (real[:10] - T0[:, :10].mean(axis=0)) - own_gap  # for later weeks only
        for v in VARIANTS:
            shape = {"raw": "base", "prod": "base", "perrank": "base", "factor": "factor", "twopart": "twopart",
                     "all": "allshape"}[v]
            d, T = shapes[shape]
            if v == "prod":
                T = T + prod_shift
                if prod_widen > 1:
                    T = T + np.random.default_rng(gw).normal(0, np.sqrt(prod_widen ** 2 - 1) * T[:, 9].std(),
                                                             size=(len(T), 1))
            elif v != "raw":
                T = T + per_k[None, :]
            rec = {"pit_1": float((T[:, 0] < real[0]).mean()), "pit_10": float((T[:, 9] < real[9]).mean())}
            for k, w in vecs.items():
                F = d @ w
                rec[f"p10_{k}"] = float((F > T[:, 9]).mean())
                rec[f"exp_{k}"] = mvt.expected_prize(F, T, prizes)
            out[v] = rec
        out["realised"] = realised
        store["rows"][gw] = out
        OUT.write_bytes(pickle.dumps(store))
        print(f"GW{gw}: shock s {fs:.3f}, per-place shift 1st {per[0]:+.1f} 10th {per[9]:+.1f}, prod {prod_shift:+.1f}"
              f" | P(top10) blind raw {out['raw']['p10_blind']:.3f} all {out['all']['p10_blind']:.3f}", flush=True)
    print(summarize(store))


def summarize(store):
    rows = []
    for v in VARIANTS:
        row = {"variant": v,
               "PIT 1st": np.mean([r[v]["pit_1"] for r in store["rows"].values()]),
               "PIT 10th": np.mean([r[v]["pit_10"] for r in store["rows"].values()])}
        for k in TEAMS:
            row[f"{k}: P(top10)"] = np.mean([r[v][f"p10_{k}"] for r in store["rows"].values()])
            row[f"{k}: top10 share"] = np.mean([r["realised"][k][0] <= 10 for r in store["rows"].values()])
            row[f"{k}: E[prize]"] = np.sum([r[v][f"exp_{k}"] for r in store["rows"].values()])
            row[f"{k}: prize"] = np.sum([r["realised"][k][1] for r in store["rows"].values()])
        rows.append(row)
    return pd.DataFrame(rows).set_index("variant").T.round(3).to_string()


if __name__ == "__main__":
    main()
