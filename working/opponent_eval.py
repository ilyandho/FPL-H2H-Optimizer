"""Chapter 4.3: does the H2H opponent model predict the opponent, and what does each of its parts add?

For every opponent-week the 20 league teams (baseline.LEAGUE) meet (gameweeks 6-38 without the opponent's chips, each
(opponent, gameweek) once), W_o is built four ways and scored against what the opponent actually fielded:

  full         the thesis's model: selection fit (his own where one exists, else the league-average manager),
               keep/buy, sparse C-vine, under his free transfers and squad value
  independent  the same margins, players sampled independently (no C-vine): what the copula adds
  average      the league-average manager's fit instead of his own (only where a personal fit exists, i.e. the own
               team's opponents): what the individual fit adds
  hold         his previous gameweek's squad, with certainty (the fallback): what the whole squad model adds

Squad metrics (W_o weighted by the sampler's counts): expected overlap with his real squad (of 15), the probability
given to his exact squad, and the expected share of his real transfers-in that W_o contains. Score metrics (the full
benchmark: W_o, every XI and captain by the fitted softmaxes, lineups.benchmark): log score and PIT of his realised
points under the normal mixture, and mu_G.

Writes outputs/opponent_eval.pkl after every opponent-week (re-running resumes).

Usage (ds_env, from working/):  OMP_NUM_THREADS=2 python opponent_eval.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "Optimizer"))
import baseline  # noqa: E402  (chdirs to structural/src)
import h2h  # noqa: E402
import lineups  # noqa: E402

OUT = HERE / "Optimizer" / "outputs" / "opponent_eval.pkl"
VARIANTS = ("full", "independent", "average", "hold")


def squad_metrics(counts, real, prev):
    tot = sum(counts.values())
    real, prev = set(real), set(prev)
    bought = real - prev
    overlap = sum(c * len(set(sq) & real) for sq, c in counts.items()) / tot
    exact = sum(c for sq, c in counts.items() if set(sq) == real) / tot
    buys = (sum(c * len(set(sq) & bought) for sq, c in counts.items()) / tot / len(bought)) if bought else np.nan
    n_tr = sum(c * len(set(sq) - prev) for sq, c in counts.items()) / tot
    return {"overlap": overlap, "p_exact": exact, "buys_found": buys, "n_buys_real": len(bought),
            "n_buys_pred": n_tr, "n_squads": len(counts)}


def score_metrics(bm, x):
    sd = np.sqrt(np.maximum(bm["Q"], 1e-6))
    p = bm["p"] / bm["p"].sum()
    logs = float(logsumexp(np.log(np.maximum(p, 1e-300)) + norm.logpdf(x, bm["mu_w"], sd)))
    pit = float(p @ norm.cdf((x - bm["mu_w"]) / sd))
    return {"log_score": logs, "pit": pit, "mu_G": bm["mu_G"], "sd_G": float(np.sqrt(bm["var_G"])),
            "z": (x - bm["mu_G"]) / np.sqrt(bm["var_G"])}


def evaluate(opp, gw, inp, s, lm, has_personal):
    league = inp.league
    orig = league.opponent_squads
    real_sq = league.squad_players(opp, gw)
    prev_sq = league.squad_players(opp, gw - 1)
    x = h2h.real_points(opp, gw, inp)
    if real_sq is None or prev_sq is None or x is None:
        return None
    pr = baseline.problem(gw, inp, s, set(prev_sq), 0.0)
    out = {}
    for v in VARIANTS:
        if v == "average" and not has_personal:
            continue
        seen = {}

        def patched(g, team_id, **kw):
            if v == "hold":
                res = None
            else:
                kw["fit_mode"] = "average" if v == "average" else kw.get("fit_mode", "personal")
                res = orig(g, team_id, copula=(v != "independent"), **kw)
            seen["res"] = res
            return res

        league.opponent_squads = patched
        try:
            bm = lineups.benchmark(gw, opp, pr, inp, s, lm)
        finally:
            league.opponent_squads = orig
        res = seen.get("res")
        counts = res["squad_counts"] if res and res["n_unique"] else {tuple(sorted(prev_sq)): 1}
        out[v] = {**squad_metrics(counts, real_sq, prev_sq), **score_metrics(bm, float(x)),
                  "fallback": bool(bm["fallback"]), "n_dep_pairs": (res or {}).get("n_dep_pairs", 0)}
    return {"opponent": opp, "gw": gw, "points": float(x), **{f"{v}": m for v, m in out.items()}}


def main():
    h2h._ENV = h2h.gp.Env(params={"OutputFlag": 0, "Threads": 2})
    rows = pickle.loads(OUT.read_bytes()) if OUT.exists() else {}
    for team in baseline.LEAGUE:
        s = h2h.Settings(own_team_id=0 if team == baseline.OWN_TEAM else team)
        inp = h2h.Inputs(s)
        lm = lineups.LineupModel(inp)
        gws = list(range(6, 39))
        chips = set(baseline.chip_weeks(inp, {g: inp.opponent_of[g] for g in gws}))
        for gw in gws:
            opp = inp.opponent_of[gw]
            if gw in chips or (opp, gw) in rows:
                continue
            fits = inp.league._fit_file(gw)
            has_personal = bool(fits) and all(f.get("team_id") == opp for f in fits.values())
            r = evaluate(opp, gw, inp, s, lm, has_personal)
            if r is None:
                continue
            rows[(opp, gw)] = {**r, "team": team, "personal": has_personal}
            OUT.write_bytes(pickle.dumps(rows))
            f, h = r["full"], r["hold"]
            print(f"team {team} GW{gw} opp {opp}: overlap full {f['overlap']:.2f} indep "
                  f"{r['independent']['overlap']:.2f} hold {h['overlap']:.2f} | log score full {f['log_score']:.2f} "
                  f"indep {r['independent']['log_score']:.2f} hold {h['log_score']:.2f}", flush=True)
    print(f"{len(rows)} opponent-weeks")


if __name__ == "__main__":
    main()
