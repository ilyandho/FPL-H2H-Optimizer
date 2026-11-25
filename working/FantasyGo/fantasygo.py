"""The 2023-24 FantasyGo season: the base team, the MVT at each r and the multiple-entry portfolio, scored in the real
contests.

The teams come from the structural pipeline's cached seasons (structural/src):

  base        the team with the highest expected points (lambda = 0; the same in every r cache)
  mvt_r{r}    Haugh & Singal's MVT with G = the simulated field's score at rank r (mvt.py, r = 0.5, 0.8, 0.9, 0.99, 1.0)
  portfolio   Algorithm 7's N entries (mvt_portfolio.py, N40_g8_r99): "strategic" (the MVT chain) and "benchmark"
              (the highest expected points, diversified the same way), scored for N = 1, 5, 10, 20, 40

Every gameweek with a usable contest from GW6 (29: GW6-21 and 26-38; GW22-25 hold other rounds' contests). Each entry
is scored on the contest's points (mvt.actual_points) and inserted into the real contest: rank = 1 + the real entries
with at least as many points (a tie goes to the real entry) + our other entries with more points. Per entry:

  rank pct    rank / the contest's entries (ours included)
  beat rate   share of weeks with more points than the real field's score at rank r (r = 0.5, 0.8, 0.9, 0.99; r = 1.0 is
              the winning score), i.e. the cut each MVT aims at
  prize       the week's real prize schedule at our rank, and FantasyGo's top-10 schedule (40, 25, 12.5, 7.5, 5, 3,
              2.5, 2, 1.5, 1 % of the pool, our entries' R17.8125 added to the pool in GW2-21); profit = prize - R20
              per entry; "average entrant" = the pool per real entry (what an average entry got back)

MVT - base is compared per gameweek (points and rank pct) with the paired t and a circular block bootstrap.

Usage:
  python fantasygo.py            # writes outputs/fantasygo.txt and outputs/weeks.csv, outputs/portfolio_weeks.csv
"""

import os
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "structural/src"
OUT = HERE / "outputs"
sys.path.insert(0, str(SRC))
os.chdir(SRC)

import mvt  # noqa: E402
import mvt_portfolio  # noqa: E402

R_LEVELS = (0.5, 0.8, 0.9, 0.99, 1.0)
CUTS = (0.5, 0.8, 0.9, 0.99, 1.0)
SIZES = (1, 5, 10, 20, 40)
FEE = mvt_portfolio.ENTRY_FEE
TOP10 = np.array(mvt_portfolio.TOP10_PCT) / 100
POOL_PER_ENTRY_ADDED = 17.8125  # each entry's share of the fee that goes into the pool, GW2-21


def name(r):
    return f"mvt_r{round(r * 100)}"


def score(points, gw, inp):
    """Our entries' points (array) in gameweek gw's real contest: ranks, rank pct, prizes and the cuts beaten."""
    real = inp.managers[inp.managers["gameweek"] == gw]
    real_pts = real["points"].to_numpy()
    prizes = np.sort(real["prize"].to_numpy())[::-1]
    pool = real["prize_pool"].iloc[0] + (POOL_PER_ENTRY_ADDED * len(points) if gw <= 21 else 0.0)
    rank = np.array([1 + (real_pts >= p).sum() + (points > p).sum() for p in points])
    n = len(real) + len(points)
    prize = np.where(rank <= len(prizes), prizes[np.minimum(rank, len(prizes)) - 1], 0.0)
    top10 = np.where(rank <= 10, TOP10[np.minimum(rank, 10) - 1] * pool, 0.0)
    order = np.sort(real_pts)[::-1]
    cuts = {c: order[max(int(np.ceil((1 - c) * len(order))), 1) - 1] for c in CUTS}
    return {"rank": rank, "entries": n, "prize": prize, "prize_top10": top10, "cuts": cuts,
            "avg_entrant": real["prize"].sum() / len(real)}


def single_entries(inp, s):
    """One row per gameweek and single-entry manager."""
    caches = {r: pickle.loads(replace(s, r_pct=r).cache().read_bytes())["results"] for r in R_LEVELS}
    rows = []
    for gw in mvt.gameweeks(inp, s):
        teams = {"base": (caches[0.8][gw]["teams"][0.0], caches[0.8][gw]["frontier"], 0.0)}
        for r in R_LEVELS:
            res = caches[r][gw]
            teams[name(r)] = (res["teams"][res["lambda_star"]], res["frontier"], res["lambda_star"])
        for m, (team, frontier, lam) in teams.items():
            pts = mvt.team_points(team, gw, inp)
            sc = score(np.array([pts]), gw, inp)
            f = frontier.set_index("lambda").loc[lam]
            rows.append({"gw": gw, "manager": m, "lambda": lam, "pred": f["mean"], "points": pts,
                         "rank": int(sc["rank"][0]), "entries": sc["entries"], "rank_pct": sc["rank"][0] / sc["entries"],
                         "prize": sc["prize"][0], "prize_top10": sc["prize_top10"][0], "avg_entrant": sc["avg_entrant"],
                         **{f"beat_{round(c * 100)}": pts > v for c, v in sc["cuts"].items()},
                         **{f"cut_{round(c * 100)}": v for c, v in sc["cuts"].items()}})
    return pd.DataFrame(rows)


def portfolio_entries(inp):
    """One row per gameweek, chain and N: the best of our first N entries, and the prizes of all N."""
    d = mvt_portfolio.out_dir(mvt_portfolio.PSettings())
    rows = []
    for f in sorted(d.glob("gw*.pkl"), key=lambda f: int(f.stem[2:])):
        res = pickle.loads(f.read_bytes())
        gw = res["gw"]
        ids = res["cands"]["fpl_id"].map(mvt.actual_points(gw, inp)).fillna(0).to_numpy()
        for chain in ("strategic", "benchmark"):
            pts_all = (res[chain]["W"] * 2).round() / 2 @ ids
            for n in SIZES:
                pts = pts_all[:n]
                sc = score(pts, gw, inp)
                best = pts.max()
                rows.append({"gw": gw, "chain": chain, "N": n, "best_points": best, "mean_points": pts.mean(),
                             "best_rank": int(sc["rank"].min()), "entries": sc["entries"],
                             "best_rank_pct": sc["rank"].min() / sc["entries"], "prize": sc["prize"].sum(),
                             "prize_top10": sc["prize_top10"].sum(), "cost": n * FEE,
                             "avg_entrant": n * sc["avg_entrant"],
                             **{f"beat_{round(c * 100)}": best > v for c, v in sc["cuts"].items()}})
    return pd.DataFrame(rows)


def block_bootstrap(d, block=4, n_boot=10_000, seed=0):
    """Circular block bootstrap of the mean of one weekly difference series: SE, t, 95% interval and two-sided p."""
    rng = np.random.default_rng(seed)
    n = len(d)
    k = -(-n // block)
    idx = (rng.integers(0, n, size=(n_boot, k))[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n] % n
    boot, mean = d[idx].mean(1), d.mean()
    se = boot.std(ddof=1)
    return {"boot t": mean / se, "CI low": np.quantile(boot, 0.025), "CI high": np.quantile(boot, 0.975),
            "p": float((np.abs(boot - mean) >= abs(mean)).mean())}


def season_table(w):
    """Season totals per single-entry manager."""
    g = w.groupby("manager", sort=False)
    t = pd.DataFrame({
        "weeks": g.size(), "points / week": g["points"].mean(), "median rank pct": g["rank_pct"].median(),
        "mean rank pct": g["rank_pct"].mean(),
        **{f"beat r={c}": g[f"beat_{round(c * 100)}"].mean() for c in CUTS},
        "top 10": g["rank"].apply(lambda x: int((x <= 10).sum())), "wins": g["rank"].apply(lambda x: int((x == 1).sum())),
        "prize": g["prize"].sum(), "prize top10": g["prize_top10"].sum(), "cost": g.size() * FEE,
        "avg entrant": g["avg_entrant"].sum()})
    t["profit"], t["profit top10"] = t["prize"] - t["cost"], t["prize top10"] - t["cost"]
    t["prize / avg entrant"] = t["prize"] / t["avg entrant"]
    return t


def paired(w):
    """Each MVT - base, per gameweek: points and rank pct (lower is better), paired t and block bootstrap."""
    base = w[w.manager == "base"].set_index("gw")
    rows = []
    for r in R_LEVELS:
        m = w[w.manager == name(r)].set_index("gw")
        for col in ("points", "rank_pct", "prize"):
            d = (m[col] - base[col]).to_numpy(dtype=float)
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std() > 0 else np.nan
            rows.append({"pair": f"{name(r)} - base", "measure": col, "per week": d.mean(), "t": t,
                         **block_bootstrap(d), "weeks better": int((d < 0).sum() if col == "rank_pct" else (d > 0).sum()),
                         "weeks same": int((d == 0).sum())})
    return pd.DataFrame(rows).set_index(["pair", "measure"])


def portfolio_table(p):
    g = p.groupby(["chain", "N"])
    t = pd.DataFrame({
        "weeks": g.size(), "best points / week": g["best_points"].mean(), "mean points / week": g["mean_points"].mean(),
        "median best rank pct": g["best_rank_pct"].median(),
        **{f"beat r={c}": g[f"beat_{round(c * 100)}"].mean() for c in CUTS},
        "top 10": g["best_rank"].apply(lambda x: int((x <= 10).sum())),
        "wins": g["best_rank"].apply(lambda x: int((x == 1).sum())),
        "prize": g["prize"].sum(), "prize top10": g["prize_top10"].sum(), "cost": g["cost"].sum(),
        "avg entrant": g["avg_entrant"].sum()})
    t["profit"], t["profit top10"] = t["prize"] - t["cost"], t["prize top10"] - t["cost"]
    t["prize / avg entrant"] = t["prize"] / t["avg entrant"]
    return t


def main():
    s = mvt.Settings()
    inp = mvt.Inputs(s)
    w = single_entries(inp, s)
    p = portfolio_entries(inp)
    OUT.mkdir(exist_ok=True)
    w.to_csv(OUT / "weeks.csv", index=False)
    p.to_csv(OUT / "portfolio_weeks.csv", index=False)
    fmt = {"float_format": lambda v: f"{v:.3f}"}
    lines = [f"FantasyGo 2023-24, {w.gw.nunique()} gameweeks ({w.gw.min()}-{w.gw.max()}, GW22-25 excluded), "
             f"entry fee R{FEE:.0f}", "", "single entries:", season_table(w).T.to_string(**fmt), "",
             "MVT - base per gameweek (rank pct: lower is better; block bootstrap, blocks of 4 weeks):",
             paired(w).to_string(**fmt), "", "portfolio (best of the first N entries; prizes over all N):",
             portfolio_table(p).T.to_string(**fmt)]
    text = "\n".join(lines)
    (OUT / "fantasygo.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
