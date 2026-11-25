"""Haugh & Singal's multiple-entry portfolio (their Algorithm 7) for the 2023-24 FantasyGo contests.

FantasyGo accepts many entries per manager (up to about 40 a gameweek; 78% of 2023-24's winners entered more than one
team), and its prizes are top-heavy. The paper's edge in top-heavy contests came from N = 50 diversified entries, so
mvt.py's single entry cannot show it. Per gameweek, entries are added one at a time:

  for i = 1..N:
    for lambda in the grid: w_lambda = argmax w'mu + lambda (w'Sigma w - 2 w'cov(delta, G))      (risk-seeking sign:
                            subject to the FantasyGo rules and sharing at most gamma players      the entries are out of
                            with each entry chosen so far                                         the money, Assn. 5.1)
    lambda* = the lambda whose w_lambda adds most to R(W*), the portfolio's expected prize
    W* = W* + w_lambda*

  R(W)   Monte Carlo over mvt.py's draws of delta and the simulated field (same seed, field and G correction as
         mvt.py at r_pct = 0.99): each draw ranks the field's teams and our entries together and pays each of our
         entries the prize of its rank; the prize per rank is the latest earlier contest's (the schedule is known
         before the deadline, its size roughly), zero beyond its paid places. The field's scores are shifted and
         widened by mvt.g_correction (the real top teams outscore the simulated ones), the noise common to the
         field in a draw
  cov    cov(delta, G) with G the field's 3rd best score (r_pct = 0.99, the top 1%; Assumption 5.2: about the same
         for every paid rank near the top)
  gamma  8 (the paper's C - 3 with C = 11 players)

The benchmark portfolio is the same greedy chain with lambda = 0 only: the highest expected points, diversified by
gamma (the paper's benchmark). The greedy chain does not depend on N, so the first n entries of the N = 40 portfolio
are the n-entry portfolio.

Scoring: our entries are inserted into the real contest's ranking (the real entries behind them move down a place)
and paid the real prize schedule of that gameweek.

Usage (from src/):
  python mvt_portfolio.py --gws 6-10          # writes ../outputs/23-24/mvt/portfolio/<tag>/gw<k>.pkl
  python mvt_portfolio.py --evaluate          # the season's table for N = 1, 5, 10, 20, 40
"""

import argparse
import pickle
from dataclasses import dataclass, replace
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd

import config
import mvt


ENTRY_FEE = 20.0  # FantasyGo's entry fee (R20); R17.8125 of it goes into the pool (pool = R9,500 + 17.8125 x entries
# in GW2-21; a flat guaranteed R9,500 from GW26), so FantasyGo keeps 10.9% and adds the guarantee
TOP10_PCT = (40, 25, 12.5, 7.5, 5, 3, 2.5, 2, 1.5, 1)  # FantasyGo's top-10 payout, % of the pool (the 2023-24 data's
# own schedules differ: top 3 then R25 to about 20th, from GW27 about 60 places)


@dataclass
class PSettings:
    n_entries: int = 40
    gamma: int = 8
    lambda_grid: tuple = (0.0, 0.01, 0.02, 0.05, 0.1)
    r_pct: float = 0.99  # the G whose covariance with delta enters the frontier
    time_limit: int = 15
    threads: int = 2

    def tag(self):
        return f"N{self.n_entries}_g{self.gamma}_r{round(self.r_pct * 100)}"


def out_dir(ps):
    return Path(config.paths("23-24").mvt_dir) / "portfolio" / ps.tag()


def payout(gw, inp, n):
    """The prize of each rank 1..n: the latest earlier contest's schedule, zero beyond its paid places."""
    prev = mvt.last_contest_before(gw, inp)
    p = inp.managers.loc[inp.managers["gameweek"] == prev, "prize"].sort_values(ascending=False).to_numpy()
    out = np.zeros(n)
    out[:min(n, len(p))] = p[:n]
    return out


def expected_prize(F, S, R):
    """Mean over draws of the prize won by our entries' scores F (draws x entries) against the field's S
    (draws x teams): rank of an entry = 1 + field teams above it + our other entries above it."""
    above = (S[:, :, None] > F[:, None, :]).sum(1) + (F[:, None, :] > F[:, :, None]).sum(2)
    return R[np.minimum(above, len(R) - 1)].sum(1).mean()


class Frontier:
    """One Gurobi model per gameweek and chain: the FantasyGo rules, the overlap cuts added as entries are chosen, and
    the objective set per lambda."""

    def __init__(self, cands, mu, sigma, cov, ps):
        self.env = gp.Env(params={"OutputFlag": 0})
        n = len(cands)
        pos, club, value = cands["pos"].to_numpy(), cands["team"].to_numpy(), cands["value"].to_numpy(dtype=float)
        m = self.m = gp.Model(env=self.env)
        m.Params.TimeLimit, m.Params.Threads, m.Params.NonConvex = ps.time_limit, ps.threads, 2
        self.pick, self.cap, self.vice = (m.addMVar(n, vtype=gp.GRB.BINARY) for _ in range(3))
        w = self.w = self.pick + (mvt.CAPTAIN - 1) * self.cap + (mvt.VICE - 1) * self.vice
        m.addConstr(self.cap + self.vice <= self.pick)
        m.addConstr(self.pick.sum() == mvt.XI_SIZE)
        m.addConstr(self.cap.sum() == 1)
        m.addConstr(self.vice.sum() == 1)
        m.addConstr(value @ self.pick <= mvt.BUDGET)
        for p, (lo, hi) in mvt.FORMATION.items():
            idx = np.where(pos == p)[0]
            m.addConstr(self.pick[idx].sum() >= lo)
            m.addConstr(self.pick[idx].sum() <= hi)
        for t in np.unique(club):
            m.addConstr(self.pick[np.where(club == t)[0]].sum() <= mvt.MAX_PER_TEAM)
        vals, vecs = np.linalg.eigh(sigma)
        keep = vals > 1e-9 * vals.max()
        L = vecs[:, keep] * np.sqrt(vals[keep])
        y = m.addMVar(L.shape[1], lb=-gp.GRB.INFINITY)
        m.addConstr(y == L.T @ w)
        self.mean, self.risk = mu @ w, y @ y - 2 * (cov @ w)
        self.cands, self.ps = cands, ps

    def solve(self, lam, start=None):
        self.m.setObjective(self.mean + lam * self.risk, gp.GRB.MAXIMIZE)
        if start is not None:
            self.pick.Start, self.cap.Start, self.vice.Start = (start > 0) * 1.0, (start == 2) * 1.0, (start == 1.5) * 1.0
        self.m.optimize()
        if self.m.SolCount == 0:
            return None, str(self.m.Status)
        pk, cp, vc = (np.round(v.X) > 0.5 for v in (self.pick, self.cap, self.vice))
        status = {gp.GRB.OPTIMAL: "optimal", gp.GRB.TIME_LIMIT: "timelimit"}.get(self.m.Status, str(self.m.Status))
        return pk + (mvt.CAPTAIN - 1) * cp + (mvt.VICE - 1) * vc, status

    def exclude(self, w):
        """Later entries share at most gamma players with w."""
        self.m.addConstr((w > 0) * 1.0 @ self.pick <= self.ps.gamma)


def run_gameweek(gw, inp, s, ps, verbose=True):
    rng = np.random.default_rng(s.seed + gw)  # mvt.run_gameweek's field and draws
    players, mu, sigma = mvt.build_mu_sigma(gw, inp, s)
    W_o = mvt.field_matrix(gw, players, inp, s, rng)
    shift, widen = mvt.g_correction(gw, inp, s)
    delta, G, cov = mvt.monte_carlo(mu, sigma, W_o, s, rng)  # uncorrected: cov(delta, G) does not depend on it
    S = delta @ W_o.T
    noise_rng = np.random.default_rng(s.seed + gw + 300_000)
    if widen > 1:  # as monte_carlo widens G, a noise common to the field in each draw
        S = S + noise_rng.normal(0, np.sqrt(widen ** 2 - 1) * G.std(), size=(len(S), 1))
    S = S + shift
    R = payout(gw, inp, len(W_o) + ps.n_entries + 1)
    idx = mvt.shortlist(players, s.shortlist)
    cands = players.reset_index().iloc[idx].reset_index(drop=True)
    mu_c, sig_c, cov_c, d_c = mu[idx], sigma[np.ix_(idx, idx)], cov[idx], delta[:, idx]

    out = {"gw": gw, "cands": cands[["fpl_id", "pos", "team", "value", "pred"]], "field_size": len(W_o),
           "shift": shift, "widen": widen}
    for chain, grid in [("strategic", ps.lambda_grid), ("benchmark", (0.0,))]:
        fr = Frontier(cands, mu_c, sig_c, cov_c, ps)
        W, F, log, start = [], np.zeros((len(d_c), 0)), [], None
        for i in range(ps.n_entries):
            best = None
            for lam in grid:
                w, status = fr.solve(lam, start)
                if w is None:
                    continue
                start = w
                value = expected_prize(np.column_stack([F, d_c @ w]), S, R)
                if best is None or value > best[0]:
                    best = (value, lam, w, status)
            if best is None:
                break
            value, lam, w, status = best
            W.append(w)
            F = np.column_stack([F, d_c @ w])
            fr.exclude(w)
            log.append({"entry": i + 1, "lambda": lam, "status": status, "R": value, "mean": mu_c @ w,
                        "sd": float(np.sqrt(w @ sig_c @ w))})
        out[chain] = {"W": np.array(W), "log": pd.DataFrame(log)}
        if verbose:
            lg = out[chain]["log"]
            print(f"GW{gw} {chain}: {len(W)} entries, R(1) {lg.R.iloc[0]:.0f} R(10) {lg.R.iloc[min(9, len(lg) - 1)]:.0f} "
                  f"R(N) {lg.R.iloc[-1]:.0f}, lambda* {lg['lambda'].value_counts().to_dict()}", flush=True)
    return out


def top10_prize(rank, gw, n_real, n, pool):
    """Our entries' prizes under TOP10_PCT on the week's pool, our own entries' contribution added where the pool grew
    with entries (GW <= 21)."""
    pool = pool + (17.8125 * n if gw <= 21 else 0.0)
    pct = np.array(TOP10_PCT) / 100
    return float((pct[rank[rank <= 10] - 1] * pool).sum())


def evaluate(ps, s, inp, sizes=(1, 5, 10, 20, 40)):
    rows = []
    for f in sorted(out_dir(ps).glob("gw*.pkl")):
        res = pickle.loads(f.read_bytes())
        gw = res["gw"]
        real = inp.managers[inp.managers["gameweek"] == gw]
        real_pts = real["points"].to_numpy()
        prizes = np.sort(real["prize"].to_numpy())[::-1]
        pool_per_entry = real["prize"].sum() / len(real)
        pts_map = mvt.actual_points(gw, inp)
        ids = res["cands"]["fpl_id"].map(pts_map).fillna(0).to_numpy()
        for chain in ("strategic", "benchmark"):
            W = res[chain]["W"]
            pts = (W * 2).round() / 2 @ ids
            for n in sizes:
                if n > len(pts):
                    continue
                ours = pts[:n]
                # rank among the real field and our other entries; ties: the real entry keeps the better rank
                rank = np.array([1 + (real_pts >= p).sum() + (ours > p).sum() for p in ours])
                won = np.where(rank <= len(prizes), prizes[np.minimum(rank, len(prizes)) - 1], 0.0)
                rows.append({"gw": gw, "chain": chain, "N": n, "best_rank": rank.min(), "entries": len(real) + n,
                             "win": rank.min() == 1, "top3": rank.min() <= 3,
                             "top1pct": rank.min() <= max(1, int(np.ceil(0.01 * len(real)))),
                             "prize": won.sum(), "stake_at_pool_rate": n * pool_per_entry, "cost": n * ENTRY_FEE,
                             "prize_top10": top10_prize(rank, gw, len(real), n, real["prize_pool"].iloc[0]),
                             "mean_pts": ours.mean(), "max_pts": ours.max(),
                             "unique": len({tuple(np.flatnonzero(w)) for w in W[:n]})})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gws", help="e.g. 6-10")
    parser.add_argument("--n-entries", type=int, default=40)
    parser.add_argument("--gamma", type=int, default=8)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    ps = PSettings(n_entries=args.n_entries, gamma=args.gamma)
    s = replace(mvt.Settings(), r_pct=ps.r_pct)
    inp = mvt.Inputs(s)
    if args.evaluate:
        e = evaluate(ps, s, inp)
        e.to_csv(out_dir(ps) / "evaluation.csv", index=False)
        t = e.groupby(["chain", "N"]).agg(weeks=("gw", "size"), wins=("win", "sum"), top3=("top3", "sum"),
                                          top1pct=("top1pct", "sum"), median_best_rank=("best_rank", "median"),
                                          prize=("prize", "sum"), prize_top10=("prize_top10", "sum"),
                                          cost=("cost", "sum"), avg_entrant=("stake_at_pool_rate", "sum"),
                                          mean_pts=("mean_pts", "mean"), unique=("unique", "mean"))
        t["return"], t["return_top10"] = t.prize / t.cost, t.prize_top10 / t.cost
        t["vs_avg_entrant"] = t.prize / t.avg_entrant  # 1 = what an average entry won (pool / entries)
        print(t.round(2).to_string())
        return
    gws = mvt.gameweeks(inp, s)
    if args.gws:
        lo, _, hi = args.gws.partition("-")
        gws = [g for g in gws if int(lo) <= g <= int(hi or lo)]
    d = out_dir(ps)
    d.mkdir(parents=True, exist_ok=True)
    for gw in gws:
        f = d / f"gw{gw}.pkl"
        if f.exists() and not args.recompute:
            continue
        f.write_bytes(pickle.dumps(run_gameweek(gw, inp, s, ps)))


if __name__ == "__main__":
    main()
