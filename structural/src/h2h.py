"""Mean-variance (MVT) and base optimizers for the 2024-25 FPL head-to-head league, on the structural model.

Ported from 24-25/optimizer/notebooks/optimization.ipynb (cells 13-23) with mu and Sigma from structural.py instead of
the old xP predictor, so the points are FPL points, as in 23-24's mvt.py:

  mu_delta     pred, the structural model's expected points for the gameweek (both fixtures in a double gameweek)
  Sigma_delta  a team factor (Haugh & Singal, Appendix C; the notebook's sigma.py): each player's errors
               (total_points - pred, per match) standardized and pooled over every pair of teammates in the earlier
               gameweeks into one correlation per position pair, fitted by rho(p, q) = a_p a_q, so
               Sigma = B B' + diag(sd^2 (1 - a^2)), B[i, club(i)] = sd_i a_pos(i); sd_i = mvt.player_std. Rank <= 20 off
               the diagonal, so the MIQP keeps all of it exactly (y = L'v with L L' = Sigma - diag, the diagonal as
               the linear term sum_i d_i (x_i + 3 c_i))
  W_o          the opponent's likely squads (h2h_opponents.League.opponent_squads: selection model, keep/buy, C-vine),
               each scored on its rule-based XI and captain; G^(r') the r-th best over the distinct squads
  decision     a 15-player squad within the budget (squad value + money in the bank) and the free transfers, at most 3
               per club (the notebook's MIQP lacked this rule), an XI in a valid formation and a captain:
                 base  lambda = 0 (most expected points); transfers when that adds more than xp_margin points
                 mvt   max mean -/+ lambda (variance - 2 cov(delta, G)) over the lambda grid, the lambda with the highest
                       Monte Carlo Prob(beat G^(r')); transfers only when that beats holding by transfer_margin
  season       our squad, free-transfer bank and bank carried week to week from the real GW5 squad (GW6 is the first
               gameweek with league fits); no hits, no chips
  solver       Gurobi (gurobipy): one model per gameweek and transfer limit, re-optimized for each lambda; the convex
               branch (sense = +1) is a convex MIQP, the nonconvex one (sense = -1) Gurobi's nonconvex MIQP. (The
               notebook used SCIP; the first runs of this script did too, saved without the "_gurobi" suffix.)

Real scores come from the Fantasy Football Scout picks (h2h_data.py): every league manager's actual XI, captain, vice
and bench order, with automatic substitutions, the vice rule, Triple Captain (multiplier 3) and Bench Boost (bench
multipliers 1). Transfer hits are not in the data and are not deducted. Our simulated squads are scored the same way.

Usage (from src/):
  python h2h.py                      # base and MVT, GW6-38, r = 0.5
  python h2h.py --methods mvt --r-pct 0.8 --gws 6-10
"""

import argparse
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import gurobipy as gp
from scipy.optimize import least_squares

import config
import h2h_opponents as ho
import mvt
import structural

SQUAD_SLOTS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
FORMATION_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
N_STARTERS, MAX_PER_CLUB, FT_CAP = 11, 3, 5
POS_ORDER = ["GK", "DEF", "MID", "FWD"]


@dataclass
class Settings:
    season: str = "24-25"
    history: tuple = ("23-24",)  # structural predictions trained with these seasons
    r_pct: float = 0.5  # G^(r'): the r-th best of the opponent's distinct likely squads (0.5 = the median one)
    n_samples_opp: int = 20_000
    n_mc: int = 2000
    lambda_grid: tuple = tuple(np.concatenate([[0.0], np.geomspace(0.01, 5.0, 10)]).round(6))
    transfer_margin: float = 0.03  # mvt: Prob(beat) a transfer must add over holding
    xp_margin: float = 0.01  # base: expected points a transfer must add
    mip_time_limit: int = 30  # Gurobi's time limit per solve, seconds
    min_history: int = 3
    start_gw: int = 6
    end_gw: int = 38
    seed: int = 42
    # the opponent's chips in his likely squads (h2h_chips): "" none, "hazard" the chance he plays each chip he has
    # left from a model known at the deadline, "population" the FFS holders' share playing it that week (an upper
    # bound, known only after the deadline); per likely squad a chip is drawn with those chances: Triple Captain
    # (captain x3), Bench Boost (bench scores), Free Hit / Wildcard (the week's best legal squad within his budget)
    opponent_chips: str = ""
    own_team_id: int = 0  # the league manager whose team is managed (0: config's own_team_id, team 205)
    # the opponents' selection model: "personal", each opponent's own manager fit where one exists (only the opponents
    # team 205 met have one), else the league-average manager; "average", the league-average manager for everyone
    opponent_fit: str = "personal"
    # the benchmark G: "opponent", G^(r') of this week's opponent's likely squads; "elite", the score of the league's
    # elite consensus team (elite_team_row): the elite_k managers top of the real H2H table before the gameweek (us
    # left out), each manager's likely squads projected to this week by the opponent model (opponent_squads) and
    # averaged into an expected XI and captain row, the elite_k rows averaged; G = elite_mix G_opponent +
    # (1 - elite_mix) G_elite, so lambda and the sign are chosen to beat the strong managers' team
    benchmark: str = "opponent"
    elite_k: int = 4
    elite_mix: float = 0.0

    def cache(self, method):
        """Result file of a method (the "_gurobi" suffix keeps the first, SCIP-solved runs' files apart)."""
        hist = f"_history_{'+'.join(self.history)}" if self.history else ""
        chips = f"_oppchips-{self.opponent_chips}" if self.opponent_chips else ""
        team = f"_team{self.own_team_id}" if self.own_team_id else ""
        fit = "_avgfit" if self.opponent_fit == "average" else ""
        bench = (f"_elite{self.elite_k}" + (f"mix{self.elite_mix:g}" if self.elite_mix else "")
                 if self.benchmark == "elite" else "")
        return (Path(config.paths(self.season).h2h_dir)
                / f"h2h_{method}_r{round(self.r_pct * 100)}{hist}{chips}{fit}{bench}{team}_gurobi.pkl")


# ---------------------------------------------------------------------------- inputs
class Inputs:
    """Predictions, league, fixtures and the real picks of the league's managers."""

    def __init__(self, s: Settings):
        p = config.paths(s.season)
        preds = pd.read_csv(config.output_path(s.season, s.history))
        if "fixture_no" in preds.columns:
            preds = structural.per_gameweek(preds)
        self.preds = preds[preds["has_target"]].copy()
        self.opponents = mvt.load_opponents(p.vaastav_cache)
        self.league = ho.League(s.season)
        self.own = s.own_team_id or config.SEASONS[s.season]["own_team_id"]
        self.opponent_of = ho.opponents_by_round(s.season, self.own)
        self.league_season = s.season
        self.managers = list(config.SEASONS[s.season]["league_managers"])
        picks = pd.read_csv(f"{p.ffs_dir}/picks.csv.gz")
        self.picks = picks[picks["team_id"].isin(self.managers)]
        pdata = self.league.player_data
        # every player's position and club (also without a fixture in a gameweek)
        self.position = {**{e: ho.SHORT[pos] for e, pos in self.league.position.items()},
                         **self.preds.drop_duplicates("fpl_id").set_index("fpl_id")["pos"].to_dict()}
        self.club = pdata.sort_values("round").drop_duplicates("element", keep="last").set_index("element")["team"]

    def week(self, gw):
        """Round gw's per-player outcome (minutes, total_points over all his fixtures) and prediction."""
        return self.preds[self.preds["round"] == gw].drop_duplicates("fpl_id").set_index("fpl_id")

    def latest(self, gw):
        """Each player's latest league row at or before gw (price in tenths, position, club)."""
        d = self.league.player_data
        return d[d["round"] <= gw].sort_values("round").drop_duplicates("element", keep="last").set_index("element")


# ---------------------------------------------------------------------------- mu and Sigma
def team_loadings(table):
    """a_p per position: least squares (weighted by pairs) of rho_same(p, q) = a_p a_q."""
    if "same" not in table.index.get_level_values("rel"):
        return dict.fromkeys(POS_ORDER, 0.0)
    same = table.loc["same"]
    i = np.array([POS_ORDER.index(p) for p, _ in same.index])
    j = np.array([POS_ORDER.index(q) for _, q in same.index])
    w = np.sqrt(same["n_pairs"].to_numpy() / same["n_pairs"].sum())
    fit = least_squares(lambda a: w * (a[i] * a[j] - same["rho"].to_numpy()), x0=np.full(4, 0.4), bounds=(0, 0.95))
    return dict(zip(POS_ORDER, fit.x))


def build_mu_sigma(gw, inp, s):
    """(players, mu, Sigma, diag) for gameweek gw from gameweeks before it: Sigma = B B' + diag(diag)."""
    players = inp.week(gw)
    hist = inp.preds[inp.preds["round"] < gw]
    a_pos = team_loadings(mvt.correlation_table(hist, inp.opponents, s.min_history))
    a = players["pos"].map(a_pos).to_numpy()
    sd = mvt.player_std(players, hist, s.min_history)
    clubs = sorted(players["team"].unique())
    B = np.zeros((len(players), len(clubs)))
    B[np.arange(len(players)), [clubs.index(t) for t in players["team"]]] = sd * a
    diag = sd ** 2 * (1 - a ** 2)
    return players, players["pred"].to_numpy(), B @ B.T + np.diag(diag), diag


# ---------------------------------------------------------------------------- line-ups and real scores
def valid_formation(starters, pos):
    counts = pd.Series([pos[e] for e in starters]).value_counts()
    return counts.get("GK", 0) == 1 and all(counts.get(p, 0) >= k for p, k in FORMATION_MIN.items() if p != "GK")


def pick_lineup(squad, mu, pos):
    """(starters, captain, vice, bench) on predicted points: the best GK, the formation minimums, then the best
    outfielders; captain and vice the two highest starters; bench in auto-sub order (reserve GK, then by mu)."""
    def by_mu(players):
        return sorted(players, key=lambda e: -mu.get(e, 0.0))
    by_pos = {p: by_mu([e for e in squad if pos.get(e) == p]) for p in FORMATION_MIN}
    starters, others = by_pos["GK"][:1], []
    for p in ["DEF", "MID", "FWD"]:
        starters += by_pos[p][:FORMATION_MIN[p]]
        others += by_pos[p][FORMATION_MIN[p]:]
    others = by_mu(others)
    n_free = N_STARTERS - len(starters)
    starters += others[:n_free]
    captain, vice = by_mu(starters)[:2]
    return starters, captain, vice, by_pos["GK"][1:] + others[n_free:]


def lineup_weights(lineup):
    starters, captain, _, _ = lineup
    w = dict.fromkeys(starters, 1.0)
    w[captain] = 2.0
    return w


def lineup_points(lineup, week, pos, captain_mult=2.0, bench_boost=False):
    """(predicted, actual) points of a line-up: automatic substitutions (a starter with 0 minutes is replaced by the
    first bench player who played and keeps a valid formation), the captain's multiplier on the vice if the captain did
    not play; with bench_boost every squad player scores and there are no substitutions."""
    starters, captain, vice, bench = lineup

    def played(e):
        return e in week.index and week.at[e, "minutes"] > 0

    xi, bench = list(starters), list(bench)
    if bench_boost:
        xi, bench = xi + bench, []
    for e in list(starters):
        if bench_boost or played(e):
            continue
        for b in bench:
            trial = [b if x == e else x for x in xi]
            if played(b) and valid_formation(trial, pos):
                xi = trial
                bench.remove(b)
                break
    w = dict.fromkeys(xi, 1.0)
    armband = next((c for c in (captain, vice) if c in w and played(c)), None)
    if armband is not None:
        w[armband] = captain_mult
    rows = week.reindex(list(w))
    m = np.array(list(w.values()))
    return float(rows["pred"].fillna(0).to_numpy() @ m), float(rows["total_points"].fillna(0).to_numpy() @ m)


def real_lineup(team_id, gw, inp):
    """A league manager's actual line-up from the FFS picks: (lineup, captain multiplier, bench boost), or None."""
    pk = inp.picks[(inp.picks["team_id"] == team_id) & (inp.picks["gw"] == gw)].sort_values("slot")
    if len(pk) < 15:
        return None
    starters = pk.loc[pk["slot"] <= 11, "element"].tolist()
    bench = pk.loc[pk["slot"] > 11, "element"].tolist()
    cap = pk.loc[pk["is_captain"], "element"]
    vice = pk.loc[pk["is_vice_captain"], "element"]
    captain = int(cap.iloc[0]) if len(cap) else starters[0]
    mult = float(pk.loc[pk["element"] == captain, "multiplier"].iloc[0]) if len(cap) else 2.0
    bench_boost = bool((pk.loc[pk["slot"] > 11, "multiplier"] > 0).all())
    return (starters, captain, int(vice.iloc[0]) if len(vice) else starters[1], bench), max(mult, 2.0), bench_boost


def real_points(team_id, gw, inp):
    """A league manager's actual gameweek points from his FFS picks (no transfer hits), or None."""
    r = real_lineup(team_id, gw, inp)
    if r is None:
        return None
    lineup, mult, bb = r
    return lineup_points(lineup, inp.week(gw), inp.position, captain_mult=mult, bench_boost=bb)[1]


# ---------------------------------------------------------------------------- the MIQP
def build_gurobi(pr, t_limit, sense, s, min_mean=None, ft=0, hit_cost=4.0):
    """The MIQP in Gurobi: squad w, starters x <= w, captain c <= x, multipliers v = x + c; variance
    y'y + sum_i d_i (x_i + 3 c_i) with y = L'v exactly (L L' = Sigma - diag, rank <= 20). With min_mean (Haugh &
    Singal's Proposition 4.1(ii)): only squads whose expected score net of their hits, mu'v - hit_cost max(transfers -
    ft, 0), is at least min_mean (mu_G)."""
    mu, price, pos, club, owned, diag = pr["mu"], pr["price"], pr["pos"], pr["club"], pr["owned"], pr["diag"]
    P = len(mu)
    m = gp.Model(env=_gurobi_env())
    m.Params.TimeLimit = s.mip_time_limit
    w, x, c = (m.addMVar(P, vtype=gp.GRB.BINARY) for _ in range(3))
    v = x + c
    m.addConstr(w.sum() == 15)
    for p, k in SQUAD_SLOTS.items():
        m.addConstr(w[np.where(pos == p)[0]].sum() == k)
        idx = np.where(pos == p)[0]
        n_p = x[idx].sum()
        m.addConstr(n_p == FORMATION_MIN[p]) if p == "GK" else m.addConstr(n_p >= FORMATION_MIN[p])
    for team in np.unique(club):
        m.addConstr(w[np.where(club == team)[0]].sum() <= MAX_PER_CLUB)
    m.addConstr(price @ w <= pr["budget"])
    m.addConstr((1 - owned) @ w <= t_limit)
    m.addConstr(x <= w)
    m.addConstr(c <= x)
    m.addConstr(x.sum() == N_STARTERS)
    m.addConstr(c.sum() == 1)
    vals, vecs = np.linalg.eigh(pr["sigma"] - np.diag(diag))
    keep = vals > 1e-9 * max(vals.max(), 1e-12)
    L = vecs[:, keep] * np.sqrt(vals[keep])
    y = m.addMVar(L.shape[1], lb=-gp.GRB.INFINITY)
    m.addConstr(y == L.T @ v)
    if min_mean is not None:
        hit = m.addVar(lb=0.0)
        m.addConstr(hit >= (1 - owned) @ w - ft)
        m.addConstr(mu @ v - hit_cost * hit >= min_mean)
    return {"m": m, "w": w, "x": x, "c": c, "variance": y @ y + diag @ x + (3 * diag) @ c,
            "mean": mu @ v, "cov": pr["cov_g"] @ v, "sense": sense}


_ENV = None


def _gurobi_env():
    global _ENV
    if _ENV is None:
        _ENV = gp.Env(params={"OutputFlag": 0})
    return _ENV


def solve_gurobi(model, lam):
    """(w, v) maximizing mean - sense lam (variance - 2 cov), or None without a feasible solution."""
    m = model["m"]
    m.setObjective(model["mean"] - model["sense"] * lam * (model["variance"] - 2 * model["cov"]), gp.GRB.MAXIMIZE)
    m.optimize()
    if m.SolCount == 0:
        return None
    w = np.round(model["w"].X)
    return w, np.round(model["x"].X) + np.round(model["c"].X)


def solve(pr, t_limit, sense, lam, s):
    """The MIQP's (w, v) for lambda; the Gurobi model is built once per (transfer limit, sense) and kept in pr."""
    models = pr.setdefault("_gurobi", {})
    if (t_limit, sense) not in models:
        models[(t_limit, sense)] = build_gurobi(pr, t_limit, sense, s)
    return solve_gurobi(models[(t_limit, sense)], lam)


# ---------------------------------------------------------------------------- one gameweek
def gameweek_problem(gw, inp, s, squad, bank, opponent):
    """Everything both optimizers (and irl_h2h.py) need for a gameweek: the universe (every predicted player plus our
    squad's players without a fixture), mu, Sigma, prices, the opponent's XIs W_o and the Monte Carlo of G^(r')."""
    players, mu_all, sigma_all, diag_all = build_mu_sigma(gw, inp, s)
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
    price = latest.loc[universe, "value"].to_numpy(dtype=float)  # tenths of a million, as the notebook
    pos = np.array([inp.position[e] for e in universe])
    club = latest.loc[universe, "team"].to_numpy()
    owned = np.array([float(e in squad) for e in universe])
    mu_map, at = dict(zip(universe, mu)), {e: i for i, e in enumerate(universe)}

    opp = inp.league.opponent_squads(gw, opponent, n_samples=s.n_samples_opp, seed=s.seed, fit_mode=s.opponent_fit)
    fallback = opp is None or opp["n_unique"] == 0
    squads = ([tuple(sorted(inp.league.squad_players(opponent, gw - 1) or ()))] if fallback
              else list(opp["squad_counts"]))
    rows, lineups = [], []
    for sq in squads:
        row = np.zeros(n)
        lineup = pick_lineup(sq, mu_map, inp.position)
        for e, wt in lineup_weights(lineup).items():
            if e in at:
                row[at[e]] = wt
        rows.append(row)
        lineups.append(lineup)
    chip_probs, chips_drawn = {}, {}
    if s.opponent_chips:
        chip_probs = opponent_chip_probs(inp, opponent, gw, gw, s.opponent_chips)
        names = list(chip_probs)
        p = np.array([chip_probs[c] for c in names] + [max(1 - sum(chip_probs.values()), 0.0)])
        crng = np.random.default_rng(s.seed + gw + 7)
        fh_row = None
        for k, lineup in enumerate(lineups):
            c = (names + [None])[crng.choice(len(p), p=p / p.sum())]
            chips_drawn[c] = chips_drawn.get(c, 0) + 1
            if c == "tc" and lineup[1] in at:
                rows[k][at[lineup[1]]] = 3.0
            elif c == "bb":
                for e in lineup[3]:
                    if e in at:
                        rows[k][at[e]] = 1.0
            elif c in ("fh", "wc"):
                if fh_row is None:
                    fh_row = best_squad_row(mu, sigma, diag, price, pos, club, latest, squads[0], s)
                rows[k] = fh_row.copy()
    W_o = np.array(rows)
    rng = np.random.default_rng(s.seed)
    k = min(max(int(np.ceil((1 - s.r_pct) * len(W_o))), 1), len(W_o))
    delta = rng.multivariate_normal(mu, sigma, size=s.n_mc, method="eigh")
    G = np.sort(delta @ W_o.T, axis=1)[:, -k]
    elite = None
    if s.benchmark == "elite":
        elite_row, elite = elite_team_row(gw, inp, s, mu_map, at, n)
        G = s.elite_mix * G + (1 - s.elite_mix) * (delta @ elite_row)
        if s.elite_mix == 0:  # only the elite matter: W_o is their consensus row (mvt_plan's lambda_select_draws)
            W_o = elite_row[None, :]
    cov_g = ((delta - mu) * (G - G.mean())[:, None]).mean(axis=0)
    return {"gw": gw, "universe": universe, "mu": mu, "sigma": sigma, "diag": diag, "price": price, "pos": pos,
            "club": club, "owned": owned, "budget": float(latest.loc[sorted(squad), "value"].sum()) + bank,
            "mu_map": mu_map, "at": at, "W_o": W_o, "delta": delta, "G": G, "cov_g": cov_g,
            "fallback": fallback, "opponent": opponent, "n_opp_squads": len(W_o), "opp_chip_probs": chip_probs,
            "opp_chips_drawn": chips_drawn, "elite": elite}


def elite_managers(gw, inp, k):
    """The k league managers top of the real H2H table over GW1..gw-1 (H2H points, then points scored), us left out."""
    cache = inp.__dict__.setdefault("_elite_managers", {})
    if (gw, k) not in cache:
        gws = list(range(1, gw))
        table, _ = real_table(inp, gws) if gws else (pd.DataFrame({"manager": inp.managers}), None)
        cache[(gw, k)] = [m for m in table["manager"] if m != inp.own][:k]
    return cache[(gw, k)]


def elite_team_row(gw, inp, s, mu_map, at, n):
    """The elite consensus team as one row of multipliers over the universe: per elite manager, his likely squads this
    week (opponent model, weighted by how often each was sampled; his last squad when the model has no history), each
    on its rule-based XI and captain, averaged; then averaged over the managers. Returns (row, info)."""
    managers = elite_managers(gw, inp, s.elite_k)
    rows = []
    for e in managers:
        opp = inp.league.opponent_squads(gw, e, n_samples=s.n_samples_opp, seed=s.seed, fit_mode=s.opponent_fit)
        if opp is None or opp["n_unique"] == 0:
            counts = {tuple(sorted(inp.league.squad_players(e, gw - 1) or ())): 1}
        else:
            counts = opp["squad_counts"]
        row, total = np.zeros(n), 0
        for sq, c in counts.items():
            for p, wt in lineup_weights(pick_lineup(sq, mu_map, inp.position)).items():
                if p in at:
                    row[at[p]] += c * wt
            total += c
        rows.append(row / max(total, 1))
    return np.mean(rows, axis=0), {"managers": managers}


def opponent_chip_probs(inp, opponent, gw, fit_gw, mode):
    """{chip: P(opponent plays it in gameweek gw)} (h2h_chips.ChipModel), the model built once per Inputs."""
    if getattr(inp, "chips", None) is None:
        import h2h_chips

        inp.chips = h2h_chips.ChipModel(inp)
    return inp.chips.probs(opponent, gw, fit_gw=fit_gw, mode=mode)


def best_squad_row(mu, sigma, diag, price, pos, club, latest, opp_squad, s):
    """The XI and captain multipliers of the week's best legal squad (expected points) within the opponent's squad
    value: his likely team when he plays the Free Hit or a Wildcard."""
    budget = float(latest["value"].reindex(list(opp_squad)).fillna(0).sum())
    pr = {"mu": mu, "sigma": sigma, "diag": diag, "price": price, "pos": pos, "club": club,
          "owned": np.zeros(len(mu)), "budget": max(budget, 800.0), "cov_g": np.zeros(len(mu))}
    sol = solve_gurobi(build_gurobi(pr, 15, 1, s), 0.0)
    return np.zeros(len(mu)) if sol is None else sol[1]


def lineup_vector(lineup, pr):
    v = np.zeros(len(pr["universe"]))
    for e, wt in lineup_weights(lineup).items():
        if e in pr["at"]:
            v[pr["at"][e]] = wt
    return v


def lineup_from_solution(w, v, pr, inp):
    """The MIQP's squad, XI and captain; vice and bench order by predicted points."""
    u = pr["universe"]
    squad = [u[i] for i in range(len(u)) if w[i] > 0.5]
    starters = [u[i] for i in range(len(u)) if v[i] > 0.5]
    captain = u[int(np.argmax(v))]
    vice = next(e for e in sorted(starters, key=lambda e: -pr["mu_map"].get(e, 0)) if e != captain)
    rule_bench = pick_lineup(squad, pr["mu_map"], inp.position)[3]
    bench = [e for e in rule_bench if e not in starters] + [e for e in squad if e not in starters and e not in rule_bench]
    return squad, (starters, captain, vice, bench)


def run_gameweek(gw, inp, s, squad, ft, bank, method):
    """One decision: hold or the optimizer's squad, its line-up and its predicted / actual points."""
    pr = gameweek_problem(gw, inp, s, squad, bank, inp.opponent_of[gw])
    grid = [0.0] if method == "base" else list(s.lambda_grid)
    hold_lineup = pick_lineup(sorted(squad), pr["mu_map"], inp.position)
    hold_v = lineup_vector(hold_lineup, pr)
    prob_hold = float((pr["delta"] @ hold_v - pr["G"] > 0).mean())
    sense = 1 if method == "base" or pr["mu"] @ hold_v - pr["G"].mean() >= 0 else -1
    best = (-1.0, None, None, None)
    for lam in grid:
        sol = solve(pr, ft, sense, lam, s)
        if sol is None:
            continue
        prob = float((pr["delta"] @ sol[1] - pr["G"] > 0).mean())
        if prob > best[0]:
            best = (prob, sol[0], sol[1], lam)
    prob, w, v, lam = best
    if w is None:
        adopt = False
    elif method == "base":
        adopt = float(pr["mu"] @ v - pr["mu"] @ hold_v) > s.xp_margin
    else:
        adopt = prob > prob_hold + s.transfer_margin
    if adopt:
        new_squad, lineup = lineup_from_solution(w, v, pr, inp)
    else:
        new_squad, lineup, prob, lam = sorted(squad), hold_lineup, prob_hold, None
    return finish(gw, inp, pr, squad, new_squad, lineup, ft, prob, {"lambda_star": lam, "made_transfer": adopt,
                                                                     "prob_hold": prob_hold, "sense": sense})


def finish(gw, inp, pr, squad, new_squad, lineup, ft, prob, extra):
    """The rollout row of a decision: transfers, bank, predicted and actual points against the opponent's real ones."""
    week = inp.week(gw)
    latest = inp.latest(gw)
    n_tr = len(set(new_squad) - set(squad))
    pred, pts = lineup_points(lineup, week, inp.position)
    opp_pts = real_points(pr["opponent"], gw, inp)
    own_real = real_points(inp.own, gw, inp)
    return {"gw": gw, "opponent": pr["opponent"], "squad_before": sorted(squad), "squad": sorted(new_squad),
            "lineup": lineup, "n_transfers": n_tr, "ft_before": ft, "ft_after": min(FT_CAP, ft - n_tr + 1),
            "bank_after": pr["budget"] - float(latest.loc[sorted(new_squad), "value"].sum()),
            "prob_beat": prob, "pred": pred, "points": pts, "opponent_points": opp_pts, "real_points": own_real,
            "fallback_benchmark": pr["fallback"], "n_opp_squads": pr["n_opp_squads"], **extra}


# ---------------------------------------------------------------------------- the season
def run(s: Settings, method, gws=None, recompute=False, verbose=True, decide=None):
    """A rolling season, cached after every gameweek (resumes where it stopped). decide(gw, inp, s, squad, ft, bank)
    replaces run_gameweek (irl_h2h.py)."""
    inp = Inputs(s)
    decide = decide or (lambda gw, inp, s, squad, ft, bank: run_gameweek(gw, inp, s, squad, ft, bank, method))
    path = s.cache(method)
    cache = (pickle.loads(path.read_bytes()) if path.exists() and not recompute
             else {"settings": asdict(s), "rows": {}})
    for legacy in ("solver", "eig_tol"):  # settings of the SCIP version, no longer used
        cache["settings"].pop(legacy, None)
    cache["settings"].setdefault("opponent_chips", "")  # added later
    cache["settings"].setdefault("own_team_id", 0)
    cache["settings"].setdefault("opponent_fit", "personal")
    for k, v in (("benchmark", "opponent"), ("elite_k", 4), ("elite_mix", 0.0)):
        cache["settings"].setdefault(k, v)
    if cache["settings"] != asdict(s):
        raise ValueError(f"{path} was made with other settings; delete it or pass recompute=True")
    gws = list(gws or range(s.start_gw, s.end_gw + 1))
    squad = set(inp.league.squad_players(inp.own, s.start_gw - 1))
    ft, bank = inp.league.free_transfers(inp.own, s.start_gw), 0.0
    for gw in range(s.start_gw, max(gws) + 1):
        if gw not in cache["rows"]:
            if gw not in gws:
                raise ValueError(f"GW{gw} is needed before GW{max(gws)} (the season is sequential)")
            try:
                row = decide(gw, inp, s, squad, ft, bank)
            except Exception as e:  # an unsolvable week holds the squad, like a real week without a transfer
                if verbose:
                    print(f"[{method}] GW{gw} failed: {e!r}", flush=True)
                row = {"gw": gw, "status": "failed", "squad_before": sorted(squad), "squad": sorted(squad),
                       "n_transfers": 0, "ft_before": ft, "ft_after": min(FT_CAP, ft + 1), "bank_after": bank}
            cache["rows"][gw] = row
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pickle.dumps(cache))
            if verbose and "points" in row:
                r = row
                res = "W" if r["opponent_points"] is not None and r["points"] > r["opponent_points"] else (
                    "D" if r["points"] == r["opponent_points"] else "L")
                print(f"[{method}] GW{gw}: {r['n_transfers']} transfers, P(beat) {r['prob_beat']:.2f}, "
                      f"pred {r['pred']:.1f}, points {r['points']:.0f} vs opponent {r['opponent_points']} [{res}] "
                      f"| real us {r['real_points']}", flush=True)
        row = cache["rows"][gw]
        squad, ft, bank = set(row["squad"]), row["ft_after"], row["bank_after"]
    return pd.DataFrame([cache["rows"][g] for g in gws]).set_index("gw"), inp


# ---------------------------------------------------------------------------- evaluation
def h2h_result(a, b):
    return np.where(a > b, 3, np.where(a == b, 1, 0))


def real_table(inp, gws):
    """The league's real H2H table over gws on the FFS points (both legs' fixtures): points (3 / 1 / 0) and score."""
    fixtures = {}
    cfg = config.SEASONS["24-25"]
    leg1, leg2 = (ho.create_fixtures(list(cfg["league_managers"]), seed=sd) for sd in cfg["fixture_seeds"])
    for gw in gws:
        fixtures[gw] = leg1[gw] if gw <= 19 else leg2[gw - 19]
    pts = {(m, gw): real_points(m, gw, inp) for m in inp.managers for gw in gws}
    rows = []
    for m in inp.managers:
        own = np.array([pts[(m, g)] or 0 for g in gws], dtype=float)
        opp = np.array([pts[(fixtures[g][m], g)] or 0 for g in gws], dtype=float)
        rows.append({"manager": m, "h2h_points": int(h2h_result(own, opp).sum()), "score": own.sum(),
                     "wins": int((own > opp).sum())})
    return pd.DataFrame(rows).sort_values(["h2h_points", "score"], ascending=False).reset_index(drop=True), pts


def evaluate(rollouts, inp):
    """Per method, the season against the opponents' real scores, next to our real season; and where each method's
    season would have finished in the real league table (in place of our real team)."""
    gws = sorted(set.intersection(*[set(r.index[r["points"].notna()]) for r in rollouts.values()]))
    table, pts = real_table(inp, gws)
    out = {}
    real = np.array([pts[(inp.own, g)] for g in gws], dtype=float)
    opp = np.array([pts[(inp.opponent_of[g], g)] for g in gws], dtype=float)
    for name, own in [("real", real)] + [(m, r.loc[gws, "points"].to_numpy(dtype=float)) for m, r in rollouts.items()]:
        h2h = int(h2h_result(own, opp).sum())
        others = table[table["manager"] != inp.own]
        rank = 1 + int(((others["h2h_points"] > h2h) | ((others["h2h_points"] == h2h) & (others["score"] > own.sum())))
                       .sum())
        out[name] = {"points": own.sum(), "h2h_points": h2h, "wins": int((own > opp).sum()),
                     "draws": int((own == opp).sum()), "losses": int((own < opp).sum()), "league_rank": rank}
        if name != "real":
            r = rollouts[name].loc[gws]
            out[name] |= {"mean_prob_beat": r["prob_beat"].mean(), "transfers": int(r["n_transfers"].sum()),
                          "brier": float(((r["prob_beat"] - (own > opp)) ** 2).mean())}
    return pd.DataFrame(out).T, table, gws


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--methods", nargs="*", default=["base", "mvt"])
    parser.add_argument("--r-pct", type=float, default=0.5)
    parser.add_argument("--gws", help="e.g. 6-38 (default: the whole season from GW6)")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--team", type=int, default=0, help="the league manager whose team is managed (default 205)")
    parser.add_argument("--opponent-fit", default="personal", choices=["personal", "average"])
    args = parser.parse_args()
    s = Settings(r_pct=args.r_pct, own_team_id=args.team, opponent_fit=args.opponent_fit)
    gws = None
    if args.gws:
        lo, _, hi = args.gws.partition("-")
        gws = range(int(lo), int(hi or lo) + 1)
    rollouts = {}
    for method in args.methods:
        rollouts[method], inp = run(s, method, gws, args.recompute)
    summary, table, _ = evaluate(rollouts, inp)
    print(summary.to_string())


if __name__ == "__main__":
    main()
