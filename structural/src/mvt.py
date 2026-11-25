"""Mean-variance (MVT) optimizer for the 2023-24 FantasyGo contests, on the structural model's predictions.

Haugh & Singal's "beat the benchmark" optimizer, as in 23-24/optimizer/MVT/mvt.ipynb and sigma.py, with mu_delta
from structural.py instead of the direct models. Choose the team w that maximizes Prob{w'delta > G^(r')(W_o, delta)}:

  delta      the players' uncertain points, N(mu_delta, Sigma_delta)
  mu_delta   pred, the structural model's expected points (includes the chance of not playing)
  Sigma      Settings.sigma = "factor" (default): the 2024-25 one-factor team model B B' + D, as h2h.build_mu_sigma;
             "pairs": diag(sigma) rho diag(sigma): sigma = sd of each player's past errors (total_points - pred), rho one
             correlation per (same team / opponents, position pair), pooled over earlier gameweeks (Appendix C)
  W_o        the contest field: simulated with the team generator and the league selection model
             (field="simulated"), or the previous gameweek's real FantasyGo teams (field="previous"). The
             generator's alphas are tilted by p_play^gamma (fit_play_tilt, fitted on the earlier rounds' picks) and, with
             fixture_tilt, by n_fixtures^kappa as well (fit_tilts: the real managers load up on double-gameweek players)
  G^(r')     the score at rank r_pct of the field (0.8 = the top 20%), by Monte Carlo

Per gameweek: (1) mu, Sigma from gameweeks before gw; (2) W_o; (3) Monte Carlo draws of (delta, G^(r'));
(4) w(lambda) for each lambda in the grid, a mixed-integer quadratic program (Gurobi):
max w'mu -/+ lambda (w'Sigma w - 2 w'cov(delta, G)); (5) lambda* = the lambda whose team beats G^(r') in most draws.
lambda = 0 is the base optimizer (the team with the highest expected points) on the same shortlist.

FantasyGo: 11 players, captain x2, vice-captain x1.5, 1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD, budget 80, at most 3 per club,
unlimited changes each gameweek.

The players are every player with a fixture (the earlier pipeline kept those with minutes_rolling_3 > 30, which the
structural model's pred, weighted by the chance of playing, no longer needs); the decision is restricted to the
`shortlist` best-predicted players per position.

FantasyGo gameweeks 22-25 hold other rounds' contests (README, "FantasyGo data"): they are not solved or scored by
default (Settings.excluded_fg_gws), and the field of the next gameweek is sized and, for field="previous", taken from
the latest usable contest (GW21 for GW26).

Writes one cached result per setting in paths(season).mvt_dir.

Usage (from src/):
  python mvt.py                                  # 2023-24, simulated field, top 20%, predictions with 22-23 history
  python mvt.py --field previous --r-pct 0.5 --gws 6-10
"""

import argparse
import os
import pickle
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import config
import joblib
import numpy as np
import pandas as pd
import structural

BUDGET = 80.0
XI_SIZE = 11
FORMATION = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
FORMATIONS = [
    (d, m, 10 - d - m) for d in range(3, 6) for m in range(2, 6) if 1 <= 10 - d - m <= 3
]
MAX_PER_TEAM = 3
CAPTAIN, VICE = 2.0, 1.5
POS_ORDER = ["GK", "DEF", "MID", "FWD"]


@dataclass
class Settings:
    season: str = "23-24"
    history: tuple = (
        "22-23",
    )  # the structural predictions to use: structural.py --history ...
    field: str = "simulated"  # "simulated" (team generator) or "previous" (last gameweek's real teams)
    r_pct: float = 0.8  # rank to beat in the field: 0.8 = top 20% of the teams
    n_mc: int = 2000  # Monte Carlo draws
    lambda_grid: tuple = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
    shortlist: int = 30  # candidates per position, by expected points
    time_limit: int = 30  # seconds per solve
    min_history: int = 3  # gameweeks of errors a player needs for his own sd
    seed: int = 0
    # team generator (Algorithm 1)
    shared_p: bool = True  # one selection vector per position for the whole field
    q_stack: float = 0.35  # probability that a simulated manager stacks teammates
    budget_lb_quantile: float = 0.25  # simulated teams spend at least this quantile of the earlier real teams' cost
    budget_min_accept: float = 0.1  # ...lowered, in a week where it would reject more than 1 - this share of the
    # generator's teams, to the cost that this share reaches (a blank week such as GW29: 8 clubs play)
    budget_pilot: int = 2000  # teams drawn to check budget_min_accept
    play_tilt: bool = (
        True  # multiply the selection model's alphas by p_play^gamma (fit_play_tilt)
    )
    # ... and by n_fixtures^kappa, gamma and kappa fitted together (fit_tilts). Without it the simulated field held a
    # third to a half of the real field's double-gameweek players (GW28 11% of XI places against 32%, GW34 41% against
    # 81%) and its top-20% cut fell 11-36 points short of the real one there (7 in single weeks)
    fixture_tilt: bool = True
    g_shift: bool = True  # shift G by the mean real cut - simulated G of the earlier gameweeks (g_correction)
    g_widen: bool = True  # widen G by the spread of those gaps (g_correction)
    g_shrink: float = (
        5.0  # both corrections are trusted n / (n + g_shrink) with n earlier gameweeks
    )
    team_shift: bool = True  # also shift G by the base team's mean overprediction in earlier gameweeks (g_correction)
    polish: bool = (
        True  # polish every MIQP solution by swap-improvement (solve_for_lambda)
    )
    xp_anchor_pct: float = (
        75  # a stack's anchor must be predicted above this percentile
    )
    first_gw: int = (
        6  # gameweek 6 has errors from GW1-5 (history predictions start at GW1)
    )
    # FantasyGo gameweeks treated as missing (another round's contest, config.SEASONS): not solved or scored by
    # default, and left out of the generator's formations, budget bound and field size
    excluded_fg_gws: tuple = tuple(
        config.SEASONS["23-24"].get("fantasygo_excluded_gws", [])
    )
    # "mv": the lambda* team of the mean-variance frontier (Haugh & Singal's approximation); "saa": the team that beats
    # G^(r') in the most of n_saa Monte Carlo scenarios (sample average approximation of Prob(beat) itself, solve_saa),
    # drawn apart from the n_mc draws its Prob(beat) is then estimated on; "topheavy": Haugh & Singal's Algorithm 4
    # for a top-heavy contest (FantasyGo pays the top places): always the variance-seeking frontier
    # mu'w + lambda (w'Sigma w - 2 w'cov(delta, G)), cov averaged over the paid places' order statistics (their
    # Assumption 5.2), lambda* = the most expected prize money over the payout table of the previous contest
    # (topheavy_payout); G corrected per paid place (topheavy_correction); r_pct only names the run and sets the
    # report's target rank (0.96 ~ the 10th of ~230 entries)
    objective: str = "mv"
    # Sigma_delta: "factor" = the 2024-25 one-factor team model, B B' + D (h2h.team_loadings: same-club correlation
    # a_p a_q, none across clubs); "pairs" = one correlation per (same team / opponents, position pair), nearest PSD
    sigma: str = "factor"
    n_saa: int = 1000
    saa_time_limit: int = 180
    # Chapter 4.7, the optimizer's curse: sigma_shrink pulls each player's error variance to his position's median by
    # the empirical-Bayes weight (player_std); track_k lets the lambda > 0 teams differ from the week's lambda = 0
    # (blind) team in at most track_k players (a tracking constraint, Jagannathan & Ma's caps as shrinkage)
    sigma_shrink: bool = False
    track_k: int = None

    def cache(self):
        """Result file for these settings (the generator settings are the defaults unless named; the "_gurobi"
        suffix keeps the first, SCIP-solved runs' files apart)."""
        hist = f"_history_{'+'.join(self.history)}" if self.history else ""
        objective = "" if self.objective == "mv" else f"_{self.objective}"
        nfix = "_nfix" if self.fixture_tilt else ""
        sig = "_factor" if self.sigma == "factor" else ""
        extra = ("_sshrink" if self.sigma_shrink else "") + (f"_track{self.track_k}" if self.track_k is not None else "")
        return (
            Path(config.paths(self.season).mvt_dir)
            / f"mvt_{self.field}_r{round(self.r_pct * 100)}{hist}{objective}{nfix}{sig}{extra}_gurobi.pkl"
        )


# ---------------------------------------------------------------------------- inputs
class Inputs:
    """The season's predictions, fixtures and FantasyGo data."""

    def __init__(self, s: Settings):
        p = config.paths(s.season)
        if not os.path.exists(p.fantasygo):
            raise FileNotFoundError(
                f"no FantasyGo data for {s.season} ({p.fantasygo}); the MVT stage is 2023-24 only"
            )
        preds = pd.read_csv(config.output_path(s.season, s.history))
        if (
            "fixture_no" in preds.columns
        ):  # one row per fixture: sum a double gameweek's two
            preds = structural.per_gameweek(preds)
        preds = preds[preds.has_target].copy()
        preds["value"] = preds["value"] / 10
        self.preds = preds
        self.season = s.season
        picks = pd.read_csv(f"{p.fantasygo}/fg_manager_picks.csv")
        managers = pd.read_csv(f"{p.fantasygo}/fg_managers.csv")
        self.picks = picks[~picks["gameweek"].isin(s.excluded_fg_gws)]
        self.managers = managers[~managers["gameweek"].isin(s.excluded_fg_gws)]
        self.contest_gws = np.sort(self.picks["gameweek"].unique())
        self.alpha = joblib.load(
            f"{p.selection_dir}/league_alpha_draws.pkl"
        )  # selection.py
        self.opponents = load_opponents(p.vaastav_cache)
        # every player's position, also for picked players without a fixture that gameweek
        self.positions = (
            pd.concat(
                [
                    self.picks[["player_id", "position"]].set_axis(
                        ["fpl_id", "pos"], axis=1
                    ),
                    preds[["fpl_id", "pos"]],
                ]
            )
            .drop_duplicates("fpl_id")
            .set_index("fpl_id")["pos"]
        )


def last_contest_before(gw, inp):
    """The latest gameweek before gw with usable FantasyGo entries (gw - 1 unless that one is excluded)."""
    earlier = inp.contest_gws[inp.contest_gws < gw]
    return int(earlier.max()) if len(earlier) else None


def gameweeks(inp, s, gws=None):
    """gws, or by default every gameweek from s.first_gw with a usable FantasyGo contest."""
    if gws is not None:
        return list(gws)
    return [
        gw
        for gw in range(s.first_gw, int(inp.preds["round"].max()) + 1)
        if gw in inp.contest_gws
    ]


def load_opponents(src):
    """One row per (round, team, opp): two rows for a double gameweek, none for a blank."""
    names = pd.read_csv(f"{src}/teams.csv").set_index("id")["name"]
    fx = pd.read_csv(f"{src}/fixtures.csv").dropna(subset=["event"])
    fx["event"] = fx["event"].astype(int)
    home, away = fx["team_h"].map(names), fx["team_a"].map(names)
    return pd.concat(
        [
            pd.DataFrame({"round": fx["event"], "team": home, "opp": away}),
            pd.DataFrame({"round": fx["event"], "team": away, "opp": home}),
        ],
        ignore_index=True,
    )


# ---------------------------------------------------------------------------- step 1: mu and Sigma
def pos_pair(a, b):
    return tuple(sorted((a, b), key=POS_ORDER.index))


def standardised_errors(hist, min_history):
    """Each player's errors (total_points - pred) per match (a double gameweek's / sqrt 2), standardised by his own
    mean and sd."""
    e = hist[["round", "fpl_id", "pos", "team"]].copy()
    e["error"] = match_errors(hist)
    g = e.groupby("fpl_id")["error"]
    e["z"] = (e["error"] - g.transform("mean")) / g.transform("std")
    return e[(g.transform("size") >= min_history) & np.isfinite(e["z"])]


def correlation_table(hist, opponents, min_history):
    """rho per (relationship, position pair), pooled over every same-team / opposing pair."""
    e = standardised_errors(hist, min_history)
    same = e.merge(e, on=["round", "team"], suffixes=("_i", "_j"))
    same = same[same["fpl_id_i"] < same["fpl_id_j"]].assign(rel="same")
    opp = e.merge(opponents, on=["round", "team"]).merge(
        e, left_on=["round", "opp"], right_on=["round", "team"], suffixes=("_i", "_j")
    )
    opp = opp[opp["fpl_id_i"] < opp["fpl_id_j"]].assign(rel="opp")
    pairs = pd.concat([same, opp], ignore_index=True)
    pairs["pos_pair"] = [pos_pair(a, b) for a, b in zip(pairs["pos_i"], pairs["pos_j"])]
    return (
        (pairs["z_i"] * pairs["z_j"])
        .groupby([pairs["rel"], pairs["pos_pair"]])
        .agg(rho="mean", n_pairs="size")
    )


def nearest_psd_corr(corr, min_eig_frac=1e-4):
    vals, vecs = np.linalg.eigh((corr + corr.T) / 2)
    vals = np.clip(vals, max(min_eig_frac * vals.max(), 1e-8), None)
    c = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


def match_errors(hist):
    """total_points - pred per gameweek, on the scale of one match: divided by sqrt(n_fixtures), as the sum of a
    double gameweek's two errors has about twice the variance."""
    n = hist["n_fixtures"] if "n_fixtures" in hist.columns else 1
    return (hist["total_points"] - hist["pred"]) / np.sqrt(n)


def player_std(players, hist, min_history, shrink=False):
    """sd of each player's past errors per match, times sqrt(his fixtures in the gameweek); his position's median
    with fewer than min_history gameweeks, or the position's pooled error variance when no player has enough yet.
    shrink: each player's variance v (from n matches) pulled to the median m of players of his position and fifth of
    the week's mu by the empirical-Bayes weight noise / (noise + tau2), noise = 2 v^2 / (n - 1) (the sampling variance
    of a variance) and tau2 that group's spread of true variances, Var(v) - mean(noise) (floor 0: the median)."""
    errors = match_errors(hist)
    stats = errors.groupby(hist["fpl_id"]).agg(["var", "size"])
    var = stats.loc[stats["size"] >= min_history, "var"].reindex(players.index)
    pos_median = var[var > 0].groupby(players["pos"]).median()
    pooled = errors.groupby(hist["pos"]).var()
    pos_median = pos_median.reindex(pooled.index).fillna(pooled)
    if shrink:  # the target and tau2 per (position, fifth of the week's mu within it): like players, not the bench
        n_obs = stats["size"].reindex(players.index)
        noise = 2 * var ** 2 / (n_obs - 1).clip(lower=1)
        grp = players["pos"] + "_" + players.groupby("pos")["pred"].transform(
            lambda x: pd.qcut(x.rank(method="first"), 5, labels=False)).astype(str)
        ok = var > 0
        target = var[ok].groupby(grp[ok]).median()
        tau2 = (var[ok].groupby(grp[ok]).var() - noise[ok].groupby(grp[ok]).mean()).clip(lower=0)
        wgt = (noise / (noise + grp.map(tau2))).fillna(1)
        m = grp.map(target)
        var = var.where(var.isna() | ~ok | m.isna(), (1 - wgt) * var + wgt * m)
    n = players["n_fixtures"].to_numpy() if "n_fixtures" in players.columns else 1
    return np.sqrt(var.where(var > 0, players["pos"].map(pos_median)).to_numpy() * n)


def build_mu_sigma(gw, inp, s):
    """(players, mu_delta, Sigma_delta) for gameweek gw, from gameweeks before gw only."""
    preds = inp.preds
    players = preds[preds["round"] == gw].drop_duplicates("fpl_id").set_index("fpl_id")
    hist = preds[preds["round"] < gw]
    table = correlation_table(hist, inp.opponents, s.min_history)
    std = player_std(players, hist, s.min_history, shrink=s.sigma_shrink)
    if s.sigma == "factor":  # as h2h.build_mu_sigma: club factor with loading sd * a_pos, the rest idiosyncratic
        from h2h import team_loadings

        a = players["pos"].map(team_loadings(table)).to_numpy()
        clubs = sorted(players["team"].unique())
        B = np.zeros((len(players), len(clubs)))
        B[np.arange(len(players)), [clubs.index(c) for c in players["team"]]] = std * a
        return players, players["pred"].to_numpy(), B @ B.T + np.diag(std ** 2 * (1 - a ** 2))
    rho = table["rho"]
    pos, team = players["pos"].to_numpy(), players["team"].to_numpy()
    week = inp.opponents[inp.opponents["round"] == gw]
    plays = set(zip(week["team"], week["opp"]))
    lookup = {(rel, pp): r for (rel, pp), r in rho.items()}
    corr = np.eye(len(players))
    for a in range(len(players)):
        for b in range(a + 1, len(players)):
            if team[a] == team[b]:
                rel = "same"
            elif (team[a], team[b]) in plays:
                rel = "opp"
            else:
                continue
            corr[a, b] = corr[b, a] = lookup.get((rel, pos_pair(pos[a], pos[b])), 0.0)
    sigma = nearest_psd_corr(corr) * np.outer(std, std)
    return players, players["pred"].to_numpy(), (sigma + sigma.T) / 2


# ---------------------------------------------------------------------------- step 2: the field W_o
def formation_frequencies(gw, inp):
    """Share of each formation (DEF, MID, FWD) among the FantasyGo teams before gw (every valid one possible)."""
    earlier = inp.picks[inp.picks["gameweek"] < gw]
    counts = earlier.groupby(["row", "position"]).size().unstack(fill_value=0)
    shape = counts.reindex(columns=["DEF", "MID", "FWD"], fill_value=0).apply(
        tuple, axis=1
    )
    freq = shape.value_counts().reindex(FORMATIONS, fill_value=0) + 1
    return freq / freq.sum()


def budget_lower_bound(gw, inp, s):
    """The budget_lb_quantile of the real FantasyGo teams' cost in the gameweeks before gw."""
    preds = inp.preds
    earlier = inp.picks[
        (inp.picks["gameweek"] < gw) & (inp.picks["gameweek"] >= preds["round"].min())
    ]
    value = earlier.merge(
        preds[["round", "fpl_id", "value"]],
        left_on=["gameweek", "player_id"],
        right_on=["round", "fpl_id"],
        how="left",
    ).groupby(["gameweek", "row"])["value"]
    cost = value.sum()[value.count() == XI_SIZE]  # teams with every price known
    return float(cost.quantile(s.budget_lb_quantile))


def relaxed_budget_lb(budget_lb, pools, formations, anchor_cut, s, rng, p=None):
    """budget_lb, or lower if fewer than s.budget_min_accept of the generator's teams (under p, if shared) would meet
    it: then the (1 - budget_min_accept) quantile of their cost. Returns (bound, share of the pilot teams meeting the
    original bound). Uses its own random stream (rng.spawn), so the field's draws are the same as without it."""
    pilot = rng.spawn(1)[0]
    costs = []
    for _ in range(s.budget_pilot):
        team = simulate_team(
            pools, formations, anchor_cut, 0.0, s, pilot, p, return_cost=True
        )
        if team is not None:
            costs.append(team[2])
    if not costs:
        return budget_lb, np.nan
    costs = np.array(costs)
    share = float((costs >= budget_lb).mean())
    if share >= s.budget_min_accept:
        return budget_lb, share
    return min(budget_lb, float(np.quantile(costs, 1 - s.budget_min_accept))), share


def sharpen(alpha, power):
    """alpha^power per posterior draw (row), rescaled to the row's total."""
    sharp = np.exp(power * (np.log(alpha) - np.log(alpha).max(axis=1, keepdims=True)))
    return sharp * (alpha.sum(axis=1, keepdims=True) / sharp.sum(axis=1, keepdims=True))


def _tilt_rounds(inp):
    """Per contest round with alphas: per position, the log of the mean sharpened pick shares, log p_play and the
    real pick counts of the candidates with a fixture."""
    if getattr(inp, "_tilt_data", None) is None:
        inp._tilt_data = {}
        for r in inp.contest_gws:
            if r not in inp.alpha:
                continue
            pl = (
                inp.preds[inp.preds["round"] == r]
                .drop_duplicates("fpl_id")
                .set_index("fpl_id")
            )
            pk = inp.picks[inp.picks["gameweek"] == r]
            rows = []
            for pos, d in inp.alpha[r].items():
                keep = np.isin(d["fpl_id"], pl.index)
                ids = d["fpl_id"][keep]
                a = sharpen(d["alpha"][:, keep].astype(float), d["power"])
                rows.append(
                    (
                        np.log((a / a.sum(axis=1, keepdims=True)).mean(axis=0)),
                        np.log(np.clip(pl.loc[ids, "p_play"].to_numpy(), 0.01, 1.0)),
                        pk.loc[pk["position"] == pos, "player_id"]
                        .value_counts()
                        .reindex(ids)
                        .fillna(0)
                        .to_numpy(),
                        np.log(pl.loc[ids, "n_fixtures"].clip(lower=1).to_numpy()),
                    )
                )
            inp._tilt_data[r] = rows
    return inp._tilt_data


def fit_play_tilt(gw, inp):
    """gamma in share ~ alpha^kappa p_play^gamma: the multinomial maximum likelihood of the real pick counts of the
    contest rounds before gw, given the selection model's out-of-sample shares for them (0 with no earlier round).
    The selection model has no information on availability; p_play (structural model, from the pre-deadline injury
    and suspension flags) adds it: the real managers' picks play more often than the simulated ones
    (generator_checks.py)."""
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp

    cache = inp.__dict__.setdefault("_tilt", {})
    if gw not in cache:
        train = [x for r, rows in _tilt_rounds(inp).items() if r < gw for x in rows]
        if not train:
            cache[gw] = 0.0
        else:

            def nll(g):
                return -sum(
                    (c * (b + g * lp - logsumexp(b + g * lp))).sum()
                    for b, lp, c, _ in train
                )

            cache[gw] = float(
                minimize_scalar(nll, bounds=(-1.0, 5.0), method="bounded").x
            )
    return cache[gw]


def fit_tilts(gw, inp):
    """(gamma, kappa) in share ~ alpha^power p_play^gamma n_fixtures^kappa, the multinomial maximum likelihood of the
    real pick counts of the contest rounds before gw (as fit_play_tilt); kappa is 0 until an earlier round has a double
    gameweek."""
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    cache = inp.__dict__.setdefault("_tilts", {})
    if gw not in cache:
        train = [x for r, rows in _tilt_rounds(inp).items() if r < gw for x in rows]
        if not train:
            cache[gw] = (0.0, 0.0)
        elif not any((ln > 0).any() for _, _, _, ln in train):
            cache[gw] = (fit_play_tilt(gw, inp), 0.0)
        else:

            def nll(th):
                g, k = th
                return -sum(
                    (c * (b + g * lp + k * ln - logsumexp(b + g * lp + k * ln))).sum()
                    for b, lp, c, ln in train
                )

            res = minimize(
                nll,
                x0=[fit_play_tilt(gw, inp), 0.0],
                bounds=[(-1.0, 5.0), (-2.0, 6.0)],
                method="L-BFGS-B",
            )
            cache[gw] = (float(res.x[0]), float(res.x[1]))
    return cache[gw]


def candidate_pools(gw, players, inp, s=None):
    """Per position: the generator's candidates with a prediction this gameweek, and their sharpened alpha draws
    (tilted by p_play^gamma, and n_fixtures^kappa with s.fixture_tilt, at the same total so the field's concentration
    is unchanged)."""
    pools = {}
    gamma, kappa = 0.0, 0.0
    if s is not None and s.play_tilt:
        gamma, kappa = (
            fit_tilts(gw, inp) if s.fixture_tilt else (fit_play_tilt(gw, inp), 0.0)
        )
    for pos, d in inp.alpha[gw].items():
        keep = np.isin(d["fpl_id"], players.index)
        ids = d["fpl_id"][keep]
        alpha = sharpen(d["alpha"][:, keep].astype(float), d["power"])
        if gamma or kappa:
            tilt = (
                np.clip(players.loc[ids, "p_play"].to_numpy(), 0.01, 1.0) ** gamma
                * players.loc[ids, "n_fixtures"].clip(lower=1).to_numpy() ** kappa
            )
            tilted = alpha * tilt
            alpha = tilted * (
                alpha.sum(axis=1, keepdims=True) / tilted.sum(axis=1, keepdims=True)
            )
        pools[pos] = {
            "ids": ids,
            "alpha": alpha,
            "value": players.loc[ids, "value"].to_numpy(),
            "team": players.loc[ids, "team"].to_numpy(),
            "pred": players.loc[ids, "pred"].to_numpy(),
        }
    return pools


def selection_vector(alpha, rng):
    """p ~ Dirichlet(alpha), robust to very small alphas (gamma draws that all underflow to 0)."""
    g = rng.gamma(alpha)
    p = g / g.sum() if g.sum() > 0 else alpha / alpha.sum()
    p = np.maximum(p, 1e-12)
    return p / p.sum()


def stack(selected, p, pool_from, pool_to, anchor_cut, rng, replace_worst=True):
    """Replace a selected player of pool_to with a teammate of the best selected player of pool_from."""
    best = selected["from"][np.argmax(pool_from["pred"][selected["from"]])]
    if pool_from["pred"][best] < anchor_cut:
        return selected["to"]
    mates = np.where(
        (pool_to["team"] == pool_from["team"][best])
        & ~np.isin(np.arange(len(p)), selected["to"])
    )[0]
    if len(mates) == 0:
        return selected["to"]
    mate = rng.choice(mates, p=p[mates] / p[mates].sum())
    out = selected["to"].copy()
    out[np.argmin(pool_to["pred"][out]) if replace_worst else 0] = mate
    return out


def selection_vectors(pools, rng):
    """One posterior draw of the selection model and, per position, p ~ Dirichlet(alpha) under it."""
    k = rng.integers(len(next(iter(pools.values()))["alpha"]))
    return {pos: selection_vector(pools[pos]["alpha"][k], rng) for pos in pools}


def simulate_team(
    pools, formations, anchor_cut, budget_lb, s, rng, p=None, return_cost=False
):
    """One team (fpl ids, multipliers[, cost]), or None if it breaks the budget or club rules."""
    p = selection_vectors(pools, rng) if p is None else p
    form = formations.index[rng.choice(len(formations), p=formations.to_numpy())]
    sizes = {"GK": 1, "DEF": form[0], "MID": form[1], "FWD": form[2]}
    chosen = {
        pos: rng.choice(len(p[pos]), size=n, replace=False, p=p[pos])
        for pos, n in sizes.items()
    }
    if rng.random() < s.q_stack:
        chosen["MID"] = stack(
            {"from": chosen["FWD"], "to": chosen["MID"]},
            p["MID"],
            pools["FWD"],
            pools["MID"],
            anchor_cut,
            rng,
        )
        chosen["GK"] = stack(
            {"from": chosen["DEF"], "to": chosen["GK"]},
            p["GK"],
            pools["DEF"],
            pools["GK"],
            anchor_cut,
            rng,
            replace_worst=False,
        )
    ids = np.concatenate([pools[pos]["ids"][chosen[pos]] for pos in sizes])
    cost = sum(pools[pos]["value"][chosen[pos]].sum() for pos in sizes)
    clubs = np.concatenate([pools[pos]["team"][chosen[pos]] for pos in sizes])
    if (
        not (budget_lb <= cost <= BUDGET)
        or pd.Series(clubs).value_counts().max() > MAX_PER_TEAM
    ):
        return None
    pred = np.concatenate([pools[pos]["pred"][chosen[pos]] for pos in sizes])
    multiplier = np.ones(len(ids))
    order = np.argsort(-pred)
    multiplier[order[0]], multiplier[order[1]] = CAPTAIN, VICE
    return (ids, multiplier, cost) if return_cost else (ids, multiplier)


def simulate_field(
    gw, players, inp, s, rng, n_teams=None, max_tries=200_000, return_budget=False
):
    """n_teams simulated teams (default: the number of entries of the latest contest before gw) and the acceptance
    rate (and, with return_budget, {earlier-weeks bound, bound used, pilot share meeting the earlier bound})."""
    n_teams = (
        n_teams
        or inp.picks.loc[
            inp.picks["gameweek"] == last_contest_before(gw, inp), "row"
        ].nunique()
    )
    pools = candidate_pools(gw, players, inp, s)
    formations = formation_frequencies(gw, inp)
    anchor_cut = np.percentile(players["pred"], s.xp_anchor_pct)
    p = selection_vectors(pools, rng) if s.shared_p else None
    lb_earlier = budget_lower_bound(gw, inp, s)
    budget_lb, pilot_share = relaxed_budget_lb(
        lb_earlier, pools, formations, anchor_cut, s, rng, p
    )
    teams, tries = [], 0
    while len(teams) < n_teams and tries < max_tries:
        tries += 1
        team = simulate_team(pools, formations, anchor_cut, budget_lb, s, rng, p)
        if team is not None:
            teams.append(team)
    if return_budget:
        return (
            teams,
            len(teams) / tries,
            {"lb_earlier": lb_earlier, "lb": budget_lb, "pilot_share": pilot_share},
        )
    return teams, len(teams) / tries


def entries(gw, inp):
    """The FantasyGo teams of gameweek gw as (fpl ids, multipliers)."""
    picks = inp.picks[inp.picks["gameweek"] == gw]
    return [
        (
            g["player_id"].to_numpy(),
            np.where(g["is_captain"], CAPTAIN, np.where(g["is_vice"], VICE, 1.0)),
        )
        for _, g in picks.groupby("row")
    ]


def field_matrix(gw, players, inp, s, rng):
    """W_o: one row of multipliers over `players` per team of the field."""
    teams = (
        simulate_field(gw, players, inp, s, rng)[0]
        if s.field == "simulated"
        else entries(last_contest_before(gw, inp), inp)
    )
    return teams_matrix(teams, players)


def teams_matrix(teams, players):
    """W_o for a list of teams (fpl ids, multipliers); players without a row in `players` (no fixture) score 0."""
    position = {e: i for i, e in enumerate(players.index)}
    W_o = np.zeros((len(teams), len(players)))
    for k, (ids, multiplier) in enumerate(teams):
        for e, m in zip(ids, multiplier):
            if e in position:
                W_o[k, position[e]] += m
    return W_o


# ---------------------------------------------------------------------------- step 3: Monte Carlo
def field_gap(t, inp, s):
    """(real cut, mean and sd of the unshifted simulated G) for contest gameweek t: the field, Monte Carlo draws and
    seed of run_gameweek(t). G before the correction does not depend on it, so no gameweek depends on another's run."""
    rng = np.random.default_rng(s.seed + t)
    players, mu, sigma = build_mu_sigma(t, inp, s)
    W_o = field_matrix(t, players, inp, s, rng)
    _, G, _ = monte_carlo(mu, sigma, W_o, s, rng)
    pts = (
        inp.managers.loc[inp.managers["gameweek"] == t, "points"]
        .sort_values(ascending=False)
        .to_numpy()
    )
    k = max(int(np.ceil((1 - s.r_pct) * len(pts))), 1)  # the winner at r_pct = 1
    return float(pts[k - 1]), float(G.mean()), float(G.std())


def base_team_gap(t, inp, s):
    """(predicted, actual points) of gameweek t's base team (lambda = 0: the highest expected points on the
    shortlist). Its points fall short of the prediction on average: the optimizer picks the players whose mu is
    highest, which are also the ones whose mu errs upward most (each position's top player by mu is overpredicted by
    0.9 points, the captain by 2.5), and recalibrating mu with pre-deadline signals picks no better team."""
    players, mu, sigma = build_mu_sigma(t, inp, s)
    idx = shortlist(players, s.shortlist)
    cands = players.reset_index().iloc[idx].reset_index(drop=True)
    team, _ = solve_for_lambda(
        cands, mu[idx], sigma[np.ix_(idx, idx)], np.zeros(len(idx)), 0.0, s
    )
    w = (team["w"] * 2).round() / 2
    return float((team["pred"] * w).sum()), float(
        (team["fpl_id"].map(actual_points(t, inp)).fillna(0) * w).sum()
    )


def g_correction(gw, inp, s, verbose=False):
    """(shift, widen) for G in gameweek gw, from the contest gameweeks from s.first_gw before gw with one fixture per
    team (double gameweeks are left out until the predictions cover both fixtures):

      shift  the mean of the gaps (real cut - mean simulated G), shrunk to 0 by n / (n + s.g_shrink) with n gameweeks
      team   with s.team_shift, minus the base teams' mean (actual - predicted points), shrunk the same way: our own
             team falls short of its prediction (base_team_gap), so beating G takes more than mu says; added to shift
      widen  k = sqrt(mean squared deviation of the gaps from their mean / mean sd(G)^2), shrunk to 1 the same way:
             if G's distribution were right, the gaps would vary as much as G does; they vary more (a common
             high- or low-scoring week the model misses), so G gets independent noise with sd sqrt(k^2 - 1) sd(G)

    Why the real cut beats G on average: the players the real managers pick outscore mu, more so the more popular
    they are (+0.9 points at a pick share above 50%), while mu is unbiased over all players, so the managers know
    something at the deadline (team news) the predictions do not; pre-deadline signals (FPL ownership, transfers, last
    week's FantasyGo picks) do not recover it. Neither correction changes cov(delta, G), so the frontier is the same;
    they change Prob(beat) and lambda*. Each gameweek's gap is cached in mvt_dir/g_gaps_r{pct}.pkl."""
    nfix = "_nfix" if s.fixture_tilt else ""
    sig = "_factor" if s.sigma == "factor" else ""
    shr = "_sshrink" if s.sigma_shrink else ""
    path = (
        Path(config.paths(s.season).mvt_dir)
        / f"g_gaps_r{round(s.r_pct * 100)}{nfix}{sig}{shr}_gurobi.pkl"
    )
    key = {
        k: v
        for k, v in asdict(s).items()
        if not k.startswith("g_") and k not in ("team_shift", "polish", "track_k") + SAA_SETTINGS
    }
    if not s.sigma_shrink:  # made before the setting existed
        key.pop("sigma_shrink", None)
    cache = inp.__dict__.get("_g_gaps")
    if cache is None or cache["settings"] != key:
        cache = pickle.loads(path.read_bytes()) if path.exists() else None
        if cache is not None:
            cache["settings"].pop("solver", None)  # a setting of the SCIP version
            cache["settings"].setdefault("sigma", "pairs")  # made before the factor model
        if (
            cache is None
            or cache["settings"] != key
            or any(len(v) != 3 for v in cache["gaps"].values())
        ):
            cache = {"settings": key, "gaps": {}}
        inp._g_gaps = cache
    o = inp.opponents
    double = set(
        o.groupby(["round", "team"])
        .size()
        .loc[lambda x: x > 1]
        .index.get_level_values(0)
    )
    weeks = [t for t in inp.contest_gws if s.first_gw <= t < gw and t not in double]
    cache.setdefault("base", {})
    for t in weeks:
        if t not in cache["gaps"]:
            cache["gaps"][t] = field_gap(t, inp, s)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pickle.dumps(cache))
            if verbose:
                print(
                    f"  gap GW{t}: cut {cache['gaps'][t][0]}, G {cache['gaps'][t][1]:.1f}",
                    flush=True,
                )
        if s.team_shift and t not in cache["base"]:
            cache["base"][t] = base_team_gap(t, inp, s)
            path.write_bytes(pickle.dumps(cache))
            if verbose:
                print(
                    f"  base team GW{t}: predicted {cache['base'][t][0]:.1f}, actual {cache['base'][t][1]}",
                    flush=True,
                )
    if not weeks:
        return 0.0, 1.0
    g = np.array([cache["gaps"][t] for t in weeks])
    gaps, sds = g[:, 0] - g[:, 1], g[:, 2]
    trust = len(weeks) / (len(weeks) + s.g_shrink)
    shift = trust * gaps.mean() if s.g_shift else 0.0
    if s.team_shift:
        b = np.array([cache["base"][t] for t in weeks])
        shift -= trust * (b[:, 1] - b[:, 0]).mean()
    k_raw = (
        np.sqrt(((gaps - gaps.mean()) ** 2).mean() / (sds**2).mean())
        if len(weeks) > 1
        else 1.0
    )
    widen = (
        max(1.0, 1 + trust * (k_raw - 1)) if s.g_widen else 1.0
    )  # widened only, never narrowed
    return float(shift), float(widen)


def monte_carlo_top(mu, sigma, W_o, s, rng, n_top, correction=(0.0, 1.0)):
    """Draws of delta and the field's n_top best scores per draw (descending), all shifted by the same amount and
    widened by the same independent noise per draw (g_correction, measured at s.r_pct), and cov(delta, G) averaged
    over the top min(n_top, 10) places (Haugh & Singal's Assumption 5.2)."""
    delta = rng.multivariate_normal(mu, sigma, size=s.n_mc, method="eigh")
    scores = delta @ W_o.T
    n_top = min(n_top, scores.shape[1])
    T = -np.sort(-scores, axis=1)[:, :n_top]
    shift, widen = correction
    if widen > 1:
        T = T + rng.normal(0.0, np.sqrt(widen**2 - 1) * T[:, min(9, n_top - 1)].std(), size=(len(T), 1))
    T = T + shift
    k = min(10, n_top)
    cov = np.mean([((delta - mu) * (T[:, d] - T[:, d].mean())[:, None]).mean(axis=0) for d in range(k)], axis=0)
    return delta, T, cov


def topheavy_place_gap(t, inp, s):
    """Contest gameweek t's real 1st-10th best scores minus the mean of the unshifted simulated ones (the field,
    draws and seed of run_gameweek(t)), as a vector of 10."""
    rng = np.random.default_rng(s.seed + t)
    players, mu, sigma = build_mu_sigma(t, inp, s)
    W_o = field_matrix(t, players, inp, s, rng)
    _, T, _ = monte_carlo_top(mu, sigma, W_o, s, rng, 10)
    real = inp.managers.loc[inp.managers["gameweek"] == t, "points"].sort_values(ascending=False).to_numpy()[:10]
    return real - T.mean(axis=0)


def topheavy_correction(gw, inp, s, verbose=False):
    """Fix 2 (calibration_check_fg.py): a shift per paid place, from the contest gameweeks before gw (one fixture per
    team, as g_correction): shrink * mean over them of [place gap (topheavy_place_gap) - (base team's actual -
    predicted)], shrink = n / (n + s.g_shrink). Returns 10 shifts (1st..10th; the 10th's is used beyond it).
    Cached per gameweek in mvt_dir/topheavy_gaps{nfix}{sig}_gurobi.pkl."""
    g_correction(gw, inp, s, verbose)  # fills inp._g_gaps["base"] (the base team's gap) for the weeks before gw
    nfix = "_nfix" if s.fixture_tilt else ""
    sig = "_factor" if s.sigma == "factor" else ""
    shr = "_sshrink" if s.sigma_shrink else ""
    path = Path(config.paths(s.season).mvt_dir) / f"topheavy_gaps{nfix}{sig}{shr}_gurobi.pkl"
    key = {k: v for k, v in asdict(s).items()
           if not k.startswith("g_") and k not in ("team_shift", "polish", "r_pct", "track_k") + SAA_SETTINGS}
    if not s.sigma_shrink:  # made before the setting existed
        key.pop("sigma_shrink", None)
    cache = pickle.loads(path.read_bytes()) if path.exists() else None
    if cache is None or cache["settings"] != key:
        cache = {"settings": key, "gaps": {}}
    o = inp.opponents
    double = set(o.groupby(["round", "team"]).size().loc[lambda x: x > 1].index.get_level_values(0))
    weeks = [t for t in inp.contest_gws if s.first_gw <= t < gw and t not in double]
    for t in weeks:
        if t not in cache["gaps"]:
            cache["gaps"][t] = topheavy_place_gap(t, inp, s)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pickle.dumps(cache))
    if not weeks:
        return np.zeros(10)
    base = inp._g_gaps["base"]
    net = np.array([cache["gaps"][t] - (base[t][1] - base[t][0]) for t in weeks])
    return len(weeks) / (len(weeks) + s.g_shrink) * net.mean(axis=0)


def topheavy_payout(gw, inp):
    """The prize per rank (descending, positive prizes only) of the latest contest before gw: known at the deadline,
    unlike gw's own (its pool depends on gw's entries)."""
    prev = last_contest_before(gw, inp)
    f = inp.managers[inp.managers["gameweek"] == prev].sort_values("points", ascending=False)
    pr = f["prize"].to_numpy(dtype=float)
    return pr[pr > 0]


def expected_prize(F, T, prizes):
    """Mean prize when our score is F (per draw) and the field's best scores are T: rank = 1 + the field teams above."""
    above = (T > F[:, None]).sum(axis=1)
    padded = np.append(prizes, 0.0)
    return float(padded[np.minimum(above, len(prizes))].mean())


def monte_carlo(mu, sigma, W_o, s, rng, correction=(0.0, 1.0)):
    """Draws of delta, G (shifted and widened by correction = (shift, widen), g_correction) and cov(delta, G)."""
    delta = rng.multivariate_normal(mu, sigma, size=s.n_mc, method="eigh")
    k = min(
        max(int(np.ceil((1 - s.r_pct) * len(W_o))), 1), len(W_o)
    )  # r' counted from the top
    G = np.sort(delta @ W_o.T, axis=1)[:, -k]
    shift, widen = correction
    if widen > 1:  # independent of delta, so cov(delta, G) is unchanged
        G = G + rng.normal(0.0, np.sqrt(widen**2 - 1) * G.std(), size=len(G))
    G = G + shift
    cov_delta_G = ((delta - mu) * (G - G.mean())[:, None]).mean(axis=0)
    return delta, G, cov_delta_G


# ---------------------------------------------------------------------------- step 4: w(lambda)
def shortlist(players, n):
    best = (
        players.reset_index()
        .groupby("pos", group_keys=False)
        .apply(lambda g: g.nlargest(n, "pred"))
    )
    return best.index.to_numpy()  # positions in `players`


def team_objective(w, mu, sigma, cov_delta_G, lam, sense):
    return mu @ w - sense * lam * (w @ sigma @ w - 2 * cov_delta_G @ w)


def feasible(picks, cands):
    """The FantasyGo rules for an XI given as positions in cands (formation, budget, at most 3 per club)."""
    pos, club, value = (
        cands["pos"].to_numpy(),
        cands["team"].to_numpy(),
        cands["value"].to_numpy(),
    )
    counts = pd.Series(pos[picks]).value_counts()
    return (
        len(set(picks)) == XI_SIZE
        and all(lo <= counts.get(p, 0) <= hi for p, (lo, hi) in FORMATION.items())
        and value[picks].sum() <= BUDGET + 1e-9
        and pd.Series(club[picks]).value_counts().max() <= MAX_PER_TEAM
    )


def best_armband(picks, mu, sigma, cov_delta_G, lam, sense):
    """Captain and vice for a fixed XI that maximize the objective (all 110 pairs)."""
    best, best_w = -np.inf, None
    for c in picks:
        for v in picks:
            if c == v:
                continue
            w = np.zeros(len(mu))
            w[picks] = 1.0
            w[c], w[v] = CAPTAIN, VICE
            o = team_objective(w, mu, sigma, cov_delta_G, lam, sense)
            if o > best:
                best, best_w = o, w
    return best_w, best


def improve(picks, cands, mu, sigma, cov_delta_G, lam, sense):
    """Swap-improvement from an XI: replace one player at a time (the newcomer takes the outgoing player's
    multiplier) while the objective rises, re-choosing captain and vice after every pass. Returns (w, objective)."""
    w, cur = best_armband(picks, mu, sigma, cov_delta_G, lam, sense)
    n = len(mu)
    while True:
        improved = False
        for i in range(len(picks)):
            for j in np.setdiff1d(np.arange(n), picks):
                new = picks.copy()
                new[i] = j
                if not feasible(new, cands):
                    continue
                w_new = w.copy()
                w_new[j], w_new[picks[i]] = w[picks[i]], 0.0
                o = team_objective(w_new, mu, sigma, cov_delta_G, lam, sense)
                if o > cur + 1e-9:
                    picks, w, cur, improved = new, w_new, o, True
        w_arm, o = best_armband(picks, mu, sigma, cov_delta_G, lam, sense)
        if o > cur + 1e-9:
            w, cur, improved = w_arm, o, True
        if not improved:
            return w, cur


_GUROBI_ENV = None


def _miqp_gurobi(cands, mu, sigma, cov_delta_G, lam, s, sense, base_idx=None):
    """The MIQP: (team or None, status). The nonconvex sign (sense = -1) is Gurobi's nonconvex MIQP. base_idx (with
    s.track_k): at least XI_SIZE - track_k of these candidates picked."""
    import gurobipy as gp

    global _GUROBI_ENV
    if _GUROBI_ENV is None:
        _GUROBI_ENV = gp.Env(params={"OutputFlag": 0})
    n = len(cands)
    pos, club, value = (
        cands["pos"].to_numpy(),
        cands["team"].to_numpy(),
        cands["value"].to_numpy(dtype=float),
    )
    m = gp.Model(env=_GUROBI_ENV)
    m.Params.TimeLimit = s.time_limit
    pick, captain, vice = (m.addMVar(n, vtype=gp.GRB.BINARY) for _ in range(3))
    w = pick + (CAPTAIN - 1) * captain + (VICE - 1) * vice
    m.addConstr(captain + vice <= pick)
    m.addConstr(pick.sum() == XI_SIZE)
    m.addConstr(captain.sum() == 1)
    m.addConstr(vice.sum() == 1)
    m.addConstr(value @ pick <= BUDGET)
    for p, (low, high) in FORMATION.items():
        idx = np.where(pos == p)[0]
        m.addConstr(pick[idx].sum() >= low)
        m.addConstr(pick[idx].sum() <= high)
    for team in np.unique(club):
        m.addConstr(pick[np.where(club == team)[0]].sum() <= MAX_PER_TEAM)
    if base_idx is not None and s.track_k is not None:
        m.addConstr(pick[np.asarray(base_idx)].sum() >= XI_SIZE - s.track_k)
    eigvals, eigvecs = np.linalg.eigh(sigma)
    keep = eigvals > 1e-9 * eigvals.max()
    L = eigvecs[:, keep] * np.sqrt(eigvals[keep])
    y = m.addMVar(L.shape[1], lb=-gp.GRB.INFINITY)
    m.addConstr(y == L.T @ w)
    m.setObjective(
        mu @ w - sense * lam * (y @ y - 2 * (cov_delta_G @ w)), gp.GRB.MAXIMIZE
    )
    m.optimize()
    status = {gp.GRB.OPTIMAL: "optimal", gp.GRB.TIME_LIMIT: "timelimit"}.get(
        m.Status, str(m.Status)
    )
    if m.SolCount == 0:
        return None, status
    pk, cp, vc = (
        np.round(pick.X) > 0.5,
        np.round(captain.X) > 0.5,
        np.round(vice.X) > 0.5,
    )
    return cands.assign(
        w=pk + (CAPTAIN - 1) * cp + (VICE - 1) * vc, captain=cp, vice=vc
    )[pk], status


def solve_for_lambda(cands, mu, sigma, cov_delta_G, lam, s, sense=1, start_teams=(), base_idx=None):
    """The MIQP's team for lambda (Gurobi), polished by swap-improvement (improve) from its solution and start_teams.
    With base_idx and s.track_k, a polished team that breaks the tracking constraint is not used."""
    n = len(cands)
    team, status = _miqp_gurobi(cands, mu, sigma, cov_delta_G, lam, s, sense, base_idx)
    starts = [t.index.to_numpy() for t in start_teams if t is not None]
    if team is not None:
        starts = [team.index.to_numpy()] + starts
    elif not starts:
        return None, status
    if not s.polish:
        return team, status
    # swap-improvement from the solver's team and the given starts: a solve stopped at the time limit (large lambda,
    # and the risk-seeking sign, whose objective is not concave) can return a team a local search beats
    best_w, best_obj = None, -np.inf
    for picks in starts:
        w_, o_ = improve(np.asarray(picks), cands, mu, sigma, cov_delta_G, lam, sense)
        if o_ > best_obj:
            best_w, best_obj = w_, o_
    if team is not None:
        w0 = np.zeros(n)
        w0[team.index] = (team["w"] * 2).round() / 2
        if team_objective(w0, mu, sigma, cov_delta_G, lam, sense) >= best_obj - 1e-9:
            return team, status
    picked = best_w > 0
    if base_idx is not None and s.track_k is not None and picked[np.asarray(base_idx)].sum() < XI_SIZE - s.track_k:
        return team, status  # the polish left the tracking set: keep the solver's team (None if it had none)
    team = cands.assign(w=best_w, captain=best_w == CAPTAIN, vice=best_w == VICE)[
        picked
    ]
    return team, f"{status}+polished"


# ---------------------------------------------------------------------------- step 5: lambda*, one gameweek
SAA_SETTINGS = ("objective", "n_saa", "saa_time_limit")


def team_vector(team, n):
    w = np.zeros(n)
    w[team.index] = (team["w"] * 2).round() / 2
    return w


def solve_saa(cands, delta, G, s, starts=()):
    """The team that beats G in the most scenarios: max sum_s z_s subject to w'delta_s - G_s >= -M_s (1 - z_s), the
    FantasyGo rules on (pick, captain, vice) as in _miqp_gurobi. M_s = G_s minus a lower bound on any team's score in
    scenario s (its 11 lowest delta, the captain's and vice's extra on the two lowest). starts: teams to start from
    (the frontier's); the best of them in these scenarios is Gurobi's MIP start. Returns (team, status, share won)."""
    import gurobipy as gp

    global _GUROBI_ENV
    if _GUROBI_ENV is None:
        _GUROBI_ENV = gp.Env(params={"OutputFlag": 0})
    n, S = len(cands), len(G)
    pos, club, value = (
        cands["pos"].to_numpy(),
        cands["team"].to_numpy(),
        cands["value"].to_numpy(dtype=float),
    )
    low = np.sort(delta, axis=1)
    lb = (
        low[:, :XI_SIZE].sum(axis=1)
        + (CAPTAIN - 1) * np.minimum(low[:, 0], 0)
        + (VICE - 1) * np.minimum(low[:, 1], 0)
    )
    M = np.maximum(G - lb, 0) + 1.0
    m = gp.Model(env=_GUROBI_ENV)
    m.Params.TimeLimit = s.saa_time_limit
    pick, captain, vice = (m.addMVar(n, vtype=gp.GRB.BINARY) for _ in range(3))
    z = m.addMVar(S, vtype=gp.GRB.BINARY)
    w = pick + (CAPTAIN - 1) * captain + (VICE - 1) * vice
    m.addConstr(captain + vice <= pick)
    m.addConstr(pick.sum() == XI_SIZE)
    m.addConstr(captain.sum() == 1)
    m.addConstr(vice.sum() == 1)
    m.addConstr(value @ pick <= BUDGET)
    for p, (lo, hi) in FORMATION.items():
        idx = np.where(pos == p)[0]
        m.addConstr(pick[idx].sum() >= lo)
        m.addConstr(pick[idx].sum() <= hi)
    for team in np.unique(club):
        m.addConstr(pick[np.where(club == team)[0]].sum() <= MAX_PER_TEAM)
    m.addConstr(delta @ w - G >= -M * (1 - z))
    m.setObjective(z.sum(), gp.GRB.MAXIMIZE)
    best_start, best_won = None, -1
    for team in starts:
        wins = int((delta @ team_vector(team, n) - G > 0).sum())
        if wins > best_won:
            best_start, best_won = team, wins
    if best_start is not None:
        wv = team_vector(best_start, n)
        pick.Start, captain.Start, vice.Start = (
            (wv > 0).astype(float),
            (wv == CAPTAIN).astype(float),
            (wv == VICE).astype(float),
        )
        z.Start = (delta @ wv - G > 0).astype(float)
    m.optimize()
    status = {gp.GRB.OPTIMAL: "optimal", gp.GRB.TIME_LIMIT: "timelimit"}.get(
        m.Status, str(m.Status)
    )
    if m.SolCount == 0:
        return best_start, status, best_won / S
    pk, cp, vc = (
        np.round(pick.X) > 0.5,
        np.round(captain.X) > 0.5,
        np.round(vice.X) > 0.5,
    )
    team = cands.assign(
        w=pk + (CAPTAIN - 1) * cp + (VICE - 1) * vc, captain=cp, vice=vc
    )[pk]
    return team, status, float((delta @ team_vector(team, n) - G > 0).mean())


def run_gameweek(gw, inp, s, verbose=True, teams=None):
    """The frontier and lambda* for gameweek gw. teams: a given field as [(fpl ids, multipliers)] instead of s.field
    (generator_checks.py's oracle, the real teams of gw itself)."""
    rng = np.random.default_rng(s.seed + gw)
    players, mu, sigma = build_mu_sigma(gw, inp, s)
    W_o = (
        field_matrix(gw, players, inp, s, rng)
        if teams is None
        else teams_matrix(teams, players)
    )
    topheavy = s.objective == "topheavy"
    if topheavy:
        prizes = topheavy_payout(gw, inp)
        delta, T, cov_delta_G = monte_carlo_top(mu, sigma, W_o, s, rng, len(prizes))
        per = topheavy_correction(gw, inp, s, verbose)  # fix 2: one shift per paid place
        T = T + np.concatenate([per, np.full(max(T.shape[1] - 10, 0), per[-1])])[: T.shape[1]]
        G = T[:, min(9, T.shape[1] - 1)]  # the 10th-best field score: Prob(beat) for the report
    else:
        delta, G, cov_delta_G = monte_carlo(
            mu, sigma, W_o, s, rng, g_correction(gw, inp, s, verbose)
        )
    idx = shortlist(players, s.shortlist)
    cands = players.reset_index().iloc[idx].reset_index(drop=True)
    mu_c, sigma_c, cov_c = mu[idx], sigma[np.ix_(idx, idx)], cov_delta_G[idx]

    frontier, teams, sense = [], {}, (-1 if topheavy else 1)  # topheavy: out of the money (Assumption 5.1)
    for lam in s.lambda_grid:
        team, status = solve_for_lambda(
            cands,
            mu_c,
            sigma_c,
            cov_c,
            lam,
            s,
            sense,
            start_teams=list(teams.values())[-1:],
            base_idx=teams[0.0].index.to_numpy() if (s.track_k is not None and 0.0 in teams) else None,
        )
        if team is None:
            continue
        w = np.zeros(len(cands))
        w[team.index] = team["w"]
        if lam == 0 and not topheavy:
            sense = 1 if mu_c @ w - G.mean() >= 0 else -1  # minus lambda when mu_Y >= 0
        F = delta[:, idx] @ w
        Y = F - G
        frontier.append(
            {
                "lambda": lam,
                "status": status,
                "prob_beat": (Y > 0).mean(),
                "mean": mu_c @ w,
                "sd": np.sqrt(w @ sigma_c @ w),
                **({"exp_prize": expected_prize(F, T, prizes)} if topheavy else {}),
            }
        )
        teams[lam] = team
    frontier = pd.DataFrame(frontier)
    best = frontier.loc[frontier["exp_prize" if topheavy else "prob_beat"].idxmax(), "lambda"]
    if (
        s.objective == "saa"
    ):  # fitted on its own scenarios, Prob(beat) estimated on the n_mc draws above
        sa = replace(s, n_mc=s.n_saa)
        d_fit, G_fit, _ = monte_carlo(
            mu,
            sigma,
            W_o,
            sa,
            np.random.default_rng(s.seed + gw + 100_000),
            g_correction(gw, inp, s, False),
        )
        team, status, won = solve_saa(
            cands, d_fit[:, idx], G_fit, s, starts=list(teams.values())
        )
        w = team_vector(team, len(cands))
        frontier = pd.concat(
            [
                frontier,
                pd.DataFrame(
                    [
                        {
                            "lambda": "saa",
                            "status": status,
                            "prob_beat": (delta[:, idx] @ w - G > 0).mean(),
                            "mean": mu_c @ w,
                            "sd": np.sqrt(w @ sigma_c @ w),
                            "fit_share": won,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        teams["saa"], best = team, "saa"
        # every team on a third, independent set of draws: the frontier's lambda* is chosen on the n_mc draws, so its
        # Prob(beat) there is optimistic; prob_holdout compares the two honestly
        d_ho, G_ho, _ = monte_carlo(
            mu,
            sigma,
            W_o,
            s,
            np.random.default_rng(s.seed + gw + 200_000),
            g_correction(gw, inp, s, False),
        )
        frontier["prob_holdout"] = [
            (d_ho[:, idx] @ team_vector(teams[lam], len(cands)) - G_ho > 0).mean()
            for lam in frontier["lambda"]
        ]
    if verbose:
        print(
            f"GW{gw}: field of {len(W_o)}, G mean {G.mean():.1f} sd {G.std():.1f}, lambda* = {best}, "
            f"Prob(beat) {frontier.set_index('lambda').loc[best, 'prob_beat']:.1%} "
            f"(best on the frontier {frontier.loc[frontier['lambda'] != 'saa', 'prob_beat'].max():.1%})",
            flush=True,
        )
    return {
        "gw": gw,
        "frontier": frontier,
        "teams": teams,
        "lambda_star": best,
        "G_mean": G.mean(),
        "G_sd": G.std(),
    }


def run(s: Settings, gws=None, recompute=False, verbose=True):
    """Every gameweek's solution, cached in s.cache() (one entry per gameweek; delete the file to recompute)."""
    inp = Inputs(s)
    gws = gameweeks(inp, s, gws)
    path = s.cache()
    cache = (
        pickle.loads(path.read_bytes())
        if path.exists() and not recompute
        else {"settings": asdict(s), "results": {}}
    )
    cache["settings"].pop("solver", None)  # a setting of the SCIP version
    for k in SAA_SETTINGS:  # added later, with these defaults
        cache["settings"].setdefault(k, asdict(Settings())[k])
    cache["settings"].setdefault("sigma", "pairs")  # made before the factor model
    cache["settings"].setdefault("sigma_shrink", False)  # made before the optimizer's-curse options
    cache["settings"].setdefault("track_k", None)
    if cache["settings"] != asdict(s):
        raise ValueError(
            f"{path} was made with other settings; delete it or pass recompute=True"
        )
    for gw in gws:
        if gw not in cache["results"]:
            cache["results"][gw] = run_gameweek(gw, inp, s, verbose)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(
                pickle.dumps(cache)
            )  # after every gameweek, so a long run can resume
    return {gw: cache["results"][gw] for gw in gws}, inp


# ---------------------------------------------------------------------------- evaluation
def actual_points(gw, inp):
    """Each player's points in gameweek gw as the FantasyGo contest scored them, so that a team is ranked on the same
    scale as the real entries: the contest's own points for every player someone picked (their picks' points over
    the captain / vice multiplier), else FPL's total over all his fixtures (merged_gw.csv; the predictions file keeps
    only the first fixture of a double gameweek, and the contest counts both)."""
    cache = inp.__dict__.setdefault("_actual", {})
    if gw not in cache:
        if getattr(inp, "_fpl_points", None) is None:
            m = pd.read_csv(
                f"{config.paths(inp.season).vaastav_cache}/merged_gw.csv",
                usecols=["element", "GW", "total_points"],
            )
            inp._fpl_points = m.groupby(["GW", "element"])["total_points"].sum()
        fpl = (
            inp._fpl_points.xs(gw, level="GW")
            if gw in inp._fpl_points.index.get_level_values("GW")
            else pd.Series(dtype=float)
        )
        pk = inp.picks[inp.picks["gameweek"] == gw]
        mult = np.where(pk["is_captain"], CAPTAIN, np.where(pk["is_vice"], VICE, 1.0))
        contest = (pk["points"] / mult).groupby(pk["player_id"]).median()
        cache[gw] = pd.concat([contest, fpl[~fpl.index.isin(contest.index)]]).astype(
            float
        )
    return cache[gw]


def team_points(team, gw, inp):
    """The team's actual points (captain x2, vice x1.5), on the contest's scale (actual_points)."""
    w = (
        (team["w"] * 2).round() / 2
    )  # the solver's multipliers to 0, 1, 1.5, 2 (0.9999999 would lose a tied rank)
    return float((team["fpl_id"].map(actual_points(gw, inp)).fillna(0) * w).sum())


def contest_rank(points, gw, inp):
    """Rank in the gameweek's FantasyGo contest (1 + entries with more points), the number of entries and the prize."""
    fieldg = inp.managers[inp.managers["gameweek"] == gw].sort_values(
        "points", ascending=False
    )
    rank = int((fieldg["points"] > points).sum()) + 1
    prize = fieldg["prize"].iloc[rank - 1] if rank <= len(fieldg) else 0.0
    return rank, len(fieldg), prize


def evaluate(results, inp, s):
    """Per gameweek with a usable contest: the MVT team (lambda*) and the base team (lambda = 0), predicted and actual."""
    rows = []
    for gw, res in results.items():
        if gw not in inp.contest_gws:  # no real contest to rank the teams in
            continue
        row = {"round": gw, "lambda_star": res["lambda_star"], "G_mean": res["G_mean"]}
        for label, lam in [("mvt", res["lambda_star"]), ("base", 0.0)]:
            team = res["teams"][lam]
            f = res["frontier"].set_index("lambda").loc[lam]
            points = team_points(team, gw, inp)
            rank, n, prize = contest_rank(points, gw, inp)
            row |= {
                f"{label}_prob": f["prob_beat"],
                f"{label}_mean": f["mean"],
                f"{label}_sd": f["sd"],
                f"{label}_points": points,
                f"{label}_rank": rank,
                f"{label}_prize": prize,
            }
        row["entries"] = n
        row["target_rank"] = max(int(np.ceil((1 - s.r_pct) * n)), 1)
        rows.append(row)
    return pd.DataFrame(rows).set_index("round")


def summary(evaluation, s):
    """Season totals for the MVT and base teams."""
    out = {}
    for label in ["mvt", "base"]:
        e = evaluation
        out[label] = {
            "points": e[f"{label}_points"].sum(),
            "median_rank_pct": (e[f"{label}_rank"] / e["entries"]).median(),
            f"weeks_in_top_{round((1 - s.r_pct) * 100)}pct": int(
                (e[f"{label}_rank"] <= e["target_rank"]).sum()
            ),
            "predicted_prob_mean": e[f"{label}_prob"].mean(),
            "prizes": e[f"{label}_prize"].sum(),
        }
    return pd.DataFrame(out).T


# ---------------------------------------------------------------------------- is the simulated field like the real one?
def field_summary(teams, gw, inp, s, real_picks=None):
    """A field scored on the actual points: mean cost, the top-(1 - r_pct) cut, mean points, the share of each
    position's picks that went to its 5 most-picked players and, against real_picks, the rank correlation of the
    players' pick counts. Also returns the pick counts."""
    if not teams:  # no simulated team met the budget and club rules
        return {"cost": np.nan, "cut_points": np.nan, "mean_points": np.nan}, pd.Series(
            dtype=float
        )
    actual = (
        inp.preds[inp.preds["round"] == gw]
        .drop_duplicates("fpl_id")
        .set_index("fpl_id")
    )
    scored = actual_points(gw, inp)
    points = np.array(
        [(scored.reindex(ids).fillna(0).to_numpy() * m).sum() for ids, m in teams]
    )
    picks = pd.Series(np.concatenate([ids for ids, _ in teams])).value_counts()
    pos = picks.index.map(inp.positions)
    out = {
        "cost": np.mean([actual["value"].reindex(ids).sum() for ids, _ in teams]),
        "cut_points": np.quantile(points, s.r_pct),
        "mean_points": points.mean(),
        **{
            f"top5_{p}": picks[pos == p].nlargest(5).sum() / picks[pos == p].sum()
            for p in POS_ORDER
        },
    }
    if real_picks is not None:
        both = pd.concat([picks, real_picks], axis=1).fillna(0)
        out["share_corr"] = both.corr(method="spearman").iloc[0, 1]
    return out, picks


def field_check(
    s: Settings, n_fields=10, gws=None, max_tries=20_000, recompute=False, verbose=True
):
    """Per gameweek: the real contest, the previous gameweek's teams and n_fields simulated fields from each generator
    (one selection vector per team, "per_team_p", and one for the whole field, "shared_p"), all summarized by
    field_summary. Cached in mvt_dir/field_check_r{pct}.pkl after every gameweek (about 25 minutes from scratch)."""
    inp = Inputs(s)
    gws = gameweeks(inp, s, gws)
    path = (
        Path(config.paths(s.season).mvt_dir)
        / f"field_check_r{round(s.r_pct * 100)}.pkl"
    )
    key = {**asdict(s), "n_fields": n_fields, "max_tries": max_tries}
    cache = (
        pickle.loads(path.read_bytes())
        if path.exists() and not recompute
        else {"settings": key, "rows": {}}
    )
    if cache["settings"] != key:
        raise ValueError(
            f"{path} was made with other settings; delete it or pass recompute=True"
        )
    for gw in gws:
        if gw in cache["rows"]:
            continue
        players = (
            inp.preds[inp.preds["round"] == gw]
            .drop_duplicates("fpl_id")
            .set_index("fpl_id")
        )
        prev_gw = last_contest_before(gw, inp)
        real, real_picks = field_summary(entries(gw, inp), gw, inp, s)
        rows = [
            {"round": gw, "field": "real", **real},
            {
                "round": gw,
                "field": "previous",
                **field_summary(entries(prev_gw, inp), gw, inp, s, real_picks)[0],
            },
        ]
        n_teams = inp.picks.loc[inp.picks["gameweek"] == prev_gw, "row"].nunique()
        for name, shared in [("per_team_p", False), ("shared_p", True)]:
            rng = np.random.default_rng(s.seed + gw)
            sg = Settings(**{**asdict(s), "shared_p": shared})
            for k in range(n_fields):
                teams, accepted, budget = simulate_field(
                    gw, players, inp, sg, rng, max_tries=max_tries, return_budget=True
                )
                rows.append(
                    {
                        "round": gw,
                        "field": name,
                        "draw": k,
                        **field_summary(teams, gw, inp, s, real_picks)[0],
                        "teams": len(teams),
                        "full_field": n_teams,
                        "accept_rate": accepted,
                        **budget,
                    }
                )
        cache["rows"][gw] = rows
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(cache))
        if verbose:
            print(f"GW{gw} done", flush=True)
    return pd.DataFrame([r for gw in gws for r in cache["rows"][gw]])


def field_tables(fields, s: Settings):
    """(means per field, the simulated fields that came out short, how the simulated cuts cover the real cut)."""
    cols = [
        "cost",
        "cut_points",
        "mean_points",
        "share_corr",
        "accept_rate",
        "teams",
    ] + [f"top5_{p}" for p in POS_ORDER]
    means = (
        fields.groupby("field")[cols]
        .mean()
        .loc[["real", "previous", "per_team_p", "shared_p"]]
    )
    short = (
        fields[fields["teams"] < 0.9 * fields["full_field"]]
        .groupby(["field", "round"])[["teams", "full_field", "accept_rate"]]
        .mean()
    )
    sim = fields[fields["field"].isin(["per_team_p", "shared_p"])].dropna(
        subset=["cut_points"]
    )
    real_cut = fields[fields["field"] == "real"].set_index("round")["cut_points"]
    g = sim.groupby(["field", "round"])["cut_points"]
    cover = pd.DataFrame(
        {
            "sd_between_fields": g.std(),
            "error": g.mean() - g.mean().index.get_level_values("round").map(real_cut),
            "real_below_all": g.min()
            > g.min().index.get_level_values("round").map(real_cut),
            "real_above_all": g.max()
            < g.max().index.get_level_values("round").map(real_cut),
        }
    )
    coverage = cover.groupby("field").mean().rename(columns={"error": "mean_error"})
    return means, short, coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--season", default="23-24")
    parser.add_argument(
        "--history",
        nargs="*",
        default=["22-23"],
        help="structural predictions trained with these",
    )
    parser.add_argument(
        "--field", default="simulated", choices=["simulated", "previous"]
    )
    parser.add_argument("--r-pct", type=float, default=0.8)
    parser.add_argument(
        "--gws", help="e.g. 6-38 or 20 (default: every gameweek from 6)"
    )
    parser.add_argument(
        "--objective",
        default="mv",
        choices=["mv", "saa", "topheavy"],
        help="mv: lambda* of the mean-variance frontier; saa: maximize Prob(beat) on scenarios; topheavy: Haugh & "
        "Singal's Algorithm 4 (variance-seeking frontier, lambda* by expected prize money)",
    )
    parser.add_argument("--sigma", default="factor", choices=["factor", "pairs"])
    parser.add_argument("--sigma-shrink", action="store_true", help="empirical-Bayes shrinkage of sigma_p")
    parser.add_argument("--track-k", type=int, help="lambda > 0 teams differ from the blind team in <= k players")
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    s = Settings(
        season=args.season,
        history=tuple(args.history),
        field=args.field,
        r_pct=args.r_pct,
        objective=args.objective,
        sigma=args.sigma,
        sigma_shrink=args.sigma_shrink,
        track_k=args.track_k,
    )
    gws = None
    if args.gws:
        lo, _, hi = args.gws.partition("-")
        gws = range(int(lo), int(hi or lo) + 1)
    results, inp = run(s, gws, args.recompute)
    e = evaluate(results, inp, s)
    print(f"\nWrote {s.cache()}\n")
    print(summary(e, s).round(3).to_string())


if __name__ == "__main__":
    main()
