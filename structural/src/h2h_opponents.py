"""The 2024-25 head-to-head opponent model: who an opponent is likely to field next gameweek.

Ported from 24-25/optimizer/notebooks/optimization.ipynb (cells 5-11; the equations are the dissertation's Chapter 3),
with the notebook's globals held by a League object and its lookups cached (the keep/buy calibration loops over every
manager, round and position). The functions and their arithmetic are the notebook's.

  selection model   alpha_{j,m,t,k} = exp(intercept_m + x_std' (beta_league + delta_beta_m) + gamma_m y_prev): the league
                    betas (league_models_{pos}) and each manager's deviation and retention (manager_fits_{GW}), fitted
                    by 24-25/optimizer/beta estimators on the rolled player features
  keep/buy          eq. 3.6: last week's squad is kept with a logistic probability, the remaining slots go to everyone
                    else by a sharpened softmax, both fitted on the whole league's history before the gameweek
  W_o               a sparse C-vine (Gaussian / independence, BIC, one tree, |tau| > 0.35) over the latent uniforms of the
                    opponent's ownership history, keyed by alpha into full squads under his free transfers and squad
                    value (run_copula_pipeline)
  fixtures          one randomised round robin per half season (create_fixtures, the seeds of longitudinal.ipynb)

Positions are the long names of the fits ("Goalkeeper", ...); h2h.py maps them to the structural model's GK/DEF/MID/FWD.
"""

import ast
import random

import joblib
import numpy as np
import pandas as pd
import pyvinecopulib as pv
from scipy.optimize import minimize_scalar
from scipy.special import expit, logsumexp
from scipy.stats import kendalltau
from sklearn.linear_model import LogisticRegression

import config

POSITIONS = ["Goalkeeper", "Defender", "Midfielder", "Forward"]
SHORT = {"Goalkeeper": "GK", "Defender": "DEF", "Midfielder": "MID", "Forward": "FWD"}
LONG = {v: k for k, v in SHORT.items()}
FIT_NAME = {"Goalkeeper": "gk", "Defender": "def", "Midfielder": "mid", "Forward": "fwd"}
BINARY_COL = {"Goalkeeper": "goalkeeper_binary", "Defender": "defender_binary", "Midfielder": "midfielder_binary",
              "Forward": "forward_binary"}
SQUAD_SLOTS = {"Goalkeeper": 2, "Defender": 5, "Midfielder": 5, "Forward": 3}

# the sparse vine of the notebook: the only setting whose in-sample fit separates from a permutation null
BICOP_FAMILY_SET = [pv.BicopFamily.indep, pv.BicopFamily.gaussian]
SELECTION_CRITERION = "bic"
TRUNC_LVL = 1
TAU_THRESHOLD = 0.35


def create_fixtures(managers, seed=None):
    """Randomized single round-robin fixture (verbatim from longitudinal.ipynb)."""
    if seed is not None:
        random.seed(seed)
    managers = managers.copy()
    random.shuffle(managers)
    fixed, rotating = managers[0], managers[1:]
    fixtures = {}
    for gw in range(1, len(managers)):
        current = [fixed] + rotating
        gw_fixtures = {}
        for i in range(len(current) // 2):
            m1, m2 = current[i], current[-(i + 1)]
            gw_fixtures[m1], gw_fixtures[m2] = m2, m1
        fixtures[gw] = gw_fixtures
        rotating = [rotating[-1]] + rotating[:-1]
    return fixtures


def opponents_by_round(season="24-25", own=None):
    """A manager's opponent in every gameweek (ours by default): GW1-19 from the first draw, GW20-38 from the second."""
    cfg = config.SEASONS[season]
    managers, own = list(cfg["league_managers"]), own or cfg["own_team_id"]
    leg1, leg2 = (create_fixtures(managers, seed=s) for s in cfg["fixture_seeds"])
    return {gw: leg1[gw][own] for gw in range(1, 20)} | {gw + 19: leg2[gw][own] for gw in range(1, 20)}


class League:
    """The league's squads, the player features and the selection model's fits, with cached lookups."""

    def __init__(self, season="24-25"):
        p = config.paths(season)
        models = config.SEASONS[season]["league_models"]
        self.player_data = pd.read_csv(f"{p.league_dir}/player_features.csv", low_memory=False)
        self.selections = pd.read_csv(f"{p.league_dir}/league_selections.csv")
        self.fits = {pos: joblib.load(f"{models}/league betas/league_models_{FIT_NAME[pos]}") for pos in POSITIONS}
        self.manager_fits_path = f"{models}/opponent betas/estimates/manager_fits_{{GW}}.joblib"
        self.position = self.player_data.drop_duplicates("element").set_index("element")["position"].to_dict()
        self._rows = {(r.team_id, r.round): r for r in self.selections.itertuples(index=False)}
        self._parsed, self._pools, self._calib, self._overhaul = {}, {}, {}, {}
        self._fit_files = {}

    def manager_fits(self, gw, team_id, mode="personal"):
        """{position: fit} for team_id at gw: its own fit when manager_fits_{gw} is his (the files hold one manager each,
        the opponent our team met that week) and mode is "personal"; otherwise the league-average manager, the mean of
        delta_beta, gamma and intercept over every fit available at gw's deadline (files for gameweeks <= gw)."""
        own = self._fit_file(gw)
        if mode == "personal" and own and all(f.get("team_id") == team_id for f in own.values()):
            return own
        files = [self._fit_file(g) for g in range(6, gw + 1)]
        files = [f for f in files if f]
        out = {}
        for pos in POSITIONS:
            fits = [f[pos] for f in files]
            # the league fits' features change over the season: average each coefficient by feature name and align to
            # the features of the league fit used at gw (a feature a manager fit lacks counts as 0)
            feats = list(self.league_fit(pos, gw)["feats"])
            delta = np.array([[dict(zip(f["feats"], f["delta_beta_mean"])).get(k, 0.0) for k in feats] for f in fits])
            out[pos] = {"delta_beta_mean": delta.mean(axis=0),
                        "gamma_mean": float(np.mean([f["gamma_mean"] for f in fits])),
                        "intercept_mean": float(np.mean([f.get("intercept_mean", 0.0) for f in fits]))}
        return out

    def _fit_file(self, gw):
        if gw not in self._fit_files:
            try:
                self._fit_files[gw] = joblib.load(self.manager_fits_path.format(GW=gw))[gw]
            except FileNotFoundError:
                self._fit_files[gw] = None
        return self._fit_files[gw]

    # ------------------------------------------------------------------ squads and transfers
    def _parse(self, team_id, rnd, col):
        key = (team_id, rnd, col)
        if key not in self._parsed:
            row = self._rows.get((team_id, rnd))
            self._parsed[key] = None if row is None else ast.literal_eval(getattr(row, col))
        return self._parsed[key]

    def detect_squad_overhaul(self, team_id, gw, threshold=8):
        """(n_changed, is_overhaul) between rounds gw-2 and gw-1: more than `threshold` of the 15 slots changed is the
        signature of a Wildcard / Free Hit; (None, None) if either round is missing."""
        key = (team_id, gw, threshold)
        if key not in self._overhaul:
            prev, cur = self._parse(team_id, gw - 2, "squad"), self._parse(team_id, gw - 1, "squad")
            if prev is None or cur is None:
                self._overhaul[key] = (None, None)
            else:
                n = len(set(cur) - set(prev))
                self._overhaul[key] = (n, n > threshold)
        return self._overhaul[key]

    def free_transfers(self, team_id, target_round, cap=5, overhaul_threshold=8):
        """Banked free transfers entering target_round from FPL's accrual rule (1 a week, up to `cap`; a hit resets the
        bank to 1); Wildcard / Free Hit weeks neither use nor reset it."""
        ft = 1
        for w in range(2, target_round):
            n, overhaul = self.detect_squad_overhaul(team_id, w, overhaul_threshold)
            if n is None:
                continue
            if overhaul:
                ft = min(cap, ft + 1)
            elif n <= ft:
                ft = min(cap, ft - n + 1)
            else:
                ft = 1
        return ft

    def squad_players(self, team_id, rnd):
        """The 15 players of team_id's round-rnd squad (the Assistant Manager's 16th element dropped), or None."""
        squad = self._parse(team_id, rnd, "squad")
        return None if squad is None else {e for e in squad if e in self.position}

    def squad_in_position(self, team_id, rnd, position):
        return {e for e in self.squad_players(team_id, rnd) or () if self.position[e] == position}

    def get_owned(self, team_id, position, rnd, elements):
        """{element: 0/1} for `position` at round rnd, or None if that round's export doesn't line up with elements."""
        y = self._parse(team_id, rnd, BINARY_COL[position])
        if y is None or len(y) != len(elements):
            return None
        return dict(zip(elements.tolist(), np.asarray(y, dtype=float).tolist()))

    def squad_prices(self, gw):
        """{element: price} for squads entering round gw: the latest price at or before gw-1, or round gw's."""
        rows = self.player_data[self.player_data["round"] <= gw].sort_values("round")
        prices = rows.drop_duplicates("element", keep="last").set_index("element")["value"].to_dict()
        prices.update(rows[rows["round"] <= gw - 1].drop_duplicates("element", keep="last")
                      .set_index("element")["value"].to_dict())
        return prices

    # ------------------------------------------------------------------ selection model
    def league_fit(self, position, target_round):
        available = [gw for gw in self.fits[position] if gw <= target_round]
        if not available:
            raise ValueError(f"no league fit at or before round {target_round}")
        return self.fits[position][max(available)]

    def _position_pool(self, position, rnd, feats):
        """The position's pool at round rnd, sorted by element (the order of league_selections' binary columns)."""
        key = (position, rnd, tuple(feats))
        if key not in self._pools:
            pool = self.player_data[(self.player_data["round"] == rnd) & (self.player_data["position"] == position)]
            pool = pool.sort_values("element")
            self._pools[key] = (pool["element"].to_numpy(), pool[list(feats)].fillna(0).to_numpy(dtype=float))
        return self._pools[key]

    @staticmethod
    def alpha_manager(manager_fit, league_fit, X_new_raw, y_prev):
        X_std = (X_new_raw - league_fit["X_mean"]) / league_fit["X_std"]
        beta_m = league_fit["beta_mean"] + manager_fit["delta_beta_mean"]
        linear = manager_fit.get("intercept_mean", 0.0) + X_std @ beta_m + manager_fit["gamma_mean"] * y_prev
        return np.exp(np.clip(linear, -30, 30))

    @staticmethod
    def marginal_inclusion_probabilities(weights, k, rng, n_sim=3000):
        """P(item is one of the k drawn) under weighted sampling without replacement (Efraimidis-Spirakis keys)."""
        weights = np.clip(np.asarray(weights, dtype=float), 1e-12, None)
        keys = rng.uniform(size=(n_sim, len(weights))) ** (1.0 / weights)[None, :]
        topk = np.argpartition(-keys, kth=k - 1, axis=1)[:, :k]
        return np.bincount(topk.ravel(), minlength=len(weights)) / n_sim

    def position_week_design(self, team_id, position, league_fit, rnd):
        """Pool and ownership at round rnd-1, the retention lag from round rnd-2 (its own pool), standardized
        features."""
        elements, X_raw = self._position_pool(position, rnd - 1, league_fit["feats"])
        prev_elements, _ = self._position_pool(position, rnd - 2, league_fit["feats"])
        owned_prev = self.get_owned(team_id, position, rnd - 2, prev_elements) or {}
        y_prev = np.array([owned_prev.get(e, 0.0) for e in elements])
        owned_map = self.get_owned(team_id, position, rnd - 1, elements)
        X_std = (X_raw - league_fit["X_mean"]) / league_fit["X_std"]
        return elements, X_std, y_prev, owned_map

    def position_alpha_at_week(self, team_id, position, manager_fit, league_fit, rnd):
        elements, X_raw = self._position_pool(position, rnd - 1, league_fit["feats"])
        prev_elements, _ = self._position_pool(position, rnd - 2, league_fit["feats"])
        owned_prev = self.get_owned(team_id, position, rnd - 2, prev_elements) or {}
        y_prev = np.array([owned_prev.get(e, 0.0) for e in elements])
        alpha = self.alpha_manager(manager_fit, league_fit, X_raw, y_prev)
        return dict(zip(elements.tolist(), alpha.tolist())), self.get_owned(team_id, position, rnd - 1, elements)

    @staticmethod
    def keep_buy_marginals(eta, y_prev, k, calib):
        """eq. 3.6: P(kept) = logit^-1(a + b eta) for last week's squad, the expected buys spread over everyone else by
        softmax(tau eta)."""
        p = np.zeros(len(eta))
        owned = y_prev == 1
        p_keep = expit(calib["a"] + calib["b"] * eta[owned])
        p[owned] = p_keep
        if (~owned).any():
            n_buy = max(k - p_keep.sum(), 0.0)
            z = calib["tau"] * eta[~owned]
            p[~owned] = np.minimum(n_buy * np.exp(z - logsumexp(z)), 1.0)
        return p

    def fit_keep_buy_calibration(self, position, league_fit, max_round, min_round=5):
        """Maximum-likelihood (a, b, tau) of keep_buy_marginals, pooled over every league manager's non-overhaul weeks
        in [min_round, max_round); None before any such week."""
        key = (position, id(league_fit), max_round, min_round)
        if key in self._calib:
            return self._calib[key]
        keep_eta, keep_x, buy_groups = [], [], []
        for team_id in self.selections["team_id"].unique():
            for rnd in range(min_round, max_round):
                if self.detect_squad_overhaul(team_id, rnd)[1]:
                    continue
                elements, X_std, y_prev, owned_map = self.position_week_design(team_id, position, league_fit, rnd)
                if owned_map is None or y_prev.sum() == 0:
                    continue
                x = np.array([owned_map[e] for e in elements])
                eta = X_std @ league_fit["beta_mean"]
                owned = y_prev == 1
                keep_eta.append(eta[owned])
                keep_x.append(x[owned])
                if x[~owned].sum() > 0:
                    buy_groups.append((eta[~owned], x[~owned]))
        if not keep_eta:
            self._calib[key] = None
            return None
        keep_eta, keep_x = np.concatenate(keep_eta), np.concatenate(keep_x)
        if 0 < keep_x.mean() < 1:
            model = LogisticRegression(C=1.0).fit(keep_eta[:, None], keep_x)
            a, b = float(model.intercept_[0]), float(model.coef_[0, 0])
        else:
            rate = np.clip(keep_x.mean(), 1e-3, 1 - 1e-3)
            a, b = float(np.log(rate / (1 - rate))), 0.0

        def buy_nll(tau):
            return -sum((x * tau * eta).sum() - x.sum() * logsumexp(tau * eta) for eta, x in buy_groups)

        tau = minimize_scalar(buy_nll, bounds=(0.01, 50.0), method="bounded").x if buy_groups else 1.0
        self._calib[key] = {"a": a, "b": b, "tau": float(tau), "n_keep_obs": len(keep_x),
                            "n_buy_weeks": len(buy_groups)}
        return self._calib[key]

    def position_marginals_at_week(self, team_id, position, manager_fit, league_fit, rnd, rng, calib=None):
        if calib is not None:
            elements, X_std, y_prev, owned_map = self.position_week_design(team_id, position, league_fit, rnd)
            eta = X_std @ (league_fit["beta_mean"] + manager_fit["delta_beta_mean"])
            p = self.keep_buy_marginals(eta, y_prev, SQUAD_SLOTS[position], calib)
            return dict(zip(elements.tolist(), p.tolist())), owned_map
        alpha_map, owned_map = self.position_alpha_at_week(team_id, position, manager_fit, league_fit, rnd)
        p = self.marginal_inclusion_probabilities(np.array(list(alpha_map.values())), SQUAD_SLOTS[position], rng)
        return dict(zip(alpha_map.keys(), p.tolist())), owned_map

    def position_alpha_target(self, team_id, position, manager_fit, league_fit, gw):
        """alpha at the target week: round gw's (lagged) features and the gw-1 squad as the retention lag; squad members
        without a round-gw row are scored on their latest earlier features."""
        owned = self.squad_in_position(team_id, gw - 1, position)
        elements, X_raw = self._position_pool(position, gw, league_fit["feats"])
        missing = sorted(owned - set(elements.tolist()))
        if missing:
            latest = (self.player_data[(self.player_data["round"] < gw) & self.player_data["element"].isin(missing)]
                      .sort_values("round").drop_duplicates("element", keep="last"))
            elements = np.concatenate([elements, latest["element"].to_numpy()])
            X_raw = np.vstack([X_raw, latest[league_fit["feats"]].fillna(0).to_numpy(dtype=float)])
        y_prev = np.array([1.0 if e in owned else 0.0 for e in elements])
        return dict(zip(elements.tolist(), self.alpha_manager(manager_fit, league_fit, X_raw, y_prev).tolist())), owned

    # ------------------------------------------------------------------ W_o
    def opponent_squads(self, gw, team_id, n_top=30, n_samples=20_000, t_limit=None, min_history_weeks=1, seed=42,
                        keep_buy=True, fit_mode="personal", copula=True):
        """The notebook's run_copula_pipeline without its diagnostics: {squad (sorted tuple): count} of the feasible
        sampled squads and a summary, or None when the manager has too little history. copula=False samples the
        players independently from the same margins (the ablation of the C-vine)."""
        if t_limit is None:
            t_limit = self.free_transfers(team_id, gw)
        rng = np.random.default_rng(seed)
        manager_fits = self.manager_fits(gw, team_id, fit_mode)
        calib = {pos: self.fit_keep_buy_calibration(pos, self.league_fit(pos, gw), gw - 1) if keep_buy else None
                 for pos in POSITIONS}

        alpha_target, owned_target, records = {}, {}, []
        for pos in POSITIONS:
            amap, owned = self.position_alpha_target(team_id, pos, manager_fits[pos], self.league_fit(pos, gw), gw)
            alpha_target[pos], owned_target[pos] = amap, owned
            top = {e for e, _ in sorted(amap.items(), key=lambda kv: kv[1], reverse=True)[:n_top]}
            records += [{"element": e, "position": pos} for e in top | owned]
        pool = pd.DataFrame(records).sort_values(["position", "element"]).reset_index(drop=True)
        elements = pool["element"].to_numpy()
        position_arr = pool["position"].to_numpy()

        history = sorted(r for (t, r) in self._rows if t == team_id and r < gw - 1)
        overhaul = [r for r in history if self.detect_squad_overhaul(team_id, r)[1]]
        history = [r for r in history if r not in overhaul]
        if len(history) < min_history_weeks:
            return None

        X_hist = pd.DataFrame(index=history, columns=elements, dtype=float)
        P_hist = pd.DataFrame(index=history, columns=elements, dtype=float)
        for pos in POSITIONS:
            pos_elements = pool.loc[pool["position"] == pos, "element"].tolist()
            for rnd in history:
                p_map, owned_map = self.position_marginals_at_week(team_id, pos, manager_fits[pos],
                                                                   self.league_fit(pos, gw), rnd, rng, calib[pos])
                for e in pos_elements:
                    X_hist.loc[rnd, e] = (owned_map or {}).get(e, 0.0)
                    P_hist.loc[rnd, e] = p_map.get(e, 1e-6)
        a = np.where(X_hist.values == 0, 0.0, 1.0 - P_hist.values)
        b = np.where(X_hist.values == 0, 1.0 - P_hist.values, 1.0)
        U = a + rng.uniform(size=X_hist.shape) * (b - a)

        informative = np.where(X_hist.values.sum(axis=0) > 0)[0]
        vine, n_dep = None, 0
        if len(informative) >= 2 and copula:
            vine = fit_sparse_cvine(U[:, informative])
            n_dep = sum(f != pv.BicopFamily.indep for tree in vine.families for f in tree)

        alpha_vec = np.array([alpha_target[pos].get(e, 1e-12) for e, pos in zip(elements, position_arr)])
        current = np.array([float(e in owned_target[pos]) for e, pos in zip(elements, position_arr)])
        prices = self.squad_prices(gw)
        value = np.array([prices.get(e, np.nan) for e in elements])
        current_value = float(current @ np.nan_to_num(value))

        sim_U = rng.uniform(size=(n_samples, len(elements)))
        if vine is not None:
            sim_U[:, informative] = vine.simulate(n=n_samples, seeds=[seed])
        w_star = np.zeros_like(sim_U, dtype=int)
        for pos, k in SQUAD_SLOTS.items():
            idx = np.where(position_arr == pos)[0]
            keys = sim_U[:, idx] ** (1.0 / np.clip(alpha_vec[idx], 1e-12, None))[None, :]
            topk = np.argpartition(-keys, kth=k - 1, axis=1)[:, :k]
            w_star[np.repeat(np.arange(n_samples), k), idx[topk.ravel()]] = 1
        transfers = 0.5 * np.abs(w_star - current[None, :]).sum(axis=1)
        feasible = (transfers <= t_limit) & (w_star @ np.nan_to_num(value) <= current_value)
        counts = {}
        for row in w_star[feasible]:
            sq = tuple(sorted(elements[row.astype(bool)].tolist()))
            counts[sq] = counts.get(sq, 0) + 1
        return {"squad_counts": counts, "n_feasible": int(feasible.sum()), "n_unique": len(counts),
                "t_limit": t_limit, "n_history_weeks": len(history), "overhaul_rounds": overhaul,
                "n_informative": len(informative), "n_dep_pairs": n_dep}


def kendall_cvine_order(U):
    m = U.shape[1]
    tau = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            t, _ = kendalltau(U[:, i], U[:, j])
            tau[i, j] = tau[j, i] = 0.0 if np.isnan(t) else t
    return (np.argsort(-np.abs(tau).sum(axis=1)) + 1).tolist()


def fit_sparse_cvine(U):
    structure = pv.CVineStructure(order=kendall_cvine_order(U), trunc_lvl=TRUNC_LVL)
    controls = pv.FitControlsVinecop(family_set=BICOP_FAMILY_SET, selection_criterion=SELECTION_CRITERION,
                                     trunc_lvl=TRUNC_LVL, threshold=TAU_THRESHOLD)
    return pv.Vinecop.from_data(U, structure=structure, controls=controls)
