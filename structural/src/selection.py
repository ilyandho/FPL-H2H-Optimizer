"""FantasyGo selection model: how likely the league's managers are to pick each player (2023-24 only).

The opponent side of the MVT (mvt.py): the team generator draws the simulated field from it. Ported from
23-24/optimizer/MVT/beta estimators.ipynb, on data/{season}/players.csv (the same rows, order and features as that
notebook's input, with xP_rolling_3 rebuilt the same way), so it reproduces that notebook's fits. For player j of a
position in round t:

  alpha_jt = exp(b0 + x_jt' beta),   c_t ~ DirichletMultinomial(n_t, alpha_t)

  c_jt       the number of FantasyGo entries that picked player j in round t, n_t = sum_j c_jt
  x_jt       the player's standardized features, known before the deadline
  b0         sets how concentrated the picks are; alpha_jt / sum_k alpha_kt is the expected pick share

One Dirichlet-multinomial per round, beta ~ N(0, BETA_SD), b0 ~ N(0, INTERCEPT_SD), NUTS. The priors are the tight
ones (0.5, 1.0) chosen by selection_checks.py: equal or better leave-future-out ELPD than the original N(0, 1), N(0, 2)
and more plausible prior predictive pick shares. Features: rank the candidates by |correlation|
with the pick share and drop the ones redundant with a better one, fit, keep the features whose 94% HDI excludes 0,
refit. One fit per gameweek in FIT_ROUNDS on the rounds before it only. A power kappa per position and gameweek
(fitted on the earlier rounds' picks) sharpens the shares for the team generator.

Candidates: players with a fixture and minutes in the last 5 rounds or picked by a FantasyGo manager last round.

FantasyGo gameweeks in config.SEASONS[season]["fantasygo_excluded_gws"] (2023-24: 22-25, which hold other rounds'
contests) are treated as missing: left out of the fits, kappa and the evaluation, and "last round" is the latest
round with picks. Their alphas are still predicted.

Writes to paths(season).selection_dir:
  league_models.pkl          the fits, {pos: {round: fit}} (resumable: one entry is saved after every fit)
  league_select_probs.csv    every candidate's alpha and expected pick share, with the actual FantasyGo share
  league_alpha_draws.pkl     {round: {pos: fpl_id, alpha (N_DRAWS x candidates), power}}, read by mvt.py

Usage (from src/):
  python selection.py            # fit what is not cached yet (about an hour from scratch), write the outputs
  python selection.py --refit    # refit everything
"""

import argparse
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import betabinom, binomtest, spearmanr

import config

POSITIONS = ["GK", "DEF", "MID", "FWD"]
FIT_ROUNDS = list(range(6, 39))  # one fit per gameweek, on the rounds before it
FIRST_ROUND = 4  # rounds 1-3 have little history for the rolling features
N_DRAWS = 500  # posterior draws kept per fit
SEED = 42
BETA_SD, INTERCEPT_SD = 0.5, 1.0  # priors: beta ~ N(0, BETA_SD), b0 ~ N(0, INTERCEPT_SD) (see selection_checks.py)
PRIORS = {"beta_sd": BETA_SD, "intercept_sd": INTERCEPT_SD}

COMMON = ["value", "log_selected", "log_transfers_in", "log_transfers_out", "log_fg_share_prev", "win_prob", "elo_diff",
          "difficulty", "was_home", "xP_rolling_3", "total_points_rolling_3", "points_5_ewm", "minutes_rolling_3",
          "starts_rolling_3", "ict_index_rolling_3", "influence_rolling_3", "creativity_rolling_3", "threat_rolling_3"]
ATTACK = ["expected_goals_rolling_3", "expected_assists_rolling_3", "expected_goal_involvements_rolling_3",
          "xGChain_rolling_3", "xGBuildup_rolling_3"]
DEFENCE = ["expected_goals_conceded_rolling_3", "clean_sheets_rolling_3"]
CANDIDATES = {"GK": COMMON + DEFENCE + ["saves_rolling_3"], "DEF": COMMON + ATTACK + DEFENCE,
              "MID": COMMON + ATTACK, "FWD": COMMON + ATTACK}


# ---------------------------------------------------------------------------- data
def lagged_mean(history, rows, stat, k):
    """Mean of `stat` over each player's last k gameweeks before each row's round, 0 for a gameweek without a row in
    `history` and 0 until k earlier gameweeks exist; gameweeks where `stat` is missing for everyone are skipped."""
    rounds = sorted(rows["round"].unique())
    kept = [r for r in rounds if history.loc[history["round"] == r, stat].notna().any()]
    grid = (history[history["round"].isin(kept)].pivot_table(index="round", columns="fpl_id", values=stat, aggfunc="sum")
            .reindex(kept).fillna(0.0))
    roll = grid.rolling(k, min_periods=k).mean().reindex(rounds).ffill().shift(1).fillna(0.0)
    out = rows[["fpl_id", "round"]].merge(roll.stack().rename("x").reset_index(), on=["round", "fpl_id"], how="left")
    return out["x"].fillna(0.0).to_numpy()


def load_data(season="23-24"):
    """Every candidate player-round with its FantasyGo picks and features, and the FantasyGo picks."""
    with warnings.catch_warnings():  # pandas' "highly fragmented" PerformanceWarning on the wide players.csv
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        return _load_data(season)


def _load_data(season):
    p = config.paths(season)
    if not os.path.exists(p.fantasygo):
        raise FileNotFoundError(f"no FantasyGo data for {season} ({p.fantasygo}); the selection model is 2023-24 only")
    data = pd.read_csv(p.players, low_memory=False)
    data = data[data["fixture_no"] == 1].drop(columns=["fixture_no", "n_fixtures"])  # one row per gameweek, as fitted
    # FPL's xP over the last 3 gameweeks the player played (the earlier pipeline's xP_rolling_3)
    data["xP_rolling_3"] = lagged_mean(data[data["minutes"] > 0], data, "xP", 3)
    picks = pd.read_csv(f"{p.fantasygo}/fg_manager_picks.csv")
    picks = picks[~picks["gameweek"].isin(p.fantasygo_excluded_gws)]  # another round's contest (config.SEASONS)

    counts = picks.groupby(["gameweek", "player_id"]).size().rename("fg_picks")
    data = data.merge(counts, left_on=["round", "fpl_id"], right_index=True, how="left")
    data["has_picks"] = data["round"].isin(picks["gameweek"].unique())  # False: no usable contest that round
    data["fg_picks"] = data["fg_picks"].fillna(0).where(data["has_picks"])
    data["fg_share"] = data["fg_picks"] / data.groupby(["round", "position"])["fg_picks"].transform("sum")

    # the share of the position's picks in the latest round with picks before this one (the previous round, except
    # after an excluded one), from the picks themselves: a player without a row that round still has his picks counted,
    # and one with no picks has 0, not an older round's share
    prev = picks.groupby(["gameweek", "position", "player_id"]).size()
    prev = (prev / prev.groupby(level=["gameweek", "position"]).transform("sum")).rename("fg_share_prev").reset_index()
    with_picks = np.sort(picks["gameweek"].unique())
    source = pd.DataFrame({"round": np.sort(data["round"].unique())})
    source["gameweek"] = [with_picks[with_picks < r].max() if (with_picks < r).any() else -1 for r in source["round"]]
    prev = source.merge(prev, on="gameweek")
    data = data.merge(prev[["round", "player_id", "fg_share_prev"]], left_on=["round", "fpl_id"],
                      right_on=["round", "player_id"], how="left").drop(columns="player_id")
    data["fg_share_prev"] = data["fg_share_prev"].fillna(0)
    data = data.assign(
        log_fg_share_prev=np.log(data["fg_share_prev"] + 1e-4),
        log_selected=np.log1p(data["selected_by_percent"]),
        log_transfers_in=np.log1p(data["transfers_in"]),
        log_transfers_out=np.log1p(data["transfers_out"]),
        elo_diff=data["elo"] - data["opp_elo"],
        difficulty=np.where(data["was_home"], data["team_h_difficulty"], data["team_a_difficulty"]),
    )
    data = data[(data["round"] >= FIRST_ROUND) & ((data["minutes_rolling_5"] > 0) | (data["fg_share_prev"] > 0))]
    return data.reset_index(drop=True), picks


# ---------------------------------------------------------------------------- the model
def select_features_by_correlation(df, candidate_feats, target_col, redundancy_threshold=0.8, min_corr=0.0,
                                   verbose=True):
    """Rank the candidates by |correlation| with target_col and keep each one whose |correlation| with every feature
    already kept is below redundancy_threshold (and with the target above min_corr). In candidate order."""
    df_valid = df[candidate_feats + [target_col]].dropna()
    target_corr = df_valid[candidate_feats].corrwith(df_valid[target_col]).abs()
    ranked = target_corr[~target_corr.index.duplicated(keep="first")].sort_values(ascending=False)
    feat_corr = df_valid[candidate_feats].corr().abs()
    selected = []
    for feat in ranked.index:
        if pd.isna(ranked[feat]) or ranked[feat] < min_corr:
            continue
        if not selected or feat_corr.loc[feat, selected].max() < redundancy_threshold:
            selected.append(feat)
        elif verbose:
            worst = feat_corr.loc[feat, selected].idxmax()
            print(f"  dropping {feat!r} (|corr| with {target_col}={target_corr[feat]:.3f}) -- "
                  f"|corr|={feat_corr.loc[feat, selected].max():.3f} with already-selected {worst!r}")
    return [f for f in candidate_feats if f in selected]


def _hdi_overlap(lo1, hi1, lo2, hi2):
    """Share of the narrower interval covered by the overlap of the two."""
    overlap = max(0.0, min(hi1, hi2) - max(lo1, lo2))
    narrower = min(hi1 - lo1, hi2 - lo2)
    return overlap / narrower if narrower > 0 else 0.0


def prune_by_hdi(summary, feats, overlap_threshold=0.8, verbose=True):
    """The features whose 94% HDI excludes 0, without near-duplicate HDIs (strongest effect kept)."""
    sig = summary[(summary["hdi_3%"] > 0) | (summary["hdi_97%"] < 0)]
    if len(sig) == 0:
        return [], []
    labels = sig.index.tolist()
    indices = [int(lbl.split("[")[1].split("]")[0]) for lbl in labels]
    kept_local = []
    for i in np.argsort(-sig["mean"].abs().to_numpy()):
        lo_i, hi_i = sig["hdi_3%"].iloc[i], sig["hdi_97%"].iloc[i]
        twin = next((k for k in kept_local
                     if _hdi_overlap(lo_i, hi_i, sig["hdi_3%"].iloc[k], sig["hdi_97%"].iloc[k]) > overlap_threshold), None)
        if twin is None:
            kept_local.append(i)
        elif verbose:
            print(f"  dropping {feats[indices[i]]!r} -- HDI near-duplicate of {feats[indices[twin]]!r}")
    kept_indices = [indices[i] for i in sorted(kept_local)]
    return [feats[i] for i in kept_indices], kept_indices


def fit_dirmult(X, counts, round_idx, beta_sd=None, intercept_sd=None, draws=1000, tune=1000, chains=4, random_seed=SEED):
    """NUTS fit of counts_t ~ DirichletMultinomial(n_t, exp(b0 + X beta)), one per round."""
    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt

    n_rounds = round_idx.max() + 1
    n_t = np.bincount(round_idx, weights=counts, minlength=n_rounds)
    with pm.Model():
        intercept = pm.Normal("intercept", 0.0, INTERCEPT_SD if intercept_sd is None else intercept_sd)
        beta = pm.Normal("beta", 0.0, BETA_SD if beta_sd is None else beta_sd, shape=X.shape[1])
        alpha = pt.exp(pt.clip(intercept + pt.dot(X, beta), -20, 20))
        A = pt.inc_subtensor(pt.zeros(n_rounds)[round_idx], alpha)
        pm.Potential("dirmult", pt.sum(pt.gammaln(A) - pt.gammaln(n_t + A))
                     + pt.sum(pt.gammaln(counts + alpha) - pt.gammaln(alpha)))
        trace = pm.sample(draws, tune=tune, chains=chains, cores=chains, target_accept=0.9, random_seed=random_seed,
                          progressbar=False)
    summary = az.summary(trace, var_names=["beta"])
    n_div = int(trace.sample_stats["diverging"].sum())
    if n_div or summary["r_hat"].max() > 1.01:
        print(f"  WARNING: {n_div} divergences, max r_hat {summary['r_hat'].max():.3f} -- provisional fit")
    return trace, summary


def standardize(df, feats):
    X = df[feats].to_numpy(dtype=float)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def thin(draws):
    """N_DRAWS evenly spaced rows of the stacked posterior draws."""
    return draws[np.linspace(0, len(draws) - 1, N_DRAWS).astype(int)]


def fit_position(data, pos, fit_round, redundancy_threshold=0.7, min_corr=0.05, verbose=True):
    """Select features, fit, prune by HDI and refit, on the position's rounds with picks before fit_round."""
    train = data[(data["position"] == pos) & (data["round"] < fit_round) & data["has_picks"]]
    if verbose:
        print(f"{pos}, rounds {train['round'].min()}-{train['round'].max()}, {len(train)} rows", flush=True)
    feats = select_features_by_correlation(train, CANDIDATES[pos], "fg_share", redundancy_threshold, min_corr, verbose)
    counts = train["fg_picks"].to_numpy(dtype=float)
    round_idx = pd.factorize(train["round"])[0]
    X, _, _ = standardize(train, feats)
    _, summary = fit_dirmult(X, counts, round_idx)
    feats_kept, _ = prune_by_hdi(summary, feats, verbose=verbose)
    if not feats_kept:
        print("  WARNING: no features survived pruning -- keeping the full set")
        feats_kept = feats
    X, mean, std = standardize(train, feats_kept)
    trace, summary = fit_dirmult(X, counts, round_idx)
    summary.index = feats_kept
    return {
        "feats": feats_kept, "X_mean": mean, "X_std": std, "summary_final": summary,
        "intercept": float(trace.posterior["intercept"].mean()),
        "beta_mean": trace.posterior["beta"].mean(dim=("chain", "draw")).values,
        "n_divergences": int(trace.sample_stats["diverging"].sum()),
        "priors": PRIORS,
        "intercept_draws": thin(trace.posterior["intercept"].values.reshape(-1)),
        "beta_draws": thin(trace.posterior["beta"].values.reshape(-1, len(feats_kept))),
    }


def fit_all(data, season="23-24", refit=False, verbose=True):
    """Every position's fit for every gameweek in FIT_ROUNDS, cached in league_models.pkl after each fit."""
    path = f"{config.paths(season).selection_dir}/league_models.pkl"
    models = joblib.load(path) if os.path.exists(path) and not refit else {}
    for pos in POSITIONS:
        for r in FIT_ROUNDS:
            cached = models.setdefault(pos, {}).get(r)
            if cached is None or cached.get("priors") != PRIORS:  # fits under other priors are refitted
                models[pos][r] = fit_position(data, pos, r, verbose=verbose)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                joblib.dump(models, path)
    return models


def fit_for_round(fits, target_round):
    """The latest fit made at or before target_round."""
    available = [r for r in fits if r <= target_round]
    if not available:
        raise ValueError(f"No fit at or before round {target_round}")
    return fits[max(available)]


def alpha_draws(fit, X_new_raw):
    """alpha for new rows under each kept posterior draw: shape (N_DRAWS, rows)."""
    X = (X_new_raw - fit["X_mean"]) / fit["X_std"]
    return np.exp(np.clip(fit["intercept_draws"][:, None] + fit["beta_draws"] @ X.T, -30, 30))


def alpha_mean(fit, X_new_raw):
    """alpha for new rows at the posterior mean of the coefficients."""
    X = (X_new_raw - fit["X_mean"]) / fit["X_std"]
    return np.exp(np.clip(fit["intercept"] + X @ fit["beta_mean"], -30, 30))


def rounds(data, models, start=FIT_ROUNDS[0], with_picks=True):
    """(pos, round, candidates, that round's fit) for every position and fitted gameweek from `start`; only the rounds
    with picks (the ones that can be scored) unless with_picks is False."""
    data = data[(data["round"] >= start) & (data["has_picks"] | (not with_picks))]
    for (pos, rnd), g in data.groupby(["position", "round"]):
        yield pos, rnd, g, fit_for_round(models[pos], rnd)


# ---------------------------------------------------------------------------- outputs
def estimates(data, models):
    """Every candidate's alpha (posterior mean) and expected pick share from that gameweek's fit (fg_picks and fg_share
    are empty in a round without usable picks)."""
    out = []
    for pos, rnd, g, fit in rounds(data[data["round"].isin(FIT_ROUNDS)], models, with_picks=False):
        alpha = alpha_mean(fit, g[fit["feats"]].to_numpy(dtype=float))
        out.append(g[["round", "position", "fpl_id", "name", "team", "value", "fg_picks", "fg_share"]]
                   .assign(alpha=alpha, pred_share=alpha / alpha.sum()))
    return pd.concat(out, ignore_index=True).sort_values(["round", "position", "pred_share"],
                                                         ascending=[True, True, False])


def pick_log_loss(power, by_round):
    """Mean negative log-probability of the actual picks under the shares alpha^power / sum."""
    total, n = 0.0, 0.0
    for alpha, picks in by_round:
        log_p = power * np.log(alpha)
        log_p -= np.logaddexp.reduce(log_p)
        total -= (picks * log_p).sum()
        n += picks.sum()
    return total / n


def sharpening_powers(est):
    """kappa per (position, gameweek): the power minimizing pick_log_loss over the earlier rounds with picks (1 before
    any)."""
    by_round = {(pos, rnd): (g["alpha"].to_numpy(), g["fg_picks"].to_numpy())
                for (pos, rnd), g in est.dropna(subset=["fg_picks"]).groupby(["position", "round"])}
    powers = {}
    for pos in POSITIONS:
        for rnd in FIT_ROUNDS:
            earlier = [by_round[pos, r] for r in FIT_ROUNDS if r < rnd and (pos, r) in by_round]
            powers[pos, rnd] = (minimize_scalar(pick_log_loss, bounds=(0.1, 10.0), args=(earlier,),
                                                method="bounded").x if earlier else 1.0)
    return powers


def field_alpha_draws(data, models, powers):
    """{round: {pos: fpl_id, alpha (N_DRAWS x candidates), power}} for the team generator."""
    draws = {}
    for pos, rnd, g, fit in rounds(data[data["round"].isin(FIT_ROUNDS)], models, with_picks=False):
        draws.setdefault(rnd, {})[pos] = {
            "fpl_id": g["fpl_id"].to_numpy(),
            "alpha": alpha_draws(fit, g[fit["feats"]].to_numpy(dtype=float)).astype(np.float32),
            "power": powers[pos, rnd],
        }
    return dict(sorted(draws.items()))


def write_outputs(data, models, season="23-24"):
    """league_select_probs.csv and league_alpha_draws.pkl; returns (estimates, powers)."""
    out = config.paths(season).selection_dir
    est = estimates(data, models)
    est.to_csv(f"{out}/league_select_probs.csv", index=False)
    powers = sharpening_powers(est)
    joblib.dump(field_alpha_draws(data, models, powers), f"{out}/league_alpha_draws.pkl")
    return est, powers


# ---------------------------------------------------------------------------- evaluation
def convergence(models):
    """Per position: the worst r_hat and bulk ESS over its fits, and their divergences (NaN when not recorded)."""
    rows = [{"position": pos, "round": r, "max_r_hat": f["summary_final"]["r_hat"].max(),
             "min_ess_bulk": f["summary_final"]["ess_bulk"].min(), "divergences": f.get("n_divergences", np.nan)}
            for pos, fits in models.items() for r, f in fits.items()]
    return (pd.DataFrame(rows).groupby("position")
            .agg(fits=("round", "size"), max_r_hat=("max_r_hat", "max"), min_ess_bulk=("min_ess_bulk", "min"),
                 divergences=("divergences", lambda d: d.sum(min_count=1))).loc[POSITIONS])


def _scores(pred_share, g):
    p = np.clip(pred_share, 1e-6, None)
    p = p / p.sum()
    top = set(g.nlargest(10, "fg_picks").index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the uniform baseline has no rank correlation
        rho = spearmanr(p, g["fg_share"]).statistic
    return {"log_loss": -(g["fg_picks"] * np.log(p)).sum() / g["fg_picks"].sum(), "spearman": rho,
            "top10": len(top & set(pd.Series(p, index=g.index).nlargest(10).index))}


def out_of_sample(data, models):
    """Per position and gameweek, the expected pick shares of the model and three baselines against the actual picks:
    log loss of the picks, rank correlation with the actual shares, and hits among the 10 most-picked players."""
    rows = []
    for pos, rnd, g, fit in rounds(data, models):
        alpha = alpha_mean(fit, g[fit["feats"]].to_numpy(dtype=float))
        for name, share in {"model": alpha / alpha.sum(), "uniform": np.ones(len(g)),
                            "fpl_ownership": g["selected_by_percent"].to_numpy(dtype=float),
                            "last_round_fg": g["fg_share_prev"].to_numpy(dtype=float)}.items():
            rows.append({"position": pos, "round": rnd, "predictor": name, **_scores(share, g)})
    return pd.DataFrame(rows)


def paired_tests(check):
    """Per position and baseline: rounds where the model's log loss is lower, the mean gain (baseline minus model),
    and a two-sided sign test."""
    ll = check.pivot_table(index=["position", "round"], columns="predictor", values="log_loss")
    rows = []
    for pos in POSITIONS:
        g = ll.loc[pos]
        for base in ["fpl_ownership", "last_round_fg", "uniform"]:
            gain = g[base] - g["model"]
            better = int((gain > 0).sum())
            rows.append({"position": pos, "baseline": base, "rounds": len(gain), "model_better": better,
                         "mean_gain": gain.mean(), "sign_test_p": binomtest(better, len(gain)).pvalue})
    return pd.DataFrame(rows).set_index(["position", "baseline"])


def predictive_pit(data, models, n_draws=50, seed=SEED):
    """Randomized PIT of every candidate's pick count under its posterior predictive distribution: per posterior draw
    BetaBinomial(n_t, alpha_j, A_t - alpha_j), mixed over n_draws of the draws. Uniform when calibrated."""
    rng = np.random.default_rng(seed)
    step = max(N_DRAWS // n_draws, 1)
    out = []
    for pos, rnd, g, fit in rounds(data[data["round"].isin(FIT_ROUNDS)], models):
        c = g["fg_picks"].to_numpy()
        n = c.sum()
        if n == 0:
            continue
        alpha = alpha_draws(fit, g[fit["feats"]].to_numpy(dtype=float))[::step]
        rest = alpha.sum(axis=1, keepdims=True) - alpha
        lo = betabinom.cdf(c - 1, n, alpha, rest).mean(axis=0)
        hi = betabinom.cdf(c, n, alpha, rest).mean(axis=0)
        out.append(pd.DataFrame({"position": pos, "round": rnd,
                                 "expected_share": (alpha / alpha.sum(axis=1, keepdims=True)).mean(axis=0),
                                 "pit": lo + rng.uniform(size=len(c)) * (hi - lo)}))
    pit = pd.concat(out, ignore_index=True)
    pit["players"] = np.where(pit["expected_share"] >= 0.01, "expected share >= 1%", "expected share < 1%")
    return pit


def round_stats(c):
    """How concentrated one round's picks are."""
    s = np.sort(c)[::-1] / c.sum()
    return {"top5_share": s[:5].sum(), "max_share": s[0], "hhi": (s ** 2).sum(), "players_picked": int((c > 0).sum())}


def _dirichlet(alpha, rng):
    g = rng.gamma(alpha)
    return g / g.sum() if g.sum() > 0 else alpha / alpha.sum()


def posterior_predictive(data, models, reps=200, seed=SEED):
    """Per position, gameweek and statistic of round_stats: the actual value, the median of `reps` replicated rounds
    (a posterior draw, one p_t ~ Dirichlet(alpha_t) for the round, n_t picks from it, as in the likelihood) and the
    share of replicates at least as large (p_value; uniform over the rounds when calibrated)."""
    rng = np.random.default_rng(seed)
    rows = []
    for pos, rnd, g, fit in rounds(data[data["round"].isin(FIT_ROUNDS)], models):
        c = g["fg_picks"].to_numpy().astype(int)
        n = c.sum()
        if n == 0:
            continue
        alpha = alpha_draws(fit, g[fit["feats"]].to_numpy(dtype=float))
        rep = pd.DataFrame([round_stats(rng.multinomial(n, _dirichlet(alpha[k], rng)))
                            for k in rng.integers(len(alpha), size=reps)])
        for stat, observed in round_stats(c).items():
            rows.append({"position": pos, "round": rnd, "stat": stat, "observed": observed,
                         "replicated": rep[stat].median(), "p_value": (rep[stat] >= observed).mean()})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--season", default="23-24")
    parser.add_argument("--refit", action="store_true", help="refit every position and gameweek")
    args = parser.parse_args()
    data, _ = load_data(args.season)
    models = fit_all(data, args.season, args.refit)
    write_outputs(data, models, args.season)
    print(f"\nWrote {config.paths(args.season).selection_dir}/")
    print(convergence(models).to_string())


if __name__ == "__main__":
    main()
