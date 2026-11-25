"""The opponent as a mixture over his plausible squads, XIs and captains: the H2H benchmark for the MVT.

In head-to-head the opponent fields one team, so his plausible squads W_o (h2h_opponents.League.opponent_squads: the
league and manager selection fits, keep/buy, sparse C-vine) are not a field to rank but guesses at that one team. The
benchmark G = w_o' delta has w_o drawn from the mixture:

  squad    p(squad) = its share of the opponent model's feasible samples (the sampler's counts); his previous
           gameweek's squad when the model has no fit (the thesis's fallback)
  XI       every valid XI of the squad (550: 1 GK, >= 3 DEF, >= 2 MID, >= 1 FWD), p(XI | squad) proportional to
           exp(XI expected points / tau_xi)
  captain  every starter, p(captain | XI) proportional to exp(mu_captain / tau_cap); captain x2, no chips

tau_xi and tau_cap are fitted by maximum likelihood on the league managers' real XIs and captains (their FFS picks)
in the gameweeks before the decision, with the structural model's mu of each of those gameweeks (Bench Boost weeks
left out). Per squad the most likely (XI, captain) rows are kept (`top_k`, renormalized; the mass they held before
renormalizing is reported).

With the mixture: mu_G = sum_k p_k mu'w_k, sigma_G^2 = sum_k p_k w_k' Sigma w_k + Var_k(mu'w_k), cov(delta, G) =
Sigma wbar (wbar = sum_k p_k w_k, his expected line-up), and exactly Prob(beat) = sum_k p_k Phi(mu'(v - w_k) / sd_k),
sd_k^2 = (v - w_k)' Sigma (v - w_k).
"""

from itertools import combinations

import h2h
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import norm

FORMATIONS = [
    (d, m, 10 - d - m) for d in range(3, 6) for m in range(2, 6) if 1 <= 10 - d - m <= 3
]
SLOTS = {
    "GK": range(2),
    "DEF": range(2, 7),
    "MID": range(7, 12),
    "FWD": range(12, 15),
}  # a squad by position


def _enumerate():
    xis = []
    for gk in SLOTS["GK"]:
        for d, m, f in FORMATIONS:
            for ds in combinations(SLOTS["DEF"], d):
                for ms in combinations(SLOTS["MID"], m):
                    for fs in combinations(SLOTS["FWD"], f):
                        xis.append((gk, *ds, *ms, *fs))
    xi_mask = np.zeros((len(xis), 15))
    for k, xi in enumerate(xis):
        xi_mask[k, list(xi)] = 1.0
    members = np.array(xis)  # (550, 11) squad slots of each XI's starters
    rows = np.repeat(
        xi_mask, 11, axis=0
    )  # XI-major: row = xi * 11 + j, captain = members[xi, j]
    rows[np.arange(len(rows)), members.reshape(-1)] += 1.0
    return xi_mask, members, rows


XI_MASK, MEMBERS, ROWS = _enumerate()  # (550, 15), (550, 11), (6050, 15)
assert len(XI_MASK) == 550


def order_squad(squad, pos):
    """The squad's 15 players by position (GK, DEF, MID, FWD), or None if it is not 2-5-5-3."""
    by = {p: sorted(e for e in squad if pos.get(e) == p) for p in SLOTS}
    if [len(by[p]) for p in SLOTS] != [2, 5, 5, 3]:
        return None
    return [e for p in SLOTS for e in by[p]]


def row_logp(mu_sq, tau_xi, tau_cap):
    """log p(XI, captain | squad) for every one of the 6050 rows; mu_sq (S, 15) -> (S, 6050)."""
    pts = mu_sq @ XI_MASK.T / tau_xi
    lp_xi = pts - logsumexp(pts, axis=1, keepdims=True)
    cap = mu_sq[:, MEMBERS] / tau_cap  # (S, 550, 11)
    lp_cap = cap - logsumexp(cap, axis=2, keepdims=True)
    return (lp_xi[:, :, None] + lp_cap).reshape(len(mu_sq), -1)


# ---------------------------------------------------------------------------- the temperatures
class LineupModel:
    """tau_xi, tau_cap fitted on the league managers' real line-ups of the gameweeks before a decision."""

    def __init__(self, inp):
        self.inp, self.obs, self.fits = inp, None, {}

    def _observations(self):
        """Per manager-week: the XI points of all 550 XIs, the real XI's index, the starters' mu, the captain's slot."""
        inp, out = self.inp, []
        index = {tuple(np.flatnonzero(m)): k for k, m in enumerate(XI_MASK)}
        for (team, gw), pk in inp.picks.groupby(["team_id", "gw"]):
            week = inp.week(gw)
            if (
                len(pk) != 15
                or week.empty
                or (pk.loc[pk["slot"] > 11, "multiplier"] > 0).all()
            ):
                continue
            sq = order_squad(pk["element"].tolist(), inp.position)
            if sq is None:
                continue
            slot = {e: i for i, e in enumerate(sq)}
            xi = tuple(sorted(slot[e] for e in pk.loc[pk["slot"] <= 11, "element"]))
            cap = pk.loc[pk["is_captain"], "element"]
            if (
                xi not in index or len(cap) != 1 or slot.get(int(cap.iloc[0])) not in xi
            ):  # captain on the bench
                continue
            mu = week["pred"].reindex(sq).fillna(0.0).to_numpy()
            k = index[xi]
            out.append(
                {
                    "gw": gw,
                    "pts": XI_MASK @ mu,
                    "xi": k,
                    "mu_xi": mu[MEMBERS[k]],
                    "cap": list(MEMBERS[k]).index(slot[int(cap.iloc[0])]),
                }
            )
        return out

    def fit(self, gw):
        if gw not in self.fits:
            if self.obs is None:
                self.obs = self._observations()
            obs = [o for o in self.obs if o["gw"] < gw]
            pts, xi = (
                np.array([o["pts"] for o in obs]),
                np.array([o["xi"] for o in obs]),
            )
            mu_xi, cap = (
                np.array([o["mu_xi"] for o in obs]),
                np.array([o["cap"] for o in obs]),
            )
            r = np.arange(len(obs))

            def nll_xi(lt):
                z = pts / np.exp(lt)
                return -(z[r, xi] - logsumexp(z, axis=1)).sum()

            def nll_cap(lt):
                z = mu_xi / np.exp(lt)
                return -(z[r, cap] - logsumexp(z, axis=1)).sum()

            t_xi = np.exp(minimize_scalar(nll_xi, bounds=(-4, 5), method="bounded").x)
            t_cap = np.exp(minimize_scalar(nll_cap, bounds=(-4, 5), method="bounded").x)
            self.fits[gw] = {
                "tau_xi": float(t_xi),
                "tau_cap": float(t_cap),
                "n_obs": len(obs),
                "ll_xi_per_obs": -nll_xi(np.log(t_xi)) / len(obs),
                "ll_xi_uniform": -np.log(len(XI_MASK)),
                "ll_cap_per_obs": -nll_cap(np.log(t_cap)) / len(obs),
                "ll_cap_uniform": -np.log(11),
            }
        return self.fits[gw]


# ---------------------------------------------------------------------------- the benchmark
def benchmark(
    gw, opponent, pr, inp, s, lm, row_budget=300_000, min_k=50, chunk_rows=250_000
):
    """The opponent's mixture over (squad, XI, captain) rows on the universe of pr (players outside it, without a
    fixture, score 0). Squad s keeps its K_s = clip(p_s * row_budget, min_k, 6050) most likely rows, so the likely
    squads keep nearly all their mass and the tail squads at least min_k rows."""
    opp = inp.league.opponent_squads(
        gw, opponent, n_samples=s.n_samples_opp, seed=s.seed, fit_mode=s.opponent_fit
    )
    fallback = opp is None or opp["n_unique"] == 0
    if fallback:
        prev = h2h.real_lineup(opponent, gw - 1, inp)
        sq = (
            prev[0][0] + prev[0][3]
            if prev is not None
            else list(inp.league.squad_players(opponent, gw - 1) or ())
        )
        counts = {tuple(sorted(sq)): 1}
    else:
        counts = opp["squad_counts"]
    squads, weights = [], []
    for sq, c in counts.items():
        o = order_squad(sq, inp.position)
        if o is not None:
            squads.append(o)
            weights.append(c)
    p_sq = np.array(weights, dtype=float) / sum(weights)

    n = len(pr["universe"])
    at = pr["at"]
    idx = np.array(
        [[at.get(e, n) for e in sq] for sq in squads]
    )  # (S, 15), n = outside the universe
    mu_pad = np.append(pr["mu"], 0.0)
    sig_pad = np.zeros((n + 1, n + 1))
    sig_pad[:n, :n] = pr["sigma"]
    fit = lm.fit(gw)
    rows_idx, rows_w, rows_p, rows_mu, rows_q, kept = [], [], [], [], [], []
    k_sq = np.clip((p_sq * row_budget).astype(int), min_k, len(ROWS))
    order = np.argsort(-k_sq, kind="stable")
    chunks, a = [], 0
    while a < len(
        order
    ):  # squads of similar K together, about chunk_rows rows per chunk
        c = max(1, min(chunk_rows // k_sq[order[a]], 400))
        chunks.append(order[a : a + c])
        a += c
    for sel in chunks:
        ix, ps, top_k = idx[sel], p_sq[sel], int(k_sq[sel].max())
        mu_sq = mu_pad[ix]
        lp = row_logp(mu_sq, fit["tau_xi"], fit["tau_cap"])
        top = (
            np.argsort(-lp, axis=1)
            if top_k >= lp.shape[1]
            else np.argpartition(-lp, top_k - 1, axis=1)
        )[:, :top_k]  # (c, K) row ids
        lpt = np.take_along_axis(lp, top, axis=1)
        pt = np.exp(lpt)
        kept.append((pt.sum(1) * ps).sum())
        pt = pt / pt.sum(1, keepdims=True) * ps[:, None]
        W = ROWS[top]  # (c, K, 15)
        S_sq = sig_pad[ix[:, :, None], ix[:, None, :]]  # (c, 15, 15)
        rows_idx.append(np.repeat(ix[:, None, :], top_k, axis=1).reshape(-1, 15))
        rows_w.append(W.reshape(-1, 15))
        rows_p.append(pt.reshape(-1))
        rows_mu.append(np.einsum("ckj,cj->ck", W, mu_sq).reshape(-1))
        rows_q.append(np.einsum("ckj,cjl,ckl->ck", W, S_sq, W).reshape(-1))
    R_idx, R_w, R_p = (
        np.concatenate(rows_idx),
        np.concatenate(rows_w),
        np.concatenate(rows_p),
    )
    R_mu, R_q = np.concatenate(rows_mu), np.concatenate(rows_q)
    wbar = np.bincount(
        R_idx.reshape(-1), weights=(R_w * R_p[:, None]).reshape(-1), minlength=n + 1
    )[:n]
    mu_g = float(R_p @ R_mu)
    return {
        "idx": R_idx,
        "W": R_w,
        "p": R_p,
        "mu_w": R_mu,
        "Q": R_q,
        "wbar": wbar,
        "mu_G": mu_g,
        "var_G": float(R_p @ R_q + R_p @ (R_mu - mu_g) ** 2),
        "n_squads": len(squads),
        "top_squad_share": float(p_sq.max()),
        "mass_kept": float(sum(kept)),
        "fallback": fallback,
        **fit,
    }


def prob_beat(v, pr, bm, cost=0.0):
    """Prob(v' delta - cost > G) under the mixture, exactly (each row's difference is normal); cost = points
    deducted from our score (transfer hits)."""
    s_vec = np.append(pr["sigma"] @ v, 0.0)
    cross = (bm["W"] * s_vec[bm["idx"]]).sum(1)
    var = v @ pr["sigma"] @ v - 2 * cross + bm["Q"]
    mean = pr["mu"] @ v - cost - bm["mu_w"]
    ok = var > 1e-9  # a row identical to our line-up is a sure draw, never a win
    z = np.where(ok, mean / np.sqrt(np.where(ok, var, 1.0)), -np.inf)
    return float(bm["p"] @ norm.cdf(z))


# ---------------------------------------------------------------------------- the r-quantile benchmark
def quantile_benchmark(pr, bm, r, n_rows=2000, n_mc=2000, seed=0, chunk=200):
    """G^(r): for each draw of delta, the r-quantile of the scores of n_rows line-ups drawn from the mixture by their
    weights (the thesis's Lemma 3.2.2 with W_o = those line-ups); mu_G, Var(G) and cov(delta, G) by Monte Carlo."""
    rng = np.random.default_rng(seed)
    k = rng.choice(len(bm["p"]), size=n_rows, p=bm["p"] / bm["p"].sum())
    idx, W = bm["idx"][k], bm["W"][k]
    delta = rng.multivariate_normal(pr["mu"], pr["sigma"], size=n_mc, method="eigh")
    dpad = np.hstack([delta, np.zeros((n_mc, 1))])
    G = np.empty(n_mc)
    for a in range(0, n_mc, chunk):
        G[a:a + chunk] = np.quantile((dpad[a:a + chunk][:, idx] * W).sum(-1), r, axis=1)
    cov = ((delta - delta.mean(0)) * (G - G.mean())[:, None]).mean(0)
    return {"delta": delta, "G": G, "mu_G": float(G.mean()), "var_G": float(G.var()), "cov_g": cov, "r": r}


def prob_beat_mc(v, qb, cost=0.0):
    """Prob(v' delta - cost > G^(r)) over the Monte Carlo draws."""
    return float((qb["delta"] @ v - cost > qb["G"]).mean())
