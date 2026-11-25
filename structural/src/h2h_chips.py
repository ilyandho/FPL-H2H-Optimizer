"""Opponents' chips for the 2024-25 head-to-head league: which chips a manager has left and how likely he is to play one.

In 2024-25 our opponents played a chip in 8 of our 33 matches (GW6-38); the optimizers' Prob(beat) was 66% in those
weeks and they won 38% (48% in the others): the opponent model's G never saw chips. This module gives the chance an
opponent plays each chip in a gameweek, from information known at the deadline:

  chips played    read from each Fantasy Football Scout manager's picks (h2h_data.py): Triple Captain (captain
                  multiplier 3), Bench Boost (every bench multiplier > 0), Wildcard / Free Hit (more than 8 of the 15
                  changed from the week before; a Free Hit if the squad comes back the week after, at most 3 changes
                  from the week before it). At a deadline the last week's overhaul is a Wildcard unless the manager's
                  Wildcard for that half is already used
  chips left      2024-25: a Wildcard per half (GW1-19, GW20-38), one Free Hit, Bench Boost and Triple Captain
  hazard          (mode "hazard", known at the deadline) per chip, P(play | left) = logistic(a + b_spec [special
                  week] + b_best [best remaining special week] + b_left log(weeks left to play it) + b_last [last week
                  it can be played]); special weeks: a double gameweek for Bench Boost, the best captain's double for
                  Triple Captain (his points per match times fixtures, predictions at the deadline), a blank for the Free
                  Hit; with no special week left in the window, END_RULE / weeks left. Fitted by maximum a posteriori on
                  every FFS manager-week before the gameweek's previous one in which the chip was left, with Gaussian
                  priors (PRIOR, PRIOR_SD) set before looking at results. Log loss on the league's managers GW6-38
                  against a constant rate: Bench Boost 0.134 / 0.154, Free Hit 0.086 / 0.143, Triple Captain
                  0.166 / 0.172, Wildcard 0.387 / 0.365; it knows which chips are left and roughly when, but no earlier
                  week shows the double-gameweek rushes (Bench Boost in GW33: 3% predicted, about half of the holders
                  played it), and priors set after seeing them would be tuned on the season tested
  population      (mode "population", an upper bound: known only after the deadline) the share of FFS managers still
                  holding the chip who played it that week: how much a good forecast of chip timing would be worth
  one chip a week the probabilities are scaled down to sum to at most 0.95

The 5,147 managers' chip plays by gameweek concentrate where expected: Triple Captain 22% in the double gameweeks 24
and 32, Bench Boost 44% in GW33 (the largest double gameweek).
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import config

CHIPS = ("tc", "bb", "fh", "wc")
# prior means of (intercept, value / best remaining, best remaining week, log weeks left, last week), set before
# looking at results: a chip is played in (or near) its best remaining week, and as its window runs out
# prior means of (intercept, value / best remaining, best remaining special week, no special week left, log weeks
# left, last week): a chip is played in its best remaining special week (a double gameweek for Bench Boost, the best
# captain's double for Triple Captain, a blank for the Free Hit), else used up late in its window
# (intercept, special week, best remaining special week, log weeks left, last week); a week with no special week left
# in the chip's window is handled by END_RULE instead (no earlier week shows it)
PRIOR = {"tc": (-6.0, 3.0, 0.5, -0.3, 1.5), "bb": (-6.0, 2.0, 1.5, -0.3, 1.5),
         "fh": (-6.0, 2.0, 1.5, -0.3, 1.5), "wc": (-3.5, 0.0, 0.0, -0.3, 1.5)}
PRIOR_SD = np.array([1.5, 1.0, 1.0, 1.0, 1.0])
END_RULE = 0.5  # with no special week left, P(play) = END_RULE / weeks left in the window


def chip_histories(picks):
    """DataFrame (team_id, gw, chip) of every chip each manager played, from his picks."""
    picks = picks.sort_values(["team_id", "gw", "slot"])
    rows = []
    for team, g in picks.groupby("team_id"):
        weeks = {w: d for w, d in g.groupby("gw")}
        squads = {w: set(d["element"]) for w, d in weeks.items()}
        for w, d in weeks.items():
            chip = None
            if (d.loc[d["is_captain"], "multiplier"] == 3).any():
                chip = "tc"
            elif (d.loc[d["slot"] > 11, "multiplier"] > 0).all():
                chip = "bb"
            elif w - 1 in squads and len(squads[w] - squads[w - 1]) > 8 and not (rows and rows[-1][:3] == (team, w - 1, "fh")):
                back = w + 1 in squads and len(squads[w + 1] - squads[w - 1]) <= 3
                chip = "fh" if back else "wc"  # (the squad's return the week after a Free Hit is not a chip)
            if chip:
                rows.append((team, w, chip))
    return pd.DataFrame(rows, columns=["team_id", "gw", "chip"])


def week_table(inp):
    """Per gameweek: clubs with two fixtures, clubs without a fixture."""
    o = inp.opponents
    n = o.groupby(["round", "team"]).size()
    clubs = o["team"].nunique()
    out = {}
    for gw in range(1, 39):
        c = n.xs(gw) if gw in n.index.get_level_values(0) else pd.Series(dtype=int)
        out[gw] = {"double": int((c >= 2).sum()), "blank": int(clubs - len(c))}
    return out


def window_end(chip, gw):
    return 19 if chip == "wc" and gw <= 19 else 38


def chip_values(inp):
    """{deadline d: {chip: {week t >= d: value}}}: Triple Captain the best captain's expected points in week t with
    the predictions known at d (points per match at d times fixtures in t), Bench Boost 1 + double clubs, Free Hit
    1 + blank clubs."""
    weeks = week_table(inp)
    fx = inp.opponents.groupby(["round", "team"]).size()
    preds = inp.preds
    out = {}
    for d in range(1, 39):
        cur = preds[preds["round"] == d].drop_duplicates("fpl_id")
        if cur.empty:
            continue
        rate = (cur["pred"] / cur["n_fixtures"].clip(lower=1)).to_numpy()
        clubs = cur["team"].to_numpy()
        out[d] = {"tc": {}, "bb": {}, "fh": {}}
        for t in range(d, 39):
            n = fx.xs(t) if t in fx.index.get_level_values(0) else pd.Series(dtype=int)
            out[d]["tc"][t] = float((rate * n.reindex(clubs).fillna(0).to_numpy()).max())
            out[d].setdefault("tc_special", {})[t] = bool((n.reindex(clubs).fillna(0).to_numpy()[
                np.argmax(rate * n.reindex(clubs).fillna(0).to_numpy())] >= 2))  # the best captain plays twice
            out[d]["bb"][t] = 1.0 + weeks[t]["double"]
            out[d]["fh"][t] = 1.0 + weeks[t]["blank"]
    return out


def special(chip, t, deadline, values):
    if chip == "tc":
        return values[deadline]["tc_special"][t]
    return values[deadline][chip][t] > 1.0  # a double (Bench Boost) or blank (Free Hit) gameweek


def none_left(chip, t, deadline, values):
    """No special week left for the chip from week t to the end of its window (the Wildcard has none)."""
    return chip != "wc" and not any(special(chip, u, deadline, values) for u in range(t, window_end(chip, t) + 1))


def features(chip, t, deadline, values):
    """The hazard's features for playing `chip` in week t, as seen at `deadline`."""
    end = window_end(chip, t)
    spec = best = 0.0
    if chip != "wc":
        v = values[deadline][chip]
        specials = [u for u in range(t, end + 1) if special(chip, u, deadline, values)]
        spec = float(t in specials)
        best = float(t in specials and v[t] >= max(v[u] for u in specials) - 1e-9)
    return np.array([1.0, spec, best, np.log(end - t + 1), float(t == end)])


def used_before(history, gw):
    """{chip key: gw} of the chips played before gw (keys tc, bb, fh, wc1, wc2), as known at gw's deadline: last week's
    Free Hit is told from a Wildcard only by week gw's squad, so it counts as a Wildcard unless that half's is used."""
    out = {}
    for w, c in sorted(zip(history["gw"], history["chip"])):
        if w >= gw:
            continue
        half = "wc1" if w <= 19 else "wc2"
        if c == "fh" and w == gw - 1 and half not in out:
            c = "wc"
        out[half if c == "wc" else c] = w
    return out


def left(used, gw):
    """The chips still playable in gameweek gw."""
    return [c for c in CHIPS if not used.get(("wc1" if gw <= 19 else "wc2") if c == "wc" else c)]


class ChipModel:
    """Population chip histories and the per-chip hazard, refitted for each gameweek on the weeks before it."""

    def __init__(self, inp):
        p = config.paths(inp.league_season)
        path = Path(p.h2h_dir) / "ffs_chips.pkl"
        if path.exists():
            self.history = pickle.loads(path.read_bytes())
        else:
            self.history = chip_histories(pd.read_csv(f"{p.ffs_dir}/picks.csv.gz"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pickle.dumps(self.history))
        self.by_team = {t: g for t, g in self.history.groupby("team_id")}
        self.teams = pd.read_csv(f"{p.ffs_dir}/managers.csv").query("has_picks")["team_id"].tolist()
        self.weeks = week_table(inp)
        self.values = chip_values(inp)
        self._fits = {}
        self._shares = None

    def _team(self, team):
        return self.by_team.get(team, pd.DataFrame(columns=["gw", "chip"]))

    def fit(self, gw):
        """{chip: coefficients}, MAP on every manager-week before gw in which the chip was left."""
        if gw in self._fits:
            return self._fits[gw]
        X = {c: [] for c in CHIPS}
        y = {c: [] for c in CHIPS}
        for team in self.teams:
            h = self._team(team)
            played = dict(zip(h["gw"], h["chip"]))
            for w in range(2, gw - 1):  # week gw-1's Free Hit / Wildcard label needs week gw's squad
                for c in left(used_before(h, w), w):
                    if none_left(c, w, w, self.values):
                        continue
                    X[c].append(features(c, w, w, self.values))
                    y[c].append(float(played.get(w) == c))
        fits = {}
        for c in CHIPS:
            Xc, yc, mu0 = np.array(X[c]).reshape(-1, len(PRIOR_SD)), np.array(y[c]), np.array(PRIOR[c])

            def nlp(b):
                z = Xc @ b
                return np.sum(np.logaddexp(0, z) - yc * z) + 0.5 * np.sum(((b - mu0) / PRIOR_SD) ** 2)

            def grad(b):
                pz = 1 / (1 + np.exp(-(Xc @ b)))
                return Xc.T @ (pz - yc) + (b - mu0) / PRIOR_SD ** 2

            fits[c] = minimize(nlp, mu0, jac=grad, method="L-BFGS-B").x
        self._fits[gw] = fits
        return fits

    def population_shares(self):
        """{gw: {chip: share of the managers holding it at gw who played it}} (known only after the deadline)."""
        if self._shares is None:
            held = {gw: dict.fromkeys(CHIPS, 0) for gw in range(1, 39)}
            played = {gw: dict.fromkeys(CHIPS, 0) for gw in range(1, 39)}
            for team in self.teams:
                h = self._team(team)
                done = dict(zip(h["gw"], h["chip"]))
                for w in range(2, 39):
                    used = {}
                    for ww, cc in sorted(zip(h["gw"], h["chip"])):
                        if ww < w:
                            used[("wc1" if ww <= 19 else "wc2") if cc == "wc" else cc] = ww
                    for c in left(used, w):
                        held[w][c] += 1
                        played[w][c] += int(done.get(w) == c)
            self._shares = {w: {c: played[w][c] / held[w][c] if held[w][c] else 0.0 for c in CHIPS} for w in held}
        return self._shares

    def probs(self, team, gw, fit_gw=None, mode="hazard"):
        """{chip: P(team plays it in gameweek gw)} from the fit at fit_gw (default gw: the deadline), for the chips he
        has left at fit_gw's deadline; summing to at most 0.95."""
        fit_gw = fit_gw or gw
        used = used_before(self._team(team), fit_gw)
        if mode == "population":
            share = self.population_shares()[gw]
            out = {c: share[c] for c in left(used, gw)}
            total = sum(out.values())
            return {c: p * 0.95 / total for c, p in out.items()} if total > 0.95 else out
        fits = self.fit(fit_gw)
        out = {}
        for c in left(used, gw):
            if none_left(c, gw, fit_gw, self.values):
                out[c] = END_RULE / (window_end(c, gw) - gw + 1)
            else:
                out[c] = float(1 / (1 + np.exp(-(features(c, gw, fit_gw, self.values) @ fits[c]))))
        total = sum(out.values())
        if total > 0.95:
            out = {c: p * 0.95 / total for c, p in out.items()}
        return out
