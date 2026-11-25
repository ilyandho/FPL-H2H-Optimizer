"""Chapter 4 figures that are not in structural_components.ipynb or predictor_comparison.ipynb: the covariance model
(4.2.7-4.2.8), the opponent model (4.3), worked frontiers (4.4) and the results of both seasons (4.5-4.6).
Writes PDFs to working/figures/ (and a PNG of each to --png-dir, for a quick look).

  sigma   loadings a_p over the season; observed same-club (and opposing) error correlations against the factor
          model's a_p a_q; squad-level z = (realised - w'mu) / sqrt(w' Sigma w) for real teams: every FantasyGo entry
          (2023-24) and every H2H league manager's XI (2024-25), on Sigma = the team-factor model of each gameweek
  sigmap  player-level sigma_p against the realised errors (4.2.6)
  h2h     from working/Optimizer/outputs (baseline.py's caches, the 6 teams): cumulative points relative to blind,
          a forest plot of the paired comparisons with block-bootstrap intervals, hits per gameweek, and the
          reliability of P(beat) (each manager's P(beat) against the opponent under the mixture, prob_mix)
  fgmethods  every FantasyGo method (r sweep's best, top-heavy and its fixes) against blind (4.5; FG_VARIANTS)
  fantasygo  the 2023-24 factor-Sigma r sweep (points against blind, percentile, reliability of P(beat) per r) and
          the contests at the chosen r (FG_R_STAR)
  opponents  the line-up temperatures tau_XI, tau_cap over 2024-25 and the PIT of the opponents' realised points
          under the mixture (4.3)
  field   the simulated FantasyGo field against the real one (4.3.1; mvt.field_check at FG_R_STAR)
  squads  the H2H squad model against holding / no copula / the league-average fit (4.3.2-4.3.3; opponent_eval.py)
  frontiers  worked gameweeks (4.4): FantasyGo's cached frontier and two replayed H2H decisions of the own team

Usage (ds_env, from working/):
  python thesis_figures.py sigma h2h fantasygo opponents frontiers
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
OPT = HERE / "Optimizer"
sys.path.insert(0, str(OPT))
import baseline  # noqa: E402  (puts structural/src on the path and chdirs there)
import h2h  # noqa: E402
import mvt  # noqa: E402

# style: the same as structural_components.ipynb
BLUE, ORANGE, AQUA, YELLOW, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8f98", "#22252b", "#5b6068"
SEASON_COLOR = {"23-24": BLUE, "24-25": ORANGE}
SEASON_MARKER = {"23-24": "o", "24-25": "s"}
POS_COLOR = {"GK": BLUE, "DEF": ORANGE, "MID": AQUA, "FWD": YELLOW}  # palette slots 1-4 (lines: adjacent pairs)
POS_MARKER = {"GK": "o", "DEF": "s", "MID": "^", "FWD": "D"}
# H2H managers: hue = family, line style / marker = hit level
FAMILY_COLOR = {"blind": BLUE, "mvt": ORANGE, "cal": YELLOW, "real": AQUA, "hold": GREY}
HIT_LS = {0: "-", 1: "--", 2: ":"}
# the H2H managers of Chapter 4.6 (family, hits); blind takes no hits (the baseline)
MANAGERS = {"blind": ("blind", 0), "mvt_mix_r50": ("mvt", 0), "mvt_mix_h1_r50": ("mvt", 1), "mvt_mix_h2_r50": ("mvt", 2),
            "mvt_cal_r50": ("cal", 0), "mvt_cal_h1_r50": ("cal", 1), "mvt_cal_h2_r50": ("cal", 2)}
LABEL = {"blind": "Baseline", "blind_h1": "Baseline, ≤1 hit", "blind_h2": "Baseline, ≤2 hits",
         "mvt_mix_r50": "MVT", "mvt_mix_h1_r50": "MVT, ≤1 hit", "mvt_mix_h2_r50": "MVT, ≤2 hits",
         "mvt_cal_r50": "Calibrated MVT", "mvt_cal_h1_r50": "Calibrated MVT, ≤1 hit",
         "mvt_cal_h2_r50": "Calibrated MVT, ≤2 hits", "real": "Real managers", "hold": "Hold the squad"}
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.grid": True, "grid.color": "#e3e5e8", "grid.linewidth": 0.6,
    "lines.linewidth": 1.6, "lines.markersize": 3, "figure.dpi": 120, "legend.frameon": False,
})
POS = ["GK", "DEF", "MID", "FWD"]
PNG_DIR = None


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    if PNG_DIR:
        fig.savefig(Path(PNG_DIR) / f"{name}.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print(f"wrote {FIG / name}.pdf")


# ---------------------------------------------------------------------------- the covariance model
def season_inputs():
    """{season: (inputs, settings, decision gameweeks, a function gw -> [(team vector builder)])}."""
    s23 = mvt.Settings()  # sigma = "factor"
    i23 = mvt.Inputs(s23)
    s24 = h2h.Settings()
    i24 = h2h.Inputs(s24)
    return {"23-24": (i23, s23, [g for g in i23.contest_gws if g >= 6]), "24-25": (i24, s24, list(range(6, 39)))}


def loadings_and_correlations(inp, s, gws):
    """a_p per gameweek, and the pooled correlation table of the last gameweek (every earlier gameweek's errors)."""
    rows = []
    for gw in gws:
        hist = inp.preds[inp.preds["round"] < gw]
        a = h2h.team_loadings(mvt.correlation_table(hist, inp.opponents, s.min_history))
        rows.append({"gw": gw, **a})
    last = mvt.correlation_table(inp.preds[inp.preds["round"] <= max(gws)], inp.opponents, s.min_history)
    return pd.DataFrame(rows).set_index("gw"), last


def z_scores(season, inp, s, gws):
    """(realised - w'mu) / sqrt(w' Sigma w) for every real team: FantasyGo entries (C x2, VC x1.5) or the league
    managers' XIs (captain x his multiplier), each gameweek on that gameweek's mu and factor Sigma."""
    out = []
    for gw in gws:
        if season == "23-24":
            players, mu, sigma = mvt.build_mu_sigma(gw, inp, s)
            W = mvt.teams_matrix(mvt.entries(gw, inp), players)
        else:
            players, mu, sigma, _ = h2h.build_mu_sigma(gw, inp, s)
            at = {e: i for i, e in enumerate(players.index)}
            vecs = []
            for team in inp.managers:
                r = h2h.real_lineup(team, gw, inp)
                if r is None or r[2]:  # no picks, or a Bench Boost
                    continue
                (starters, captain, _, _), mult, _ = r
                v = np.zeros(len(players))
                for e in starters:
                    if e in at:
                        v[at[e]] = mult if e == captain else 1.0
                vecs.append(v)
            W = np.array(vecs)
        realised = players["total_points"].to_numpy()
        pred, sd = W @ mu, np.sqrt(np.einsum("ij,jk,ik->i", W, sigma, W))
        out.append(pd.DataFrame({"season": season, "gw": gw, "z": (W @ realised - pred) / sd, "sd": sd}))
    return pd.concat(out, ignore_index=True)


def fig_sigma_p():
    """4.2.6: is each player's sigma_p the size of his error? RMS error against predicted sigma_p (deciles), and the
    ratio RMS error / RMS sigma_p by position, every player with a fixture in every decision gameweek."""
    rows = []
    for season, (inp, s, gws) in season_inputs().items():
        for gw in gws:
            if season == "23-24":
                players, mu, sigma = mvt.build_mu_sigma(gw, inp, s)
            else:
                players, mu, sigma, _ = h2h.build_mu_sigma(gw, inp, s)
            rows.append(pd.DataFrame({"season": season, "gw": gw, "pos": players["pos"].to_numpy(), "mu": mu,
                                      "sd": np.sqrt(np.diag(sigma)), "y": players["total_points"].to_numpy()}))
    d = pd.concat(rows, ignore_index=True)
    d["e"] = d["y"] - d["mu"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9, 3.6))
    top = 0
    for season, g in d.groupby("season"):
        b = pd.qcut(g["sd"].rank(method="first"), 10, labels=False)
        c = g.groupby(b).agg(sd=("sd", lambda x: np.sqrt((x ** 2).mean())), rmse=("e", lambda x: np.sqrt((x ** 2).mean())))
        ax.plot(c["sd"], c["rmse"], "-" + SEASON_MARKER[season], color=SEASON_COLOR[season], lw=1.5, ms=5.5,
                mec="white", mew=1, label=f"20{season}")
        top = max(top, c.max().max())
    ax.plot([0, top * 1.05], [0, top * 1.05], color=GREY, lw=0.8)
    ax.set_xlabel(r"Predicted $\sigma_p$ (RMS within the decile)")
    ax.set_ylabel("RMS error (points)")
    ax.set_title(r"(a) Error against $\sigma_p$, deciles of $\sigma_p$", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=7.5)
    ratio = d.groupby(["season", "pos"]).apply(
        lambda g: np.sqrt((g["e"] ** 2).mean() / (g["sd"] ** 2).mean()), include_groups=False).unstack().reindex(columns=POS)
    x = np.arange(len(POS))
    for k, (season, r) in enumerate(ratio.iterrows()):
        bx.bar(x + (k - 0.5) * 0.36, r.to_numpy(), width=0.34, color=SEASON_COLOR[season], edgecolor="white",
               linewidth=1, label=f"20{season}")
    bx.axhline(1, color=INK, lw=1, ls="--")
    bx.set_xticks(x, POS)
    bx.set_ylabel(r"RMS error / RMS $\sigma_p$")
    bx.set_title("(b) Scale by position (1 = right on average)", fontsize=9.5)
    from matplotlib.patches import Patch

    bx.legend(handles=[Patch(facecolor=SEASON_COLOR[se], label=f"20{se}") for se in ratio.index], loc="upper left",
              ncol=2, fontsize=7.5)
    bx.set_ylim(0, 1.22)
    fig.suptitle(r"Player-level $\sigma_p$ against the realised errors", fontsize=10.5)
    fig.tight_layout()
    save(fig, "sigma_player")
    z = d["e"] / d["sd"]
    print(ratio.round(3).to_string())
    for season, g in d.assign(z=z).groupby("season"):
        print(f"{season}: n {len(g)}, sd(z) {g['z'].std():.3f}, |z|<1 {(g['z'].abs() < 1).mean():.3f} "
              f"(normal 0.683), |z|<2 {(g['z'].abs() < 2).mean():.3f} (0.954), z>2 {(g['z'] > 2).mean():.3f}, "
              f"z<-2 {(g['z'] < -2).mean():.3f}")


def fig_sigma():
    data = season_inputs()
    loads, corr, zs = {}, {}, []
    for season, (inp, s, gws) in data.items():
        loads[season], corr[season] = loadings_and_correlations(inp, s, gws)
        zs.append(z_scores(season, inp, s, gws))
        print(season, "loadings at the last gameweek:", loads[season].iloc[-1].round(3).to_dict())
    z = pd.concat(zs)

    # Figure 1: loadings over the season (one panel per season) and observed vs implied correlations
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for ax, season in zip(axes[:2], loads):
        for p in POS:
            ax.plot(loads[season].index, loads[season][p], color=POS_COLOR[p], marker=POS_MARKER[p], ms=3,
                    label=p)
        ax.set_title(f"Loadings a_p over 20{season}")
        ax.set_xlabel("Gameweek (errors of the gameweeks before it)")
        ax.set_ylim(0, 0.6)
    axes[0].set_ylabel("a_p")
    axes[0].legend(loc="upper right", ncol=4, handletextpad=0.3, columnspacing=0.8)
    ax = axes[2]
    lim = [-0.04, 0.12]
    ax.plot(lim, lim, color=GREY, lw=0.8, zorder=0)
    ax.axhline(0, color=GREY, lw=0.6, zorder=0)
    for season in corr:
        a = loads[season].iloc[-1]
        t = corr[season].reset_index()
        same, opp = t[t["rel"] == "same"], t[t["rel"] == "opp"]
        implied = [a[p] * a[q] for p, q in same["pos_pair"]]
        ax.scatter(same["rho"], implied, color=SEASON_COLOR[season], marker=SEASON_MARKER[season], s=22,
                   label=f"Same club, 20{season}", zorder=3, edgecolor="white", linewidth=0.5)
        ax.scatter(opp["rho"], np.zeros(len(opp)), facecolor="none", edgecolor=SEASON_COLOR[season],
                   marker=SEASON_MARKER[season], s=22, label=f"Opposing, 20{season} (set to 0)", zorder=3)
        if season == "24-25":
            for (_, r), y in zip(same.iterrows(), implied):
                if y > 0.06 or abs(y - r["rho"]) > 0.03:
                    ax.annotate("-".join(r["pos_pair"]), (r["rho"], y), xytext=(4, -9 if r["pos_pair"] == ("GK", "GK") else -2), textcoords="offset points",
                                fontsize=7, color=MUTED)
    ax.set_xlim(lim)
    ax.set_ylim(-0.01, 0.155)
    ax.set_xlabel("Observed correlation of the errors")
    ax.set_ylabel("Factor model, a_p a_q")
    ax.set_title("Observed against implied, end of season")
    ax.legend(loc="upper left", fontsize=6.5, borderaxespad=0.2)
    fig.tight_layout()
    save(fig, "sigma_loadings")
    for season in corr:
        a = loads[season].iloc[-1]
        same = corr[season].reset_index().query("rel == 'same'")
        print(season, "same-club pairs, observed vs implied:",
              {"-".join(pp): (round(r, 3), round(a[pp[0]] * a[pp[1]], 3)) for pp, r in zip(same["pos_pair"], same["rho"])})

    # Figure 2: squad-level z-scores against N(0, 1)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
    grid = np.linspace(-4, 4, 200)
    rows = []
    for k, season in enumerate(["23-24", "24-25"]):
        x = z.loc[z["season"] == season, "z"].to_numpy()
        ax = axes[k]
        ax.hist(x, bins=np.arange(-4, 4.01, 0.25), density=True, color=SEASON_COLOR[season], alpha=0.85,
                edgecolor="white", linewidth=0.6)
        ax.plot(grid, stats.norm.pdf(grid), color=INK, lw=1.2, label="N(0, 1)")
        ax.set_title(f"20{season}: {'FantasyGo entries' if season == '23-24' else 'league XIs'} (n = {len(x):,})")
        ax.set_xlabel("z = (realised − predicted) / predicted sd")
        ax.text(0.98, 0.95, f"mean {x.mean():+.2f}\nsd {x.std():.2f}\n|z| > 1.96: {np.mean(np.abs(x) > 1.96):.1%}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK)
        q = np.linspace(0.005, 0.995, 199)
        axes[2].plot(stats.norm.ppf(q), np.quantile(x, q), color=SEASON_COLOR[season], marker=SEASON_MARKER[season],
                     ms=2.5, lw=1.2, label=f"20{season}")
        gwm = z[z["season"] == season].groupby("gw")["z"].mean()
        rows.append({"season": season, "n": len(x), "mean": x.mean(), "sd": x.std(),
                     "share |z|>1.96": np.mean(np.abs(x) > 1.96), "skew": stats.skew(x),
                     "sd of gameweek means": gwm.std(), "mean predicted sd": z.loc[z["season"] == season, "sd"].mean()})
    axes[0].set_ylabel("Density")
    axes[0].legend(loc="upper left")
    axes[2].plot([-3, 3], [-3, 3], color=GREY, lw=0.8, zorder=0)
    axes[2].set_title("Quantiles against N(0, 1)")
    axes[2].set_xlabel("N(0, 1) quantile")
    axes[2].set_ylabel("z quantile")
    axes[2].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "sigma_zscores")
    print(pd.DataFrame(rows).set_index("season").round(3).to_string())


# ---------------------------------------------------------------------------- the H2H results
def hold_points(inp):
    """Every league team's points with its GW5 squad held all season (XI and captain by mu each gameweek), cached."""
    path = OPT / "outputs" / "hold_manager.pkl"
    out = pickle.loads(path.read_bytes()) if path.exists() else {}
    s = h2h.Settings()
    for team in baseline.LEAGUE:
        if team in out and len(out[team]) == 33:
            continue
        squad = sorted(inp.league.squad_players(team, s.start_gw - 1))
        out[team] = {}
        for g in range(6, 39):
            week = inp.week(g)
            lu = h2h.pick_lineup(squad, week["pred"].to_dict(), inp.position)
            out[team][g] = h2h.lineup_points(lu, week, inp.position)[1]
        path.write_bytes(pickle.dumps(out))
    return out


def h2h_frames(teams=None):
    """Per team (default: the 20 league teams): {manager: per-gameweek rows} (the MVT managers, blind with and without
    hits), the real and hold points, the opponent's points and the weeks without an opponent's chip."""
    inp = h2h.Inputs(h2h.Settings())
    hold = hold_points(inp)
    out = {}
    for team in teams or baseline.LEAGUE:
        cache = pickle.loads(baseline.cache_path(team).read_bytes())
        full = {m: pd.DataFrame(cache[m].values()).set_index("gw").sort_index().loc[6:38]
                for m in (*MANAGERS, "blind_h1", "blind_h2")}
        base = full["blind"]
        chips = set(baseline.chip_weeks(inp, base["opponent"].to_dict()))
        out[team] = {"rows": full, "real": base["real_points"].astype(float), "opp": base["opponent_points"].astype(float),
                     "hold": pd.Series(hold[team]).astype(float), "weeks": [g for g in base.index if g not in chips]}
    return out, inp


def _pts(d, m):
    return d[m] if m in ("real", "hold") else d["rows"][m]["points"].astype(float)


def _h2h(d, m):
    x, o = _pts(d, m), d["opp"]
    return pd.Series(np.where(x > o, 3.0, np.where(x == o, 1.0, 0.0)), index=x.index)


def fig_h2h():
    data, inp = h2h_frames()
    gws = list(range(6, 39))
    n_t = len(data)
    shown = [*MANAGERS, "real", "hold"]

    def style(m):
        if m in ("real", "hold"):
            fam, hits = m, 0
        elif m in MANAGERS:
            fam, hits = MANAGERS[m]
        else:  # blind_h1 / blind_h2 (the supporting comparison)
            fam, hits = "blind", int(m[-1])
        return FAMILY_COLOR[fam], HIT_LS[hits]

    # Figure 1: cumulative H2H points (3 a win, 1 a draw) relative to blind, the 20 teams together, every gameweek
    fig, ax = plt.subplots(figsize=(7.5, 3.9))
    ref = sum(_h2h(data[t], "blind").reindex(gws).fillna(0) for t in data).cumsum()
    ax.axhline(0, color=GREY, lw=0.8)
    for m in shown:
        if m == "blind":
            continue
        y = sum(_h2h(data[t], m).reindex(gws).fillna(0) for t in data).cumsum() - ref
        col, ls = style(m)
        ax.plot(gws, y, color=col, ls=ls, lw=1.9 if m in ("real", "mvt_cal_h1_r50") else 1.3,
                label=f"{LABEL[m]} ({y.iloc[-1]:+.0f})")
    ax.set_xlim(6, 38)
    ax.set_xlabel("Gameweek")
    ax.set_ylabel(f"Cumulative H2H points − baseline's\n({n_t} teams together)")
    ax.set_title(f"H2H points against the baseline (no hits), 2024-25, {n_t} teams, every gameweek")
    ax.legend(loc="upper left", ncol=2, fontsize=7)
    fig.tight_layout()
    save(fig, "h2h_cumulative")

    # Figure 2: forest plot of paired per-week differences in H2H points (weeks without an opponent's chip)
    groups = [
        ("Against the baseline (no hits)", [(m, "blind") for m in MANAGERS if m != "blind"] + [("hold", "blind")]),
        ("Calibrated − uncalibrated MVT", [("mvt_cal_r50", "mvt_mix_r50"), ("mvt_cal_h1_r50", "mvt_mix_h1_r50"),
                                           ("mvt_cal_h2_r50", "mvt_mix_h2_r50")]),
        ("Supporting: the baseline allowed the same hits", [("mvt_cal_h1_r50", "blind_h1"), ("mvt_cal_h2_r50", "blind_h2"),
                                                     ("blind_h1", "blind")]),
        ("Against the real managers", [(m, "real") for m in ("blind", "mvt_mix_h1_r50", "mvt_cal_h1_r50",
                                                              "mvt_cal_h2_r50")]),
    ]
    rows = []
    for g, pairs in groups:
        for a, b in pairs:
            per = [(_h2h(d, a) - _h2h(d, b)).loc[d["weeks"]] for d in data.values()]
            ptd = [(_pts(d, a) - _pts(d, b)).loc[d["weeks"]] for d in data.values()]
            boot = baseline.block_bootstrap(per)
            rows.append({"group": g, "pair": f"{LABEL[a]} − {LABEL[b]}", "a": a, "mean": pd.concat(per).mean(),
                         "points": pd.concat(ptd).mean(), "teams ahead": sum(x.mean() > 0 for x in per), **boot})
    f = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    y, ticks, labels = 0, [], []
    for g, pairs in groups[::-1]:
        sub = f[f["group"] == g].iloc[::-1]
        for _, r in sub.iterrows():
            col, _ = style(r["a"])
            ax.plot([r["CI low"], r["CI high"]], [y, y], color=col, lw=2, solid_capstyle="round")
            ax.plot(r["mean"], y, marker="o", ms=6, color=col, markeredgecolor="white", mew=0.8)
            ax.text(0.82, y, f"{r['mean']:+.3f}  (p = {r['p']:.3f})", va="center", fontsize=7.5, color=INK)
            ticks.append(y)
            labels.append(r["pair"])
            y += 1
        ax.text(-1.17, y - 0.2, g, fontsize=8.5, fontweight="bold", color=INK)
        y += 1.6
    ax.axvline(0, color=MUTED, lw=0.9)
    ax.set_ylim(-0.8, y - 0.6)
    ax.set_yticks(ticks, labels, fontsize=7.5)
    ax.set_xlim(-1.18, 0.8)
    ax.set_xlabel("Difference in H2H points per gameweek (3 a win, 1 a draw; block-bootstrap 95% interval)")
    n_w = sum(len(d["weeks"]) for d in data.values())
    ax.set_title(f"Paired comparisons in H2H points, 2024-25, {n_t} teams, {n_w} team-weeks without an opponent's chip",
                 pad=10, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    save(fig, "h2h_forest")
    pd.set_option("display.width", 200)
    print(f[["pair", "mean", "CI low", "CI high", "p", "teams ahead", "points"]].round(3).to_string(index=False))

    # W-D-L table (weeks without an opponent's chip)
    tab = []
    for m in [*shown, "blind_h1", "blind_h2"]:
        W = D = L = n = 0
        pts = 0.0
        for d in data.values():
            x, o = _pts(d, m).loc[d["weeks"]], d["opp"].loc[d["weeks"]]
            W += int((x > o).sum()); D += int((x == o).sum()); L += int((x < o).sum()); n += len(x); pts += x.sum()
        tab.append({"manager": LABEL[m], "W-D-L": f"{W}-{D}-{L}", "H2H points / week": (3 * W + D) / n,
                    "points / week": pts / n})
    print(pd.DataFrame(tab).set_index("manager").round(3).to_string())

    # Figure 3: hits per gameweek, uncalibrated against calibrated MVT, the teams together
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 3.9), sharex=True, sharey=True)
    w = 0.38
    for ax, (mv, cal, title) in zip(axes, [("mvt_mix_h1_r50", "mvt_cal_h1_r50", "At most 1 hit a gameweek"),
                                            ("mvt_mix_h2_r50", "mvt_cal_h2_r50", "At most 2 hits a gameweek")]):
        for k, m in enumerate([mv, cal]):
            h = sum(data[t]["rows"][m]["hits"].reindex(gws).fillna(0) for t in data)
            ax.bar(np.array(gws) + (k - 0.5) * (w + 0.04), h, width=w, color=style(m)[0],
                   label=f"{LABEL[m]} ({int(h.sum())} in all)")
        ax.set_title(title, loc="left")
        ax.legend(loc="upper left", ncol=2, fontsize=7.5)
        ax.grid(axis="x", visible=False)
    axes[1].set_xlabel("Gameweek")
    for ax in axes:
        ax.set_ylabel(f"Hits ({n_t} teams)")
    fig.tight_layout()
    save(fig, "h2h_hits")

    # Figure 4: reliability of each manager's own P(beat) (weeks without an opponent's chip)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.8, zorder=0)
    for fam, ms, col in [("MVT", ("mvt_mix_h1_r50", "mvt_mix_h2_r50"), ORANGE),
                         ("Calibrated MVT", ("mvt_cal_h1_r50", "mvt_cal_h2_r50"), YELLOW)]:
        p, won = [], []
        for d in data.values():
            for m in ms:
                r = d["rows"][m].loc[d["weeks"]]
                p.append(r["prob"].to_numpy())
                won.append((r["points"] > r["opponent_points"]).to_numpy() + 0.5 * (r["points"] == r["opponent_points"]).to_numpy())
        p, won = np.concatenate(p), np.concatenate(won)
        b = pd.qcut(pd.Series(p).rank(method="first"), 8, labels=False)
        c = pd.DataFrame({"p": p, "w": won}).groupby(b).agg(p=("p", "mean"), w=("w", "mean"), n=("w", "size"))
        se = np.sqrt(c["w"] * (1 - c["w"]) / c["n"])
        ax.errorbar(c["p"], c["w"], yerr=1.96 * se, color=col, marker="o", ms=4, lw=1.4, capsize=0, elinewidth=0.9,
                    label=f"{fam}, ≤1 and ≤2 hits (n = {len(p)})")
        print(fam, "mean P(beat)", p.mean().round(3), "won", won.mean().round(3), "Brier", np.mean((p - won) ** 2).round(3))
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.05, 0.95)
    ax.set_xlabel("Predicted P(beat the opponent)")
    ax.set_ylabel("Share of gameweeks won (draw = ½)")
    ax.set_title("Reliability of P(beat), 2024-25")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    save(fig, "h2h_reliability")

    # Figure 5: league rank if the manager had replaced the team's real manager (all gameweeks from GW6)
    import h2h_opponents as ho

    real = {(m, g): h2h.real_points(m, g, inp) for m in inp.managers for g in gws}
    opp_of = {m: ho.opponents_by_round("24-25", m) for m in inp.managers}
    hp = lambda x, y: 3 if x > y else (1 if x == y else 0)  # noqa: E731
    table = {m: sum(hp(real[(m, g)], real[(opp_of[m][g], g)]) for g in gws
                    if real[(m, g)] is not None and real[(opp_of[m][g], g)] is not None) for m in inp.managers}
    ranks = {}
    for team, d in data.items():
        others = [v for m, v in table.items() if m != team]
        for m in shown:
            s = table[team] if m == "real" else sum(hp(_pts(d, m)[g], d["opp"][g]) for g in gws)
            ranks.setdefault(m, []).append(1 + sum(v > s for v in others))
    order = ["real", "mvt_cal_h1_r50", "mvt_cal_h2_r50", "mvt_mix_h1_r50", "mvt_mix_h2_r50", "blind", "mvt_mix_r50",
             "mvt_cal_r50", "hold"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    rng = np.random.default_rng(0)
    for k, m in enumerate(order):
        r = np.array(ranks[m])
        col, _ = style(m)
        ax.scatter(k + rng.uniform(-0.18, 0.18, len(r)), r, s=18, color=col, alpha=0.75, edgecolor="white", lw=0.4)
        ax.plot([k - 0.3, k + 0.3], [r.mean()] * 2, color=INK, lw=1.6)
        ax.text(k, 21.6, f"{r.mean():.1f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(order)), [LABEL[m].replace("Calibrated MVT", "Cal. MVT") for m in order], rotation=25,
                  ha="right", fontsize=7.5)
    ax.set_ylim(22.4, 0.3)
    ax.set_yticks([1, 5, 10, 15, 20])
    ax.set_ylabel("League rank (of 20)")
    ax.set_title(f"League rank in place of the real manager, {n_t} teams (bar: mean, shown above)", fontsize=9.5)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "h2h_league_rank")
    print({LABEL[m]: round(float(np.mean(ranks[m])), 2) for m in order})


# ---------------------------------------------------------------------------- the H2H opponent model
def fig_opponents():
    import lineups

    # Figure 1: the line-up temperatures over the season, and the log-likelihood gain over a uniform choice
    inp = h2h.Inputs(h2h.Settings())
    lm = lineups.LineupModel(inp)
    gws = list(range(7, 39))
    fits = pd.DataFrame({gw: lm.fit(gw) for gw in gws}).T
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9, 3.2))
    ax.plot(fits.index, fits["tau_xi"], "-o", color=BLUE, lw=1.5, ms=4, mec="white", mew=0.8, label=r"$\tau_{\mathrm{XI}}$ (starting XI)")
    ax.plot(fits.index, fits["tau_cap"], "--s", color=ORANGE, lw=1.5, ms=4, mec="white", mew=0.8,
            label=r"$\tau_{\mathrm{cap}}$ (captain)")
    ax.set_xlabel("Gameweek of the decision")
    ax.set_ylabel("Temperature (points)")
    ax.set_ylim(0, None)
    ax.set_title("(a) Fitted temperatures", fontsize=9.5)
    ax.legend(loc="lower right", fontsize=7.5)
    bx.plot(fits.index, fits["ll_xi_per_obs"] - fits["ll_xi_uniform"], "-o", color=BLUE, lw=1.5, ms=4, mec="white",
            mew=0.8, label="XI (uniform: 1/550)")
    bx.plot(fits.index, fits["ll_cap_per_obs"] - fits["ll_cap_uniform"], "--s", color=ORANGE, lw=1.5, ms=4,
            mec="white", mew=0.8, label="Captain (uniform: 1/11)")
    bx.set_xlabel("Gameweek of the decision")
    bx.set_ylabel("Log-likelihood per line-up\nminus uniform's (nats)")
    bx.set_ylim(0, None)
    bx.set_title("(b) Fit against choosing at random", fontsize=9.5)
    bx.legend(loc="lower right", fontsize=7.5)
    fig.suptitle("The opponents' line-up model, 2024-25 (league managers' real XIs and captains before each gameweek)",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "opponent_tau")
    print(fits[["tau_xi", "tau_cap", "n_obs"]].describe().round(3).to_string())

    # Figure 2: PIT of the opponent's realised points under his mixture (moment-matched normal), weeks without chips
    data, _ = h2h_frames()
    rows = []
    for team, d in data.items():
        r = d["rows"]["mvt_mix_r50"].loc[d["weeks"]]
        r = r[r["opponent_points"].notna()]
        rows.append(pd.DataFrame({"team": team, "gw": r.index, "opp": r["opponent_points"].astype(float),
                                  "mu": r["mix_mu_G"], "sd": r["mix_sd_G"]}))
    o = pd.concat(rows, ignore_index=True)
    o["z"] = (o["opp"] - o["mu"]) / o["sd"]
    o["pit"] = stats.norm.cdf(o["z"])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9, 3.2))
    bins = np.linspace(0, 1, 11)
    ax.hist(o["pit"], bins=bins, color=ORANGE, edgecolor="white", linewidth=1, density=True)
    ax.axhline(1, color=INK, lw=1, ls="--", label="Uniform (a calibrated model)")
    ax.set_xlabel(r"PIT $= F_G$(opponent's realised points)")
    ax.set_ylabel("Density")
    ax.set_title(f"(a) PIT histogram ({len(o)} team-weeks)", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=7.5)
    gm = o.groupby("gw")["z"].mean()
    bx.axhline(0, color=GREY, lw=0.8)
    bx.bar(gm.index, gm.values, width=0.7, color=np.where(gm >= 0, ORANGE, BLUE), edgecolor="white", linewidth=0.5)
    bx.axhline(o["z"].mean(), color=INK, lw=1, ls="--", label=f"Season mean {o['z'].mean():+.2f}")
    bx.set_xlabel("Gameweek")
    bx.set_ylabel("Mean z of the opponents")
    bx.set_title(r"(b) $z = (\mathrm{realised} - \mu_G)/\sigma_G$, by gameweek", fontsize=9.5)
    bx.legend(loc="lower left", fontsize=7.5)
    fig.suptitle("Does the mixture predict the opponent's score? 2024-25", fontsize=10)
    fig.tight_layout()
    save(fig, "opponent_pit")
    ks = stats.kstest(o["pit"], "uniform")
    print(f"PIT: n {len(o)}, mean z {o['z'].mean():+.3f}, sd z {o['z'].std():.3f}, sd of GW means {gm.std():.3f}, "
          f"share above median {(o['pit'] > 0.5).mean():.3f}, KS p {ks.pvalue:.4f}, "
          f"tails <0.1 {(o['pit'] < 0.1).mean():.3f} >0.9 {(o['pit'] > 0.9).mean():.3f}")


def fig_opponent_squads():
    """4.3.2-4.3.3: the H2H squad model against simpler W_o (opponent_eval.py): expected overlap with the opponent's
    real squad, and the log score of his realised points, as paired differences from holding last week's squad."""
    rows = pickle.loads((OPT / "outputs" / "opponent_eval.pkl").read_bytes())
    recs = [{"opp": o, "gw": g, "personal": r["personal"], "variant": v, **r[v]}
            for (o, g), r in rows.items() for v in ("full", "independent", "average", "hold") if v in r]
    d = pd.DataFrame(recs)
    variants = [("full", "Full model (C-vine)", ORANGE, "o"), ("independent", "No copula (independent)", BLUE, "s"),
                ("average", "League-average fit", YELLOW, "D")]

    def paired(metric, v):
        w = d.pivot_table(index=["opp", "gw"], columns="variant", values=metric)
        x = (w[v] - w["hold"]).dropna()
        return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x)), len(x)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, metric, xlabel, title in [
        (axes[0], "overlap", r"Players of his real squad in $W_o$ − holding's", "(a) Squad overlap (of 15)"),
        (axes[1], "log_score", "Log score of his points − holding's", "(b) His realised points")]:
        ax.axvline(0, color=AQUA, lw=1.5, ls="--", label="Hold last week's squad")
        for k, (v, lab, col, mk) in enumerate(variants):
            m, ci, n = paired(metric, v)
            ax.errorbar(m, k, xerr=ci, fmt=mk, color=col, ms=7, mec="white", mew=1, capsize=0, lw=1.5)
            ax.annotate(f"{m:+.3f} (n = {n})", (m, k), xytext=(0, 9), textcoords="offset points", ha="center",
                        fontsize=7.5, color=MUTED)
        ax.set_yticks(range(len(variants)), [lab for _, lab, _, _ in variants])
        ax.set_ylim(len(variants) - 0.4, -0.7)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=9.5)
    axes[0].legend(loc="lower left", fontsize=7)
    axes[1].set_yticklabels([])
    fig.suptitle("The opponent's squad model against holding his squad, 2024-25 (paired, 95% CI)", fontsize=10)
    fig.tight_layout()
    save(fig, "opponent_squads")
    tab = d.groupby("variant")[["overlap", "p_exact", "buys_found", "n_buys_pred", "n_buys_real", "log_score",
                                "pit"]].mean().reindex(["full", "independent", "average", "hold"])
    tab["weeks"] = d.groupby("variant").size()
    print(tab.round(3).to_string())
    f = d[d["variant"] == "full"]
    print(f"C-vine with any dependent pair: {(f['n_dep_pairs'] > 0).mean():.3f} of {len(f)} weeks "
          f"(pairs: {f['n_dep_pairs'].value_counts().sort_index().to_dict()})")


# ---------------------------------------------------------------------------- the 2023-24 FantasyGo replication
FG_R = (0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0)
FG_R_STAR = 0.6  # the best points per contest against blind, the rule that picked r = 0.5 in H2H (in-sample)


def fantasygo_evaluations():
    """mvt.evaluate for each r of the factor-Sigma sweep (the 29 gameweeks with a usable contest)."""
    from dataclasses import replace

    s0 = mvt.Settings()
    inp = mvt.Inputs(s0)
    out = {}
    for r in FG_R:
        s = replace(s0, r_pct=r)
        res = pickle.loads((Path("../outputs/23-24/mvt") /
                            f"mvt_simulated_r{round(r * 100)}_history_22-23_nfix_factor_gurobi.pkl").read_bytes())
        if res["settings"].get("sigma") != "factor":
            raise ValueError(f"r = {r}: the cache is not the factor-Sigma run")
        e = mvt.evaluate(res["results"], inp, s)
        for lab in ("mvt", "base"):
            e[f"{lab}_pct"] = 1 - (e[f"{lab}_rank"] - 1) / e["entries"]  # share of the field at or below the team
        e["hit_cut"] = e["mvt_rank"] <= e["target_rank"]
        out[r] = e
    return out


# the FantasyGo methods compared in 4.5 (label, mvt.Settings changes, cache suffix); top-heavy = Algorithm 4
FG_VARIANTS = [
    ("MVT, r = 0.6 (sweep's best)", {"r_pct": 0.6}, ""),
    ("Top-heavy, one shift", {"r_pct": 0.96, "objective": "topheavy"}, "_uniformshift"),
    ("Top-heavy, per-place shifts", {"r_pct": 0.96, "objective": "topheavy"}, ""),
    ("+ shrunk $\\sigma_p$", {"r_pct": 0.96, "objective": "topheavy", "sigma_shrink": True}, ""),
    ("+ tracking ($k = 3$)", {"r_pct": 0.96, "objective": "topheavy", "track_k": 3}, ""),
]


def fantasygo_variant_evaluations():
    """{label: mvt.evaluate frame} for the FG_VARIANTS whose cache exists (a running one: the contests so far)."""
    from dataclasses import replace

    inp = mvt.Inputs(mvt.Settings())
    out = {}
    for label, kw, suffix in FG_VARIANTS:
        s = replace(mvt.Settings(), **kw)
        path = s.cache().with_name(s.cache().stem + suffix + ".pkl")
        if not path.exists():
            continue
        res = pickle.loads(path.read_bytes())["results"]
        e = mvt.evaluate(res, inp, s)
        e["pct_mvt"] = 1 - (e["mvt_rank"] - 1) / e["entries"]
        e["pct_base"] = 1 - (e["base_rank"] - 1) / e["entries"]
        out[label] = e
    return out


def fig_fantasygo_methods():
    """4.5: every FantasyGo method against blind: (a) points per contest, (b) mean percentile, (c) predicted P(top 10)
    against the realised share of top-10 finishes; and the table behind it (prizes net of the R20 fees)."""
    ev = fantasygo_variant_evaluations()
    labels = list(ev)
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    ax = axes[0]
    ax.axvline(0, color=BLUE, lw=1.5, label="Baseline")
    rows = []
    for k, lab in enumerate(labels):
        e = ev[lab]
        d = (e["mvt_points"] - e["base_points"]).astype(float)
        b = baseline.block_bootstrap([d], block=1)
        ax.plot([b["CI low"], b["CI high"]], [k, k], color=ORANGE, lw=1.5)
        ax.plot(d.mean(), k, "o", color=ORANGE, ms=6.5, mec="white", mew=1)
        ax.annotate(f"{d.mean():+.1f} (n = {len(e)})", (d.mean(), k), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=7, color=MUTED)
        top = (e["mvt_rank"] <= 10).astype(float)
        rows.append({"method": lab, "contests": len(e), "vs blind": d.mean(), "CI low": b["CI low"],
                     "CI high": b["CI high"], "p": b["p"], "percentile": e["pct_mvt"].mean(),
                     "blind percentile": e["pct_base"].mean(),
                     # top-heavy runs report P(beat the 10th-best score); the r-cut runs' prob is for their own cut
                     "P(top 10)": e["mvt_prob"].mean() if "top-heavy" in lab.lower() or lab.startswith("+") else np.nan,
                     "top-10 share": top.mean(), "top 10s": int(top.sum()), "in the money": int((e["mvt_prize"] > 0).sum()),
                     "prizes": e["mvt_prize"].sum(), "net": e["mvt_prize"].sum() - 20 * len(e),
                     "blind net": e["base_prize"].sum() - 20 * len(e)})
    ax.set_yticks(y, labels)
    ax.set_ylim(len(labels) - 0.4, -0.7)
    ax.set_xlabel("Method − baseline, points per contest (95% CI)")
    ax.set_title("(a) Points against the baseline", fontsize=9.5)
    ax.legend(loc="lower right", fontsize=7)
    t = pd.DataFrame(rows).set_index("method")
    bx = axes[1]
    for k, (pm, pb) in enumerate(zip(t["percentile"], t["blind percentile"])):
        bx.plot([pb, pm], [k, k], color=GREY, lw=1, zorder=1)
    bx.plot(t["blind percentile"], y, "o", ms=7, mfc="none", mec=BLUE, mew=1.6, label="Baseline", zorder=2)
    bx.plot(t["percentile"], y, "o", ms=6.5, color=ORANGE, mec="white", mew=1, label="Method", zorder=3)
    bx.legend(loc="upper left", fontsize=7, bbox_to_anchor=(0, -0.18), ncol=2)
    bx.set_yticks(y, [])
    bx.set_ylim(len(labels) - 0.4, -0.7)
    bx.set_xlabel("Mean percentile in the contest")
    bx.set_title("(b) Position in the field", fontsize=9.5)
    cx = axes[2]
    cx.barh(y - 0.14, t["P(top 10)"], height=0.28, color=ORANGE, edgecolor="white", label="Predicted P(top 10)")
    cx.barh(y + 0.14, t["top-10 share"], height=0.28, color=GREY, edgecolor="white", label="Realised share")
    cx.set_yticks(y, [])
    cx.set_ylim(len(labels) - 0.4, -0.7)
    cx.set_xlabel("Share of contests in the top 10")
    cx.set_title("(c) Reliability at the top", fontsize=9.5)
    cx.legend(loc="upper left", fontsize=7, bbox_to_anchor=(0, -0.18), ncol=2)
    fig.suptitle("FantasyGo 2023-24: the MVT methods against the baseline", fontsize=10.5)
    fig.tight_layout()
    save(fig, "fantasygo_methods")
    pd.set_option("display.width", 220)
    print(t.round(3).to_string())


def fig_fantasygo():
    ev = fantasygo_evaluations()
    xs = np.arange(len(FG_R))
    labels = [f"{r:g}" for r in FG_R]
    star = FG_R.index(FG_R_STAR)

    # Figure 1: the r sweep. (a) points per contest against blind, (b) percentile in the field, (c) P(beat) vs realised
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    ax = axes[0]
    ax.axhline(0, color=GREY, lw=0.8)
    for i, r in enumerate(FG_R):
        d = (ev[r]["mvt_points"] - ev[r]["base_points"]).astype(float)
        b = baseline.block_bootstrap([d], block=1)
        ax.plot([i, i], [b["CI low"], b["CI high"]], color=ORANGE, lw=1.5)
        ax.plot(i, d.mean(), "o", color=ORANGE, ms=7 if i == star else 5.5, mec="white", mew=1.2)
    ax.set_xticks(xs, labels)
    ax.set_xlabel("Benchmark quantile r")
    ax.set_ylabel("MVT − baseline, points per contest")
    ax.annotate(f"chosen r = {FG_R_STAR:g}", (star, ev[FG_R_STAR]["mvt_points"].sub(ev[FG_R_STAR]["base_points"]).mean()),
                xytext=(8, 22), textcoords="offset points", fontsize=7.5, color=MUTED,
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6})
    ax.set_title("(a) Points against the baseline (95% CI)", fontsize=9.5)

    ax = axes[1]
    base = ev[FG_R[0]]["base_pct"].mean()
    ax.axhline(base, color=BLUE, lw=1.5, label=f"Baseline ({base:.3f})")
    ax.plot(xs, [ev[r]["mvt_pct"].mean() for r in FG_R], "-o", color=ORANGE, lw=1.5, ms=5.5, mec="white", mew=1.2,
            label="MVT")
    ax.set_xticks(xs, labels)
    ax.set_xlabel("Benchmark quantile r")
    ax.set_ylabel("Mean percentile in the contest")
    ax.set_ylim(0.45, 0.62)
    ax.set_title("(b) Position in the field", fontsize=9.5)
    ax.legend(loc="lower left", fontsize=7.5)

    ax = axes[2]
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.8)
    pred = [ev[r]["mvt_prob"].mean() for r in FG_R]
    real = [ev[r]["hit_cut"].mean() for r in FG_R]
    ax.plot(pred, real, "-o", color=ORANGE, lw=1.5, ms=5.5, mec="white", mew=1.2)
    for r, x, y in zip(FG_R, pred, real):
        if r in (0.5, 0.7, 0.9, 1.0):
            off = {1.0: (2, 7), 0.9: (-42, 5)}.get(r, (4, -10))
            ax.annotate(f"r = {r:g}" + (" and 0.99" if r == 1.0 else ""), (x, y), xytext=off, textcoords="offset points",
                        fontsize=7.5, color=MUTED)
    ax.set_xlim(0, 0.85)
    ax.set_ylim(0, 0.85)
    ax.set_aspect("equal")
    ax.set_xlabel("Mean predicted P(beat the r-cut)")
    ax.set_ylabel("Share of contests above the r-cut")
    ax.set_title("(c) Reliability of P(beat)", fontsize=9.5)
    fig.suptitle("FantasyGo 2023-24: the MVT for each benchmark quantile r (29 contests)", fontsize=10.5)
    fig.tight_layout()
    save(fig, "fantasygo_r_sweep")

    # Figure 2: per contest, the main FantasyGo method (top-heavy, per-place shifts) against blind
    e = fantasygo_variant_evaluations()["Top-heavy, per-place shifts"]
    gws = list(e.index)
    pos = np.arange(len(gws))
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.5, 4.6), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
    ax.plot(pos, e["pct_base"], "-o", color=BLUE, lw=1.2, ms=5, mec="white", mew=1, label="Baseline")
    ax.plot(pos, e["pct_mvt"], "--s", color=ORANGE, lw=1.2, ms=5, mec="white", mew=1, label="Top-heavy MVT")
    for lab, col, k in (("base", BLUE, 0), ("mvt", ORANGE, 1)):
        top = e[f"{lab}_rank"] <= 10
        ax.scatter(pos[top.to_numpy()], np.full(top.sum(), 1.04), marker="v", s=40, color=col, zorder=4)
    ax.text(len(gws) - 0.4, 1.04, "top 10", fontsize=7, color=MUTED, va="center", ha="left")
    ax.set_ylim(0, 1.09)
    ax.set_ylabel("Percentile in the contest")
    ax.legend(loc="lower left", ncol=2, fontsize=7.5)
    ax.set_title("FantasyGo 2023-24 per contest: top-heavy MVT (per-place shifts) and the baseline", fontsize=10.5)
    d = e["mvt_points"] - e["base_points"]
    bx.axhline(0, color=GREY, lw=0.8)
    bx.bar(pos, d, width=0.7, color=np.where(d >= 0, ORANGE, BLUE), edgecolor="white", linewidth=0.5)
    bx.set_ylabel("MVT − baseline\n(points)")
    bx.set_xticks(pos, [str(g) for g in gws], fontsize=7)
    bx.set_xlabel("Gameweek (GW22-25 have no usable contest)")
    fig.tight_layout()
    save(fig, "fantasygo_contests")


def fig_field():
    """4.3.1: the simulated FantasyGo field (mvt.field_check at r*, 10 fields per gameweek) against the real one."""
    d = pickle.loads((Path("../outputs/23-24/mvt") / f"field_check_r{round(FG_R_STAR * 100)}.pkl").read_bytes())
    if d["settings"].get("sigma") != "factor" or d["settings"]["r_pct"] != FG_R_STAR:
        raise ValueError("field_check cache is not the factor-Sigma run at r*")
    df = pd.DataFrame([r for g in d["rows"] for r in d["rows"][g]])
    gen = {"shared_p": ("Simulated (used: one selection vector per field)", ORANGE, "s"),
           "per_team_p": ("Simulated (one selection vector per team)", BLUE, "^")}
    real = df[df["field"] == "real"].set_index("round")
    gws = list(real.index)
    pos = np.arange(len(gws))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
    ax.plot(pos, real["cut_points"], "-o", color=AQUA, lw=1.6, ms=5, mec="white", mew=1, label="Real field", zorder=3)
    for k, (f, (lab, col, mk)) in enumerate(gen.items()):
        g = df[df["field"] == f].groupby("round")["cut_points"]
        m, lo, hi = g.mean().reindex(gws), g.min().reindex(gws), g.max().reindex(gws)
        x = pos + (k - 0.5) * 0.25
        ax.vlines(x, lo, hi, color=col, lw=1, alpha=0.7)
        ax.plot(x, m, mk, color=col, ms=5.5, mec="white", mew=0.8, label=lab + " (mean, range of 10)")
    ax.set_xticks(pos, [str(g) for g in gws], fontsize=7)
    ax.set_xlabel("Gameweek (GW22-25 have no usable contest)")
    ax.set_ylabel(f"{FG_R_STAR:g}-quantile of the field's points")
    ax.set_title(f"(a) The field's r-cut (r = {FG_R_STAR:g}), scored on the realised points", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=7)
    cols = [f"top5_{p}" for p in POS]
    y = np.arange(len(POS))
    bx.plot(real[cols].mean().to_numpy(), y, "o", ms=11, mfc="none", mec=AQUA, mew=2, label="Real field", zorder=3)
    for f, (lab, col, mk) in gen.items():
        bx.plot(df.loc[df["field"] == f, cols].mean().to_numpy(), y, mk, color=col, ms=7, mec="white", mew=1,
                label=lab.split(" (")[0] + (" (used)" if f == "shared_p" else " (per team)"))
    bx.set_yticks(y, POS)
    bx.invert_yaxis()
    bx.set_xlim(0.3, 0.85)
    bx.set_xlabel("Share of the position's picks on its 5 most-picked players")
    bx.set_title("(b) How concentrated the picks are", fontsize=9.5)
    bx.legend(loc="upper left", fontsize=7)
    fig.suptitle("FantasyGo 2023-24: the simulated field against the real one", fontsize=10.5)
    fig.tight_layout()
    save(fig, "fantasygo_field")
    for f in gen:
        g = df[df["field"] == f].groupby("round").mean(numeric_only=True).reindex(gws)
        e = g["cut_points"] - real["cut_points"]
        print(f"{f}: cut bias {e.mean():+.2f}, MAE {e.abs().mean():.2f}, corr {np.corrcoef(g['cut_points'], real['cut_points'])[0, 1]:.3f}, "
              f"pick-share Spearman {g['share_corr'].mean():.3f}, cost {g['cost'].mean():.2f} vs real {real['cost'].mean():.2f}")
    prev = df[df["field"] == "previous"].set_index("round").reindex(gws)
    print(f"previous week's real field: cut MAE {(prev['cut_points'] - real['cut_points']).abs().mean():.2f}, "
          f"pick-share Spearman {prev['share_corr'].mean():.3f}")


# ---------------------------------------------------------------------------- worked gameweeks: the frontier
def h2h_frontier(gw, method="mvt_cal_r50"):
    """Replays the calibrated manager's gameweek for the own team (baseline.decide with 0 hits: the cached squad, free
    transfers and bank, the walk-forward shift cal_shift on G^(0.5)) and keeps both frontiers of Algorithm 3: the
    variance-cutting one over squads with mu_Y >= 0 and the variance-seeking one. Per lambda and sign: expected
    points, the sd of (our points - G) and Prob(beat) (Monte Carlo, as in the season)."""
    import lineups

    team = baseline.OWN_TEAM
    s = h2h.Settings(own_team_id=0)
    inp = h2h.Inputs(s)
    full = pickle.loads(baseline.cache_path(team).read_bytes())
    cache = full[method]
    row, bank = cache[gw], cache[gw - 1]["bank_after"]
    shift = baseline.cal_shift(full, gw, inp)
    pr = baseline.problem(gw, inp, s, row["squad_before"], bank)
    bm = lineups.benchmark(gw, inp.opponent_of[gw], pr, inp, s, lineups.LineupModel(inp))
    qb = lineups.quantile_benchmark(pr, bm, baseline.R_OF[method], seed=s.seed * 1000 + gw)
    qb = {**qb, "G": qb["G"] + shift, "mu_G": qb["mu_G"] + shift}
    pr["cov_g"] = qb["cov_g"]
    hold_v = h2h.lineup_vector(h2h.pick_lineup(sorted(row["squad_before"]), pr["mu_map"], inp.position), pr)
    ft = row["ft_before"]
    t_limit = min(ft, 15)
    zero = h2h.solve_gurobi(h2h.build_gurobi(pr, t_limit, 1, s), 0.0)

    def point(lam, sg, v):
        return {"lambda": lam, "sense": sg, "mean": float(pr["mu"] @ v),
                "sd": float(np.std(qb["delta"] @ v - qb["G"])), "prob_beat": lineups.prob_beat_mc(v, qb)}

    out = [point(0.0, 0, zero[1])]
    for sg in (1, -1):
        model = h2h.build_gurobi(pr, t_limit, sg, s, **({"min_mean": qb["mu_G"], "ft": ft, "hit_cost": baseline.HIT_COST}
                                                       if sg > 0 else {}))
        for lam in s.lambda_grid:
            if lam > 0:
                sol = h2h.solve_gurobi(model, lam)
                if sol is not None:
                    out.append(point(lam, sg, sol[1]))
    f = pd.DataFrame(out)
    hold = {"mean": float(pr["mu"] @ hold_v), "sd": float(np.std(qb["delta"] @ hold_v - qb["G"])),
            "prob_beat": lineups.prob_beat_mc(hold_v, qb)}
    b = f.loc[f["prob_beat"].idxmax()]
    print(f"H2H GW{gw}: shift {shift:+.2f}, mu_G {qb['mu_G']:.1f}; cached lambda* {row['lambda_star']} sense "
          f"{row['sense']}, P {row['prob']:.3f}; replay best lambda {b['lambda']} sense {b['sense']:+.0f}, "
          f"P {b['prob_beat']:.3f}; hold P {hold['prob_beat']:.3f}")
    return f, hold, float(qb["mu_G"]), shift


def fig_frontiers(fg_gw=7, h2h_gws=(10, 20)):
    """4.4.1: worked gameweeks. FantasyGo: the top-heavy frontier (Algorithm 4, per-place shifts) with its expected
    prize; H2H: the calibrated manager's two frontiers (Algorithm 3) with P(beat)."""
    from dataclasses import replace

    s_fg = replace(mvt.Settings(), r_pct=0.96, objective="topheavy")
    res = pickle.loads(s_fg.cache().read_bytes())["results"][fg_gw]
    fig, axes = plt.subplots(2, 1 + len(h2h_gws), figsize=(3.6 * (1 + len(h2h_gws)), 6.4))
    # FantasyGo column
    f = res["frontier"].reset_index(drop=True)
    best = f["exp_prize"].idxmax()
    ax, bx = axes[0, 0], axes[1, 0]
    ax.plot(f["sd"], f["mean"], "-o", color=ORANGE, lw=1.2, ms=5, mec="white", mew=0.8, label="Variance-seeking frontier")
    ax.scatter(f.loc[0, "sd"], f.loc[0, "mean"], s=120, facecolor="none", edgecolor=BLUE, linewidth=1.6, zorder=3,
               label="λ = 0 (baseline)")
    ax.scatter(f.loc[best, "sd"], f.loc[best, "mean"], marker="*", s=180, color=ORANGE, edgecolor=INK, linewidth=0.6,
               zorder=4, label=f"λ* = {f.loc[best, 'lambda']:.2g}")
    ax.set_title(f"FantasyGo GW{fg_gw}: top-heavy (Algorithm 4)", fontsize=9)
    ax.set_xlabel("sd of our points")
    ax.set_ylabel("Expected points")
    ax.legend(loc="best", fontsize=7)
    lam = f["lambda"].where(f["lambda"] > 0, f.loc[f["lambda"] > 0, "lambda"].min() / 3)
    bx.plot(lam, f["exp_prize"], "-o", color=ORANGE, lw=1.5, ms=5, mec="white", mew=1)
    bx.plot(lam.iloc[0], f.loc[0, "exp_prize"], "o", ms=9, mfc="none", mec=BLUE, mew=1.6)
    bx.set_xscale("log")
    bx.set_xticks(lam.iloc[::2], ["0"] + [f"{x:.2g}" for x in f["lambda"].iloc[::2][1:]], fontsize=7)
    bx.minorticks_off()
    bx.set_xlabel("λ")
    bx.set_ylabel("Expected prize (R, model)")
    # H2H columns
    for k, gw in enumerate(h2h_gws, start=1):
        f, hold, mu_g, shift = h2h_frontier(gw)
        ax, bx = axes[0, k], axes[1, k]
        best = f["prob_beat"].idxmax()
        for sg, lab, col, mk in ((1, r"Cut variance ($\mu_Y \geq 0$)", BLUE, "s"), (-1, "Seek variance", ORANGE, "o")):
            g = pd.concat([f[f["sense"] == 0], f[f["sense"] == sg]])
            ax.plot(g["sd"], g["mean"], "-" + mk, color=col, lw=1.2, ms=5, mec="white", mew=0.8, label=lab)
            lam = g["lambda"].where(g["lambda"] > 0, f.loc[f["lambda"] > 0, "lambda"].min() / 3)
            bx.plot(lam, g["prob_beat"], "-" + mk, color=col, lw=1.5, ms=5, mec="white", mew=1, label=lab)
        z = f[f["sense"] == 0].iloc[0]
        ax.scatter(z["sd"], z["mean"], s=120, facecolor="none", edgecolor=INK, linewidth=1.4, zorder=3, label="λ = 0 (baseline)")
        ax.scatter(f.loc[best, "sd"], f.loc[best, "mean"], marker="*", s=180, color=YELLOW, edgecolor=INK,
                   linewidth=0.6, zorder=4, label=f"Played: λ* = {f.loc[best, 'lambda']:.2g}")
        ax.scatter(hold["sd"], hold["mean"], marker="x", s=50, color=AQUA, linewidth=1.8, zorder=4, label="Hold the squad")
        ax.axhline(mu_g, color=GREY, lw=0.8, ls=":")
        ax.text(ax.get_xlim()[0], mu_g, r" $\mu_G$", fontsize=7, color=MUTED, va="bottom")
        if not (f["sense"] == 1).any():
            ax.text(0.97, 0.97, r"no squad reaches $\mu_G$:" + "\nthe cutting frontier is empty", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7, color=MUTED)
        ax.set_title(f"H2H GW{gw}: calibrated MVT (shift {shift:+.1f})", fontsize=9)
        ax.set_xlabel("sd of (our points − G)")
        ax.legend(loc="best", fontsize=6.5)
        bx.axhline(hold["prob_beat"], color=AQUA, lw=1.2, ls="--")
        bx.set_xscale("log")
        lams = sorted(f.loc[f["lambda"] > 0, "lambda"].unique())
        x0 = lams[0] / 3
        ticks = [x0] + lams[::2]
        bx.set_xticks(ticks, ["0"] + [f"{x:.2g}" for x in lams[::2]], fontsize=7)
        bx.minorticks_off()
        bx.set_xlabel("λ")
        bx.set_ylabel("P(beat the benchmark)")
    fig.suptitle("Worked gameweeks: the frontiers and what is played", fontsize=10.5)
    fig.tight_layout()
    save(fig, "worked_frontiers")


BETA_DIR = HERE.parent / "opponent_model/beta estimators"
TRACE_COLORS = [BLUE, ORANGE, AQUA, YELLOW, GREY, INK]
CHAIN_LS = ["-", "--", ":", "-."]


def _density_and_trace(ax_d, ax_t, draws, labels=None):
    """draws: (chain, draw, k). Per-chain KDE of each coefficient (line style = chain) and its trace."""
    draws = np.asarray(draws)
    if draws.ndim == 2:
        draws = draws[..., None]
    for j in range(draws.shape[2]):
        c = TRACE_COLORS[j % len(TRACE_COLORS)]
        lo, hi = np.percentile(draws[..., j], [0.1, 99.9])
        pad = 0.15 * (hi - lo)
        grid = np.linspace(lo - pad, hi + pad, 200)
        for ch in range(draws.shape[0]):
            ax_d.plot(grid, stats.gaussian_kde(draws[ch, :, j])(grid), color=c, ls=CHAIN_LS[ch % 4], lw=0.9,
                      label=(labels[j] if labels is not None and ch == 0 else None))
            if ax_t is not None:
                ax_t.plot(draws[ch, :, j], color=c, lw=0.4, alpha=0.6)
    ax_d.set_yticks([])
    if ax_t is not None:
        ax_t.set_xlabel("draw")


def _legend_on_top(fig, ax):
    """The feature names in one row (or two) above the panels, so the panels take the full width."""
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=min(len(l), 3), fontsize=7.5,
               handlelength=2.0, columnspacing=1.6)


def fig_traces(gw=38):
    """Posterior densities and MCMC traces of the H2H selection models at one gameweek (4.4), one figure per
    position: the league baseline beta (s2_league_traces_<pos>) and the manager update (delta beta_m, gamma_m) of
    that week's opponent (s2_manager_traces_<pos>)."""
    import joblib
    names = {"GK": "Goalkeeper", "DEF": "Defender", "MID": "Midfielder", "FWD": "Forward"}
    man = joblib.load(BETA_DIR / f"opponent betas/estimates/manager_fits_{gw}.joblib")[gw]
    for p in POS:
        f = joblib.load(BETA_DIR / f"league betas/league_models_{p.lower()}")[gw]
        fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.0))
        _density_and_trace(ax[0], ax[1], f["trace_final"].posterior["beta"].values, f["feats"])
        ax[0].set_title("$\\beta^{league}$ posterior", loc="left")
        ax[1].set_title("trace (4 chains)", loc="left")
        fig.tight_layout()
        _legend_on_top(fig, ax[0])
        save(fig, f"s2_league_traces_{p.lower()}")

        m = man[names[p]]
        post = m["trace"].posterior
        fig, ax = plt.subplots(2, 2, figsize=(6.3, 3.8))
        _density_and_trace(ax[0, 0], ax[0, 1], post["delta_beta"].values, m["feats"])
        _density_and_trace(ax[1, 0], ax[1, 1], post["gamma_m"].values)
        ax[0, 0].set_title("$\\Delta\\beta_m$ posterior", loc="left")
        ax[0, 1].set_title("$\\Delta\\beta_m$ trace (4 chains)", loc="left")
        ax[1, 0].set_title("$\\gamma_m$ posterior", loc="left")
        ax[1, 1].set_title("$\\gamma_m$ trace (4 chains)", loc="left")
        ax[0, 1].set_xlabel("")
        fig.tight_layout()
        _legend_on_top(fig, ax[0, 0])
        save(fig, f"s2_manager_traces_{p.lower()}")
        print(p, "opponent", m["team_id"], "weeks", m["n_weeks"])


COEF_STYLES = [dict(color="black", marker="o"), dict(color="0.45", marker="s"),
               dict(color="0.25", marker="^"), dict(color="0.6", marker="D")]


def fig_s1_coefs(min_share=0.3, max_features=4):
    """The FantasyGo selection model's coefficients by target gameweek (4.3.1): posterior mean and 94% HDI of the
    features kept in at least `min_share` of the fits (outputs/23-24/selection/league_models.pkl, as mvt.py uses);
    the same layout as thesis/make_results.py's s2_league_coefs."""
    import joblib
    import config
    fits = joblib.load(config.paths("23-24").selection_dir + "/league_models.pkl")
    names = {"GK": "Goalkeeper", "DEF": "Defender", "MID": "Midfielder", "FWD": "Forward"}
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.6))
    for a, p in zip(axes.flat, POS):
        rows = [dict(rnd=r, feat=ft, mean=f["summary_final"].loc[ft, "mean"], lo=f["summary_final"].loc[ft, "hdi_3%"],
                     hi=f["summary_final"].loc[ft, "hdi_97%"]) for r, f in fits[p].items() for ft in f["feats"]]
        x = pd.DataFrame(rows)
        freq = x["feat"].value_counts()
        keep = freq[freq >= min_share * x["rnd"].nunique()].index[:max_features]
        for k, ft in enumerate(keep):
            y = x[x.feat == ft].sort_values("rnd")
            off = (k - (len(keep) - 1) / 2) * 0.15
            a.errorbar(y["rnd"] + off, y["mean"], yerr=[y["mean"] - y["lo"], y["hi"] - y["mean"]], linestyle="none",
                       markersize=3, elinewidth=0.6, capsize=0, label=ft, **COEF_STYLES[k])
        a.axhline(0, color="0.3", linewidth=0.6)
        a.set_title(names[p])
        a.set_xlabel("target gameweek")
        a.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    for a in axes[:, 0]:
        a.set_ylabel("coefficient (standardised)")
    fig.tight_layout()
    save(fig, "s1_selection_coefs")


def main():
    global PNG_DIR
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("figures", nargs="+", choices=["sigma", "h2h", "fantasygo", "opponents", "frontiers", "field", "sigmap", "squads", "fgmethods", "traces", "s1coefs"])
    parser.add_argument("--png-dir", help="also write a PNG of each figure here")
    args = parser.parse_args()
    PNG_DIR = args.png_dir
    h2h._ENV = h2h.gp.Env(params={"OutputFlag": 0, "Threads": 2})  # the cores are shared
    for f in args.figures:
        {"sigma": fig_sigma, "h2h": fig_h2h, "fantasygo": fig_fantasygo, "opponents": fig_opponents, "frontiers": fig_frontiers, "field": fig_field, "sigmap": fig_sigma_p, "squads": fig_opponent_squads, "fgmethods": fig_fantasygo_methods, "traces": fig_traces, "s1coefs": fig_s1_coefs}[f]()


if __name__ == "__main__":
    main()
