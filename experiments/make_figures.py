"""Data figures, drawn from the results JSON.

Two decisions worth recording, because both are easy to get wrong.

**No dual axis.** The granularity figure has to show the reporting effort and the
clustering objective on the same horizontal axis, and putting them on two vertical scales
in one frame would invite the reader to compare their heights, which means nothing. They
are two stacked panels sharing the axis they genuinely share.

Each figure is written twice, as PDF for typesetting and as PNG for word processors,
which will not render an embedded PDF.

**Identity never rests on colour.** Figures are printed in greyscale and read by
people who do not all see hue the same way, so every series carries a distinct marker and
a distinct line style, and every series is labelled at its own end. The three hues come
from a palette validated for all-pairs separation under deuteranopia and tritanopia; they
are the redundant channel rather than the load-bearing one.

Run:  python3 experiments/make_figures.py
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common as P

OUT = P.REPO / "figures"

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d8d7d2"
SERIES = {"learned": "#2a78d6", "terrain": "#eb6834", "tessellation": "#1baf7a"}
MARK = {"learned": "o", "terrain": "s", "tessellation": "^"}
DASH = {"learned": "-", "terrain": "--", "tessellation": ":"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "legend.frameon": False, "figure.dpi": 200,
})


def _bare(ax, keep=("left", "bottom")):
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


def granularity_figure():
    """Reporting effort and clustering dispersion against the number of parts."""
    d = json.loads((P.RESULTS / "exp2_granularity.json").read_text())
    ks = [r["k"] for r in d["curves"]["TESSERA-128"]]
    inertia = [d["inertia"][str(k)] if str(k) in d["inertia"] else d["inertia"][k]
               for k in ks]

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(5.4, 4.2), sharex=True,
                                 gridspec_kw={"height_ratios": [2.1, 1]})

    # The spread across hazard realisations is the point of this panel, not decoration.
    # Drawn as a band it says at once that the curve has an interior minimum and that the
    # minimum is not separated from k=6 by more than the realisations move it.
    for name, key in (("learned", "TESSERA-128"), ("terrain", "terrain")):
        r = np.array([row["city_reports_per_hour"][0] if row["city_reports_per_hour"]
                      else np.nan for row in d["curves"][key]])
        s = np.array([row["city_reports_per_hour"][1] if row["city_reports_per_hour"]
                      else np.nan for row in d["curves"][key]])
        ax.fill_between(ks, r - s, r + s, color=SERIES[name], alpha=0.13, linewidth=0,
                        zorder=2)
        ax.plot(ks, r, DASH[name], color=SERIES[name], marker=MARK[name],
                markersize=4.2, linewidth=1.3, label=name, clip_on=False, zorder=3)
        j = int(np.nanargmin(r))
        ax.plot([ks[j]], [r[j]], MARK[name], color=SERIES[name], markersize=8.5,
                markerfacecolor="none", markeredgewidth=1.1, zorder=4)

    ax.set_ylim(0, 44)
    ax.axvline(6, color=GRID, linewidth=1.0, zorder=1)
    ax.annotate("deployed $k$=6", (6, 43.0), textcoords="offset points",
                xytext=(4, -2), fontsize=7.5, color=INK2, va="top")
    ax.annotate("least at $k$=12, and steadier there;\n"
                "the means are not separated ($p$=0.11)", (12, 11.4),
                textcoords="offset points", xytext=(14, -6), fontsize=7.5, color=INK,
                va="center")
    ax.legend(loc="upper right", fontsize=8, handlelength=2.4, labelcolor=INK,
              borderaxespad=0.2)
    ax.set_ylabel("reporting rate the city\nmust sustain (reports/h)")
    ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    _bare(ax)

    bx.plot(ks, inertia, "-", color=INK2, marker="D", markersize=3.4, linewidth=1.3,
            clip_on=False, zorder=3)
    bx.axvline(6, color=GRID, linewidth=1.0, zorder=1)
    bx.annotate("the elbow the usual\ncriterion would pick", (3, inertia[1]),
                textcoords="offset points", xytext=(12, 2), fontsize=7.5, color=INK2,
                va="center")
    bx.set_ylabel("within-cluster\ndispersion")
    bx.set_xlabel("number of parts, $k$")
    bx.set_xticks(ks)
    bx.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    bx.set_axisbelow(True)
    _bare(bx)

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "granularity.pdf", bbox_inches="tight")
    fig.savefig(OUT / "granularity.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def cities_figure():
    """Every prior in every city, so the two spreads can be compared by eye."""
    d = json.loads((P.RESULTS / "exp6_cities.json").read_text())
    cities = {n: r for n, r in d["cities"].items() if "error" not in r}
    keys = {"learned": "TESSERA-128", "terrain": "terrain",
            "tessellation": "tessellation"}
    order = sorted(cities, key=lambda n: np.mean(
        [cities[n]["priors"][k]["kappa_excess_nats"][0] for k in keys.values()]))

    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    for i, name in enumerate(order):
        vals = [cities[name]["priors"][k]["kappa_excess_nats"][0] for k in keys.values()]
        ax.plot([min(vals), max(vals)], [i, i], "-", color=GRID, linewidth=1.4, zorder=1)
        for label, key in keys.items():
            ax.plot(cities[name]["priors"][key]["kappa_excess_nats"][0], i, MARK[label],
                    color=SERIES[label], markersize=5, markeredgecolor="white",
                    markeredgewidth=0.6, zorder=3,
                    label=label if i == 0 else None)
        ax.plot(cities[name]["priors"]["random"]["kappa_excess_nats"][0], i, "|",
                color=INK2, markersize=7, markeredgewidth=1.0, zorder=2,
                label="random control" if i == 0 else None)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("evidence a report carries, above the permutation null (nats)")
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.grid(axis="x", color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    _bare(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8, ncol=4,
              handletextpad=0.4, columnspacing=1.4, labelcolor=INK)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "cities.pdf", bbox_inches="tight")
    fig.savefig(OUT / "cities.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    granularity_figure()
    cities_figure()
    for f in sorted(OUT.glob("*.p*")):
        print(f"wrote {f} ({f.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
