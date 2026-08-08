"""Two-way decomposition of the city by prior design.

Each settlement is scored under each candidate prior on identical hazard realisations, so
the design is complete and balanced and three terms separate: how much the settlement
moves the measurement with the prior averaged out, how much the prior moves it with the
settlement averaged out, and how much remains in the interaction, which is the part that
says a prior suited to one settlement need not suit another.

Realisation noise propagates into every cell mean at sd / sqrt(n), and a main effect
averages it down further, so noise-corrected components are reported alongside the raw
ones. The interaction is tested by parametric bootstrap under the additive model, using
each cell's own measured spread as its standard error. Permuting priors within a
settlement is not used, since it leaves each row's spread intact and therefore has almost
no power against this alternative.

Arithmetic on finished runs. Reads results/exp6_cities.json, writes
results/exp12_decomposition.json.
"""
from __future__ import annotations

import itertools
import json

import numpy as np

import common as P

PRIORS = ["TESSERA-128", "terrain", "tessellation"]
LABEL = {"TESSERA-128": "learned", "terrain": "terrain", "tessellation": "tessellation"}


def main() -> None:
    src_path = P.RESULTS / "exp6_cities.json"
    if not src_path.exists():
        raise SystemExit("run exp6_cities.py first; this reads its results")
    src = json.load(open(src_path))["cities"]
    cities = list(src)
    X = np.array([[src[c]["priors"][p]["kappa_excess_nats"][0] for p in PRIORS]
                  for c in cities])                      # cities x priors
    S = np.array([[src[c]["priors"][p]["kappa_excess_nats"][1] for p in PRIORS]
                  for c in cities])                      # sd across 8 realisations
    n_real = 8

    grand = X.mean()
    city_eff = X.mean(1) - grand
    prior_eff = X.mean(0) - grand
    inter = X - grand - city_eff[:, None] - prior_eff[None, :]

    # Realisation noise propagates into every cell mean at sd/sqrt(n), and a main effect
    # averages that down further. Subtracting it is what separates a real spread from the
    # spread a finite number of draws produces on its own.
    cell_se2 = float((S ** 2).mean()) / n_real
    var_city_raw = float(city_eff.var())
    var_prior_raw = float(prior_eff.var())
    var_inter_raw = float(inter.var())
    out = {
        "n_cities": len(cities), "n_priors": len(PRIORS), "n_realisations": n_real,
        "grand_mean_nats": round(float(grand), 4),
        "city_effects": {c: round(float(v), 4) for c, v in zip(cities, city_eff)},
        "prior_effects": {LABEL[p]: round(float(v), 4) for p, v in zip(PRIORS, prior_eff)},
        "spans_nats": {
            "city_means": round(float(np.ptp(X.mean(1))), 4),
            "prior_means": round(float(np.ptp(X.mean(0))), 4),
            "whole_table": round(float(np.ptp(X)), 4),
            "mean_within_city_across_prior_range": round(float(np.ptp(X, 1).mean()), 4),
            "mean_within_prior_across_city_range": round(float(np.ptp(X, 0).mean()), 4)},
        "variance_components_nats2": {
            "city": round(var_city_raw, 5), "prior": round(var_prior_raw, 5),
            "interaction": round(var_inter_raw, 5),
            "realisation_noise_per_cell": round(cell_se2, 5),
            "city_noise_corrected": round(max(0.0, var_city_raw - cell_se2 / 3), 5),
            "prior_noise_corrected": round(max(0.0, var_prior_raw - cell_se2 / 8), 5),
            "interaction_noise_corrected": round(max(0.0, var_inter_raw - cell_se2), 5)},
        "ratio_city_to_prior_main_effect_sd":
            round(float(np.sqrt(var_city_raw / var_prior_raw)), 2),
    }

    # A paired sign-rank on every pair of priors across cities, which is the statement
    # is defensible about the prior main effect.
    from scipy import stats
    out["paired_wilcoxon_p"] = {
        f"{LABEL[a]} vs {LABEL[b]}":
            round(float(stats.wilcoxon(X[:, PRIORS.index(a)], X[:, PRIORS.index(b)]).pvalue), 4)
        for a, b in itertools.combinations(PRIORS, 2)}

    # Is the interaction bigger than the noise? Permuting priors within a city cannot
    # answer that, because it leaves each row's spread intact and so leaves the
    # interaction almost unchanged; the test has no power and is not run. What does
    # answer it is a parametric bootstrap under the additive model, using each cell's
    # own measured spread across realisations as its standard error.
    rng = np.random.default_rng(0)
    add = grand + city_eff[:, None] + prior_eff[None, :]
    se = S / np.sqrt(n_real)
    null = np.empty(20000)
    for b in range(null.size):
        Y = add + rng.standard_normal(X.shape) * se
        g = Y.mean()
        null[b] = float((Y - g - (Y.mean(1) - g)[:, None] - (Y.mean(0) - g)[None, :]).var())
    out["interaction_bootstrap"] = {
        "observed_var": round(var_inter_raw, 5),
        "null_median_var": round(float(np.median(null)), 5),
        "p": round(float((null >= var_inter_raw).mean()), 4)}

    (P.RESULTS / "exp12_decomposition.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
