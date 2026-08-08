"""Granularity selection on a split.

exp2 finds the granularity that minimises the reporting effort a settlement must sustain,
and finds it on the same hazard realisations it then reports the effort for. That is
selection on the evaluation data, and the optimism it carries is not knowable from the
in-sample figure alone.

The realisations are therefore split. The first half selects the granularity, the second
half is not consulted until the selection is fixed, and the reported figure is the effort
at the selected granularity measured on the half that had no say in it. The in-sample
figure is printed beside it so the optimism is visible, and the count of usable
realisations is recorded per granularity, since these differ.

Both prior families are run, since a selection rule that transfers for one need not
transfer for the other.

Writes results/exp10_heldout_k.json.
"""
from __future__ import annotations

import json

import numpy as np

import common as P
import terrain as T

KS = (4, 5, 6, 8, 10, 12, 16, 20)
SELECT_SEEDS = tuple(range(10))
EVAL_SEEDS = tuple(range(10, 20))
DEPLOYED_K = 6
N_REPORTS = 3000
NOISE = 0.20
TAU = 30.0
TRIALS = 100


def effort(part, reports_list):
    """Mean city reporting rate over a set of realisations, and how many were usable."""
    vals = []
    for reports in reports_list:
        m = P.measure_kappa(part.labels, reports, shuffle_trials=TRIALS, seed=7)
        if m.get("usable") and (m.get("p_permutation") or 1) < 0.05:
            vals.append(P.city_effort(part.areas(), m["tpr"], m["fpr"],
                                      tau_min=TAU)["city_reports_per_hour"])
    return (float(np.mean(vals)) if vals else None), len(vals)


def main():
    grid = P.load_tile()
    land = P.land_mask(grid)
    meta = json.loads((P.REPO / "data/meta.json").read_text())
    elev = T.elevation_on_grid(meta, land.shape)
    stack = T.terrain_stack(elev, land)
    wet = T.wetness_index(elev, land)

    def marks(seeds):
        return [P.sample_reports(P.flood_field(grid, land, wet, seed=s), land,
                                 n=N_REPORTS, noise=NOISE, seed=s) for s in seeds]

    select, evaluate = marks(SELECT_SEEDS), marks(EVAL_SEEDS)
    out = {"config": {"ks": list(KS), "select_seeds": list(SELECT_SEEDS),
                      "eval_seeds": list(EVAL_SEEDS), "deployed_k": DEPLOYED_K,
                      "n_reports": N_REPORTS, "report_noise": NOISE, "tau_min": TAU},
           "families": {}}

    for family, build in (("learned", lambda k: P.tessera_partition(grid, k, seed=0)),
                          ("terrain", lambda k: P.terrain_partition(stack, land, k, seed=0))):
        curve = {}
        for k in KS:
            part = build(k)
            s_val, s_n = effort(part, select)
            e_val, e_n = effort(part, evaluate)
            curve[k] = {"select": s_val, "select_usable": s_n,
                        "eval": e_val, "eval_usable": e_n}
            print(f"  {family:<8} k={k:<3} select {s_val if s_val is None else round(s_val,2)}"
                  f"   eval {e_val if e_val is None else round(e_val,2)}", flush=True)

        usable = {k: v for k, v in curve.items() if v["select"] is not None}
        k_star = min(usable, key=lambda k: usable[k]["select"])
        in_sample = {k: v for k, v in curve.items() if v["eval"] is not None}
        k_naive = min(in_sample, key=lambda k: in_sample[k]["eval"])
        out["families"][family] = {
            "curve": {str(k): v for k, v in curve.items()},
            "k_selected_on_first_half": k_star,
            "effort_at_selected_k_on_heldout": round(curve[k_star]["eval"], 3)
            if curve[k_star]["eval"] else None,
            "effort_at_selected_k_in_sample": round(curve[k_star]["select"], 3),
            "effort_at_deployed_k_on_heldout": round(curve[DEPLOYED_K]["eval"], 3)
            if curve[DEPLOYED_K]["eval"] else None,
            "best_possible_k_on_heldout": k_naive,
            "effort_at_best_possible_on_heldout": round(in_sample[k_naive]["eval"], 3)}

    P.RESULTS.mkdir(parents=True, exist_ok=True)
    (P.RESULTS / "exp10_heldout_k.json").write_text(json.dumps(out, indent=2) + "\n")

    print()
    for family, r in out["families"].items():
        held, dep = r["effort_at_selected_k_on_heldout"], r["effort_at_deployed_k_on_heldout"]
        gain = None if not (held and dep) else 100 * (dep - held) / dep
        print(f"{family}: k chosen on the first half = {r['k_selected_on_first_half']}")
        print(f"  on held-out realisations it needs {held} reports/h, "
              f"against {dep} at the deployed k={DEPLOYED_K}"
              + (f", a saving of {gain:.0f}%" if gain else ""))
        print(f"  in sample the same k reads {r['effort_at_selected_k_in_sample']}, "
              f"so the optimism is "
              f"{r['effort_at_selected_k_in_sample'] - (held or 0):+.2f} reports/h")
    print(f"\n-> {P.RESULTS / 'exp10_heldout_k.json'}")


if __name__ == "__main__":
    main()
