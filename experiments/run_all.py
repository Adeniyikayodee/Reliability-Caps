"""Run every experiment in order.

Order matters only in that exp1 fetches and caches the elevation tile for the pilot
settlement, so running it first means exp2 to exp5 need no network at all. exp6 fetches a
tile per settlement and needs network throughout, so it runs late and writes its results
after each city, which makes it safe to interrupt and resume. exp12 and exp13 are
arithmetic on finished runs and must follow exp6 and exp2 respectively.

Roughly an hour end to end. exp6, exp8, exp9 and exp11 require network.
"""
from __future__ import annotations

import runpy
import sys
import time

STEPS = ["exp1_priors", "exp2_granularity", "exp3_composition",
         "exp4_metastability", "exp5_recruitment", "exp6_cities",
         "exp7_methods", "exp8_spectral", "exp9_stratify", "exp10_heldout_k",
         "exp11_sar_detectability",
         # These two read the results of exp6 and exp2, so they come after both.
         "exp12_decomposition", "exp13_alpha_margin"]


def main():
    wanted = sys.argv[1:] or STEPS
    t0 = time.time()
    for name in wanted:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}", flush=True)
        started = time.time()
        runpy.run_module(name, run_name="__main__")
        print(f"[{name} finished in {time.time() - started:.0f}s]", flush=True)
    print(f"\nall done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
