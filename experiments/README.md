# Experiments

Run from this directory. Each script writes its results to `results/` as JSON, alongside
the configuration that produced them. Those files are committed, so the results of the runs
behind this code are present before anything is rerun.

```bash
python3 run_all.py               # everything, in order, about an hour

python3 exp1_priors.py           # evidence per report under each candidate prior
python3 exp2_granularity.py      # required reporting rate against granularity
python3 exp3_composition.py      # sensitivity to how hazard variance divides
python3 exp4_metastability.py    # critical interval, periodic and Poisson arrivals
python3 exp5_recruitment.py      # solicitation policy and time to a decision
python3 exp6_cities.py           # the same measurement across eight settlements
python3 exp7_methods.py          # sensitivity to the partitioning algorithm
python3 exp8_spectral.py         # a hand-built optical composite as a further prior
python3 exp9_stratify.py         # per-part generalisation against ground character
python3 exp10_heldout_k.py       # granularity selection on a split
python3 exp11_sar_detectability.py  # whether radar registers a documented flood
python3 exp12_decomposition.py   # two-way decomposition of settlement by prior
python3 exp13_alpha_margin.py    # sensitivity of the effort curve to the alpha cap

python3 make_figures.py          # the data figures
python3 make_figure1.py          # the schematic, as a TikZ fragment
```

## Network

`exp6`, `exp8`, `exp9` and `exp11` fetch imagery and need a certificate bundle:

```bash
export SSL_CERT_FILE=$(python3 -c "import certifi;print(certifi.where())")
```

`exp6` writes after each settlement, so it is safe to interrupt and resume. A full run
caches roughly 1.3 GB of tiles and rasters under this directory, all gitignored and safe
to delete.

## Ordering

`exp1` fetches and caches the elevation tile for the pilot settlement, so running it first
lets `exp2` to `exp5` run offline. `exp12` reads `exp6`'s results and `exp13` reads
`exp2`'s, so both must follow them.

## Shared machinery

`common.py` holds what the experiments have in common: the exogenous hazard fields, the
candidate partitions, the leave-one-out estimator with its permutation control, and the
context managers that install a calibration or an alpha margin for the duration of a
block. `terrain.py` derives elevation, slope and a wetness index on the tile grid.
`flood.py` reads the radar archive. `_tile.py` loads and dequantises an embedding tile.

## A note on the estimator

The majority rule makes the effective sample size of the confusion matrix closer to the
number of parts than to the number of reports, so estimates vary substantially between
hazard realisations even with thousands of marks. Standard deviations across realisations
are reported throughout and they are large. Comparisons between conditions are paired on
the realisation and tested that way.
