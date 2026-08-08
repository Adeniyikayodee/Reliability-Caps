# Reliability Caps

Research code for measuring what a single human observation is worth to a spatial
decision, given a learned partition of the landscape.

## The quantity

A settlement is partitioned into parts, each a set of ten-metre cells, obtained by
clustering a satellite representation. The question to be decided is whether a part is in
hazard. That is never observed directly. What is observed is reports from residents
standing in individual cells.

A report is about a cell and the decision is about a part, so two stages of uncertainty
compose. Writing TPR and FPR for the probability that a cell is wet given that its part
is or is not in hazard, and s and f for the probability that a resident in a wet or dry
cell files a report in a given band, the evidence one report carries is

    Lambda(b) = log( (TPR*s + (1-TPR)*f) / (FPR*s + (1-FPR)*f) )

which is bounded above by log(TPR/FPR) for every band and every instrument. TPR and FPR
are properties of the partition and are measurable by held-out prediction, so validating
a partition and calibrating the decision it serves are the same measurement.

Evidence then accumulates towards an escalation boundary and decays at the hazard's own
autocorrelation time, which turns that bound into a condition on whether a part can be
decided at all. For arrivals at a fixed interval the boundary is reached only when the
interval is shorter than `-tau * log(1 - Lambda/A)`.

`cfas/` implements this. `experiments/` measures it.

## Layout

| | |
| --- | --- |
| `cfas/` | The evidence core: the bound, the boundaries, the leak, the neutrality filter |
| `experiments/` | Thirteen experiments, each writing its results as JSON |
| `data/` | Derived artefacts for the pilot tile: the partition, its calibration, its geometry |
| `tests/` | The arithmetic, run offline |

The evidence core carries no model. The decision is deterministic code, so the same
ledger yields the same verdict on every run and an operator can replay it. A test asserts
that property against the package rather than trusting it.

## Data

No third-party data is redistributed. Embedding tiles are fetched from GeoTessera,
elevation from the Copernicus GLO-30 public bucket, optical scenes from the Earth Search
catalogue, and radar from the Planetary Computer's terrain-corrected Sentinel-1
collection. Each is fetched by the script that needs it and cached locally.

## Running it

```bash
pip install -r requirements.txt
export SSL_CERT_FILE=$(python3 -c "import certifi;print(certifi.where())")

cd experiments
python3 run_all.py          # about an hour; 6, 8, 9 and 11 need network
python3 make_figures.py
python3 make_figure1.py
```

Results are written to `experiments/results/` as JSON, each alongside the configuration
that produced it. The results of the runs behind this code are committed there already, so
any number can be checked, and any figure redrawn, without repeating the computation.
Rerunning overwrites them. Individual experiments run standalone, subject to the ordering
noted in `run_all.py`.

`exp12` and `exp13` are arithmetic on finished runs rather than fresh measurements: they
read the results of `exp6` and `exp2` and reproduce byte for byte from the committed
files.

A full eight-settlement run caches roughly 1.3 GB of tiles and rasters under
`experiments/`. All of it is gitignored and safe to delete; the scripts refetch.

```bash
python3 -m pytest tests -q
```

## A caveat worth repeating

Version 0.8.0 of the embedding client returns v1 data when asked for v2, without error.
Two independently clustered and independently calibrated tiles agreeing to four decimal
places is not a plausible measurement, and hashing them shows why. Pin the client at
0.9.0 or later.

## Licence

CC BY-SA 4.0. See `LICENSE`. Cite via `CITATION.cff`.
