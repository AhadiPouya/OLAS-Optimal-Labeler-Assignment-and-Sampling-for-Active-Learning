# OLAS: Optimal Labeler Assignment and Sampling

Code for the paper "Active Learning with Imperfect Labels: Optimal Labeler
Assignment and Sample Selection" (submitted to INFORMS Journal on Data
Science, manuscript IJDS-2025-0105).

OLAS is an active learning framework that jointly decides which sample to
query next and which labeler should annotate it, accounting for the fact
that labelers make mistakes and that mistakes are more likely on uncertain
samples. This repo has the code for the benchmark experiments, the
scalability analysis, the significance tests, and the real-data validation
on the Music Genre Classification dataset.

A separate case study on Ford Motor Company warranty claim data is
described in the paper but is not included here, since that data is
proprietary and can't be shared.

## Files

- `olas_uci_experiments.py` — runs OLAS and the baseline methods (RS+RLA,
  RS+OLA, ES+RLA, ES+OLA, MV, KC+RLA) on the five UCI/Covertype datasets.
- `olas_covertype_optimized.py` — a faster version of the same experiment,
  used specifically for the 50,000-instance Covertype run.
- `olas_badge_runner.py` — runs the BADGE+RLA baseline separately (it's
  computationally different enough that it made sense to keep it as its
  own script rather than folding it into the main runner).
- `olas_alpha_sensitivity.py` — the α sensitivity analysis on the Heart
  Statlog and Spambase datasets.
- `music_pipeline.py` — fits the noise-estimation pipeline (Section 3.4 of
  the paper) on real crowdsourced annotations and runs the active learning
  comparison on the Music Genre Classification dataset.
- `olas_significance_tests.py` — computes the paired/unpaired significance
  tests reported in the paper's appendix, reading from the result files
  the scripts above produce.

## Setup

```bash
pip install -r requirements.txt
```

Tested on Python 3.9.6, macOS (arm64). Package versions are pinned to what
was actually used to produce the results in the paper.

## Running it

Each script expects a project folder with `Data/` and `Results/`
subfolders. By default it looks for one called `OLAS_Results` in your
current directory; set the `OLAS_BASE_DIR` environment variable if you
want it somewhere else. If you're running in Google Colab, uncomment the
Drive-mount lines near the top of each script.

1. Download the UCI/Covertype datasets into `Data/` (see below for
   sources).
2. Run `olas_uci_experiments.py` for the four smaller UCI datasets, and
   `olas_covertype_optimized.py` for Covertype.
3. Run `olas_badge_runner.py` for the BADGE+RLA results.
4. Run `olas_alpha_sensitivity.py` for the sensitivity analysis.
5. Run `music_pipeline.py` for the Music dataset results.
6. Run `olas_significance_tests.py` after the above to reproduce the
   significance tests.

Each script writes its results as `.xlsx` files (one row per replication),
which is what the downstream scripts read.

## Datasets

- Statlog (Heart), Ionosphere, Connectionist Bench (Sonar), Spambase, and
  Covertype are all from the UCI Machine Learning Repository.
- The Music Genre Classification dataset is from Rodrigues, Pereira, and
  Ribeiro (2013), *Pattern Recognition Letters*.

None of these are redistributed here; the scripts assume you've downloaded
them yourself into `Data/`.

## License

MIT (see `LICENSE`).
