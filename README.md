# A Comparative Evaluation of Uplift Modelling Methods and Metrics

A modular, reproducible Python framework comparing uplift modelling methods and
evaluation metrics (the Qini coefficient and AUUC) on the Criteo Uplift dataset.

MSc Computer Science with Data Science research project, University of Sunderland.
Author: Kemone S-G Brown.

Pre-registration: OSF DOI [10.17605/OSF.IO/5AYKJ](https://doi.org/10.17605/OSF.IO/5AYKJ)

## Overview

The study tests whether the Qini coefficient and AUUC produce stable, consistent
model rankings across 50 stratified subsamples of a large randomised experiment.
Four models are compared: an S-learner, a T-learner, a KL-divergence uplift tree
and a no-information baseline. All randomness is driven by fixed seeds, so the
pipeline is deterministic and its outputs are byte-reproducible.

## Result (summary)

Across 50 subsamples, the Qini and AUUC rankings were highly consistent (mean
Kendall tau-b 0.99, 95% bootstrap CI [0.98, 1.00]), so the pre-registered
hypothesis of cross-metric inconsistency (H2) was not supported.

## Dataset

Criteo Uplift Prediction Dataset (v2.1), approximately 13.9M rows. Not included
here because of its size; download it and place it at
data/criteo-research-uplift-v2.1.csv.gz:
https://ailab.criteo.com/criteo-uplift-prediction-dataset/

## Reproducing the analysis

Built and run in Docker for a portable, version-pinned environment.

    docker build -t uplift-eval:dev .
    docker run --rm -v "${PWD}:/project" uplift-eval:dev python -m pytest tests/ -v
    docker run --rm -v "${PWD}:/project" uplift-eval:dev python -m src.data_processing
    docker run --rm -v "${PWD}:/project" uplift-eval:dev python -m src.analysis
    docker run --rm -v "${PWD}:/project" uplift-eval:dev python -m src.correlations
    docker run --rm -v "${PWD}:/project" uplift-eval:dev python -m src.hypothesis
    docker run --rm -v "${PWD}:/project" uplift-eval:dev python -m src.reproducibility

## Structure

- src/data_processing.py: load, validate and partition the dataset
- src/models.py: the four uplift models
- src/metrics.py: Qini coefficient and AUUC
- src/analysis.py: subsample generation and per-subsample evaluation
- src/correlations.py: Kendall tau-b and Spearman rho
- src/hypothesis.py: bootstrap confidence interval and Wilcoxon tests
- src/reproducibility.py: byte-identity validation across re-runs
- tests/: unit tests (48)

## Licence

MIT (see LICENSE).
