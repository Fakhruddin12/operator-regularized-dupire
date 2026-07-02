# Operator-Regularized Dupire Local Volatility

This repository implements a complete deterministic-plus-Laplace pipeline for
recovering local volatility from option prices. The method parameterizes
log-variance, prices options with a forward Dupire PDE, and solves the inverse
problem with a Schrödinger-type operator regularizer.

## Project contents

The numbered notebooks build the project sequentially:

1. Black–Scholes utilities and grids
2. Local-volatility PDE pricer
3. Synthetic option data
4. Raw and smoothed Dupire baselines
5. Regularization operators and potentials
6. Linearized inverse calibration
7. Nonlinear damped Gauss–Newton calibration
8. GCV tuning and diagnostics
9. Laplace uncertainty approximation
10. Smooth synthetic benchmark
11. Localized-bump stress test
12. Real SPX option-data preparation
13. Real-market SSVI comparison
14. Final realistic-bumpy synthetic benchmark

Reusable implementation code is in `src/`, automated checks are in `tests/`,
and the longer Stage 13–14 runs are also available as scripts in `scripts/`.

## Installation

From Anaconda Prompt or a terminal:

```bash
conda create -n dupire_env python=3.12 -y
conda activate dupire_env
pip install -r requirements.txt
```

## Validation

Run the complete automated test suite from the project root:

```bash
python -m pytest tests -q
```

Then start Jupyter Lab:

```bash
jupyter lab
```

Open the notebooks in numerical order. Stages 7, 13 and 14 perform expensive
finite-difference calibrations; their validated outputs are included under
`outputs/` so the results can also be inspected without rerunning them.

## Surface convention

A surface has shape

```text
(number of maturities, number of log-moneyness points)
```

Rows correspond to maturities and columns to log-moneyness. Flattening uses C
order, so log-moneyness varies fastest.

## Final benchmark

On the final realistic-bumpy synthetic experiment, the operator method achieves
the lowest local-volatility reconstruction error and wins all ten repeated-noise
comparisons against SSVI. Stage 13 separately records the honest market result:
SSVI gives the better held-out price fit on the single SPX snapshot.
