# Operator-Regularized Dupire Calibration
### Recovering local volatility from sparse option prices

## Included saved artifacts

[calibration-arrays.zip](calibration-arrays.zip) contains 644 original saved files. Extract it at the repository root before using the archived arrays or data snapshots:

```bash
python -m zipfile -e calibration-arrays.zip .
```

Paths and SHA-256 hashes are recorded in [ARTIFACTS.json](ARTIFACTS.json). Source code, figures and result tables remain directly browsable.

**Research question:** Does adding location-dependent penalties to an inverse option-pricing problem improve generalisation and recovery of the underlying volatility surface?

I built a forward Dupire PDE pricer and a regularized calibration pipeline, then compared the proposed penalty with **SSVI, H1/H2 smoothing and unregularized correction** under shared numerical settings.

**Current conclusion:** The broader audit found **mixed pricing improvements and unstable known-truth recovery**. The project demonstrates numerical research, controlled comparisons and critical evaluation; it does not establish universal superiority or novelty.

![Known-truth recovery across ten noise draws](docs/assets/recovery-comparison.png)

[**Results and interpretation**](docs/RESULTS.md) · [**Full audit report**](Dupire_Literature_Audit_Report.pdf) · [**Theory**](reports/Regularizer_Theory.md) · [**Literature comparison**](reports/Literature_Comparison.md)

## The modelling idea

Option quotes are sparse and noisy. Differentiating an interpolated price surface can amplify that noise. Instead, the implementation fits a positive local-variance surface by repricing options through the PDE.

The current audit starts with an SSVI reference and fits a tapered log-variance correction. Smoothness, magnitude and location-dependent penalties control that correction. The comparison keeps the reference, correction coordinates, weights, numerical solver and optimisation budget fixed while changing the penalty.

$$
\min_h \; \|W(F(u_{\mathrm{ref}}+h)-C_{\mathrm{obs}})\|_2^2+\lambda h^\top R h,
\qquad \sigma_{\mathrm{loc}}^2=\exp(u).
$$

Here $F$ is the pricing map and $R$ combines smoothness and potential terms, with the grid scaling defined in the implementation. [The mathematical note](reports/Regularizer_Theory.md) explains the exact linearized risk calculation, the bias–variance trade-off and why stability alone does not guarantee estimation accuracy.

## Current evidence

- **50 outer folds** across literature-linked inputs and synthetic recovery experiments.
- Separate quote and expiry holdouts, plus numerical and parameter-grid sensitivity checks.
- On ten known-truth noise draws, full-domain mean local-volatility RMSE was **0.09679** for the full operator versus **0.04096** for SSVI; the operator won **5 of 10** draws.
- Improvements over H1/H2 depend on the dataset and protocol. The [complete results](docs/RESULTS.md) report the losses as well as the wins.

These are finite-budget comparisons. Many selected correction fits reached the iteration limit. H1/H2 are controlled comparators in the same coordinates, rather than exact reproductions of each paper's implementation. Market quote weights are declared numerical weights; bid–ask spreads were unavailable.

## Reproduce

The archived audit used Python 3.12.14. From the repository root:

```bash
python -m pip install -r requirements-audit.txt
python -m pytest -q
OPENBLAS_NUM_THREADS=1 python scripts/23_literature_benchmarks.py --inputs data/literature/benchmark_inputs --jobs 6
OPENBLAS_NUM_THREADS=1 python scripts/24_literature_diagnostics.py --stage outputs/stage_23_literature_audit_final --jobs 3
OPENBLAS_NUM_THREADS=1 python scripts/25_parameter_grid_sensitivity.py
OPENBLAS_NUM_THREADS=1 python scripts/26_analyze_literature_results.py
```

The full benchmark is computationally expensive. Saved tables, diagnostics and the report can be inspected immediately. See [input provenance](data/literature/benchmark_inputs/README.md).

## Where to look

| Location | Purpose |
|---|---|
| [src/pricing](src/pricing/) | Forward PDE pricing |
| [src/inverse](src/inverse/) | Calibration and sensitivities |
| [src/regularization](src/regularization/) | Operators, grid scaling and potentials |
| [tests](tests/) | Numerical and implementation checks |
| [reports/literature_audit](reports/literature_audit/) | Current evidence tables |
| [outputs/stage_23_literature_audit_final](outputs/stage_23_literature_audit_final/) | Fold-level predictions and diagnostics |
| [notebooks](notebooks/) | Earlier tutorial and research sequence |

Earlier stage outputs remain available for traceability and were not all regenerated after the audit fixes. Their favourable synthetic numbers are **historical**, not the current headline evidence. The archived Laplace module is outside the current study's scope. This is a calibration study; trading profitability is not evaluated.

**Author:** Fakhruddin Hakim Hussain · [Portfolio](https://github.com/Fakhruddin12)
