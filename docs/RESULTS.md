# Current results: literature audit

The authoritative inputs are the refined-price rows in `outputs/stage_23_literature_audit_final/final_metrics.csv`. The compact reporting tables below are in [reports/literature_audit](../reports/literature_audit/).

## Pricing: percentage change in RMSE

Negative means the full operator has lower error. Positive means it has higher error.

| Input | Holdout | Versus SSVI PDE | Versus H1 | Versus H2 |
|---|---|---:|---:|---:|
| Author-supplied Bates prices | Expiry | −75.44% | +9.39% | −23.12% |
| Author-supplied Bates prices | Quote | −75.30% | −3.34% | −16.89% |
| Euro Stoxx 2010 quotes | Expiry | +46.81% | −3.17% | +23.07% |
| Euro Stoxx 2010 quotes | Quote | −34.18% | −0.93% | +5.12% |

Source: [paired_comparisons.csv](../reports/literature_audit/paired_comparisons.csv). These are pooled pricing comparisons on specific datasets, not statements about all markets. The reference SSVI fit uses conservative sufficient butterfly conditions. The benchmark does not compare against every possible SSVI fit.

## Known-truth recovery: ten fresh noise draws

| Method | Mean full-domain local-volatility RMSE |
|---|---:|
| SSVI PDE | **0.04096** |
| H1 | 0.09546 |
| H1 plus density potential | 0.09662 |
| Full operator | 0.09679 |
| H1 plus wing potential | 0.10024 |
| H2 | 0.11643 |
| Unregularized correction | 0.19237 |

Source: [synthetic_recovery_summary.csv](../reports/literature_audit/synthetic_recovery_summary.csv). The full operator wins only five of the ten draws against SSVI. Its large mean error is driven by unstable fits; reporting only a successful seed would misrepresent the study.

## What the study establishes

The pipeline implements a testable inverse-calibration hypothesis and exposes its failure modes. Some corrections improve held-out pricing, but that does not imply better recovery of the latent local-volatility surface. An operator penalty can improve stability without guaranteeing lower statistical risk.

## Limits of the comparison

- Finite optimisation budgets: 72 selected correction fits reached the Gauss–Newton limit; only three subsequent refinements returned optimiser-success flags.
- No latent local-volatility truth is observed in market data.
- H1/H2 comparators share the project's correction parameterization, rather than duplicating the papers' original software.
- The Albani-style synthetic experiment uses the published truth and layout with new noise draws and known spot.
- Quote weights use a declared vega-based scale, not measured bid–ask noise.

[Full audit report](../reports/Dupire_Literature_Audit_Report.pdf) · [Numerical summary](../reports/literature_audit/numerical_summary.json) · [Exact assumptions and theory](../reports/Regularizer_Theory.md)
