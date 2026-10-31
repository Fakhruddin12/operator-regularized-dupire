# Operator-Regularised Dupire: research decision

Reviewed 31 October 2026. This note separates the supported result from the next research decision.

## Decision

Resolve numerical optimisation and recovery instability before claiming benefit from the spatial potentials.

## Strongest relevant comparison

Euro Stoxx quote-holdout price RMSE is 34.18% below SSVI but only 0.93% below ordinary H1; whole-expiry RMSE is 46.81% worse than SSVI. On known truth, mean volatility RMSE is 0.09679 versus SSVI 0.04096; five of ten draws favour the full operator.

## Evidence limits

Of 250 selected finite fits, 72 hit the iteration limit and 117 have active bounds. Fixed-linearisation Lipschitz continuity is conditional; it is not a proof of global nonlinear stability, novelty or superior recovery. Saved fits were repriced; full nested optimisation was not rerun.

## Checks and changes

The existing 61-test suite, fit manifests and independent repricing remain the evidence source. This review foregrounds the H1 comparator, expiry failure and optimiser limits; no new nested fit is claimed.

## Next independent milestone

Predeclare convergence criteria and resource budgets, repair failed optimisation, then rerun controlled H1/H2/operator comparisons with held-out expiries and known truth.

[Project report](../reports/PROJECT_REPORT.pdf) · [Prepared LinkedIn Projects entry](../reports/LINKEDIN_PROJECT_ENTRY.md)

The report and entry are portfolio summaries. Original experimental source version: `b1ba3bcf7eee7c1a5e6d14deb0bb517ccfbb4053`. The report records verification scope and distinguishes earlier numerical replays from this presentation/code review.
