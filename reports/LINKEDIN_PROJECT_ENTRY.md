# Dupire — LinkedIn Projects entry

## Project name
Operator-Regularised Dupire Local-Volatility Calibration

## Description
Built a PDE-constrained calibration pipeline to recover local volatility from sparse option prices, using a bounded log-variance correction to an SSVI reference. Designed a Schrödinger-type quadratic penalty combining directional smoothness with quote-density and strike-wing potentials, and derived a conditional Lipschitz bound for the fixed linearised estimator.

Implemented Crank–Nicolson pricing, discrete tangent sensitivities, damped Gauss–Newton calibration and nested regularisation selection. Compared SSVI, unregularised, H1, H2 and potential ablations across 50 outer folds using public Euro Stoxx quotes, author-supplied synthetic Bates prices and ten known-truth noise draws.

The full operator reduced Euro Stoxx quote-holdout price RMSE by 34.18% versus SSVI, but increased whole-expiry error by 46.81%. It did not consistently outperform conventional smoothness penalties, and average known-truth recovery was worse than SSVI. The report examines these failures and unresolved optimisation limits alongside the gains, distinguishing pricing accuracy from surface recovery and conditional stability.

## Skills to associate
Quantitative Research; Numerical Analysis; Derivatives Pricing; Python; Inverse Problems

If a skill label is unavailable in LinkedIn's selector, choose the closest existing label. Do not create unsupported claims about endorsements.

## Project URL
https://github.com/Fakhruddin12/operator-regularized-dupire

## Media attachment
- File: Dupire_Project_Report.pdf
- Title: Operator-Regularised Dupire — Methodology and Results
- Description: Research report covering the forward PDE, SSVI correction model, spatial regulariser, conditional stability derivation, nested evaluation, known-truth recovery and numerical limitations. Includes positive and negative results, input provenance and verification scope.

## Dates and association
Use the actual project start/end dates. They have not been inferred from commit dates or the report's assembly date. Associate with an institution only if the project was genuinely completed there; otherwise leave it independent. Add collaborators only if applicable.

## Evidence notes — not part of the pasted description
- Source version: b1ba3bcf7eee7c1a5e6d14deb0bb517ccfbb4053.
- 155 Euro Stoxx quotes / 12 maturities; 324 synthetic Bates observations / 18 maturities.
- 50 outer folds include 10 regenerated known-truth noise draws.
- Full-domain recovery RMSE: full operator mean 0.09679; SSVI mean 0.04096. Full operator beats SSVI in 5/10 draws.
- Conditional Lipschitz continuity concerns a fixed, unconstrained linearised estimator. It is not a global nonlinear theorem or a proof of improved surface accuracy.
- 72/250 selected correction fits reached the initial iteration budget; only 3/72 training-only polishing attempts reported optimiser success.
- Verification replays saved fits and reporting evidence; it does not refit the complete nested study.
- 61 repository tests passed.
- Report date is the assembly date, not a claimed project completion date.

## Public report link
https://github.com/Fakhruddin12/operator-regularized-dupire/blob/main/reports/PROJECT_REPORT.pdf

## October review clarification — evidence note
Euro Stoxx quote-holdout price RMSE is 34.18% below SSVI but only 0.93% below ordinary H1; whole-expiry RMSE is 46.81% worse than SSVI. On known truth, mean volatility RMSE is 0.09679 versus SSVI 0.04096; five of ten draws favour the full operator.

Of 250 selected finite fits, 72 hit the iteration limit and 117 have active bounds. Fixed-linearisation Lipschitz continuity is conditional; it is not a proof of global nonlinear stability, novelty or superior recovery. Saved fits were repriced; full nested optimisation was not rerun.
