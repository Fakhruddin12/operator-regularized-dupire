# Quantum-Mechanics-Inspired Regularisation for Dupire Calibration

**Can the energy structure of a Schrödinger operator help recover a more reliable local-volatility surface from sparse option prices?**

Option prices reveal information about volatility, but a finite set of noisy quotes does not uniquely determine an entire surface. A model can match those quotes while producing unstable behaviour between them. This research project investigates a **quantum-mechanics-inspired regulariser** that controls both how a fitted correction varies and where it is allowed to depart from a reference surface.

The specific modelling choice is a **Schrödinger-type quadratic energy**: a smoothness term plays the role of kinetic energy, while a spatial potential penalises corrections more strongly where quotes are sparse or strikes are far from the money. I implement that energy inside a Dupire PDE calibration, derive a conditional **Lipschitz stability bound**, and test whether the added structure improves pricing and volatility recovery against simpler penalties.

[**Theory and derivations**](reports/Regularizer_Theory.md) · [**Comparison with existing research**](reports/Literature_Comparison.md) · [**Results**](docs/RESULTS.md) · [**Full audit report**](Dupire_Literature_Audit_Report.pdf)

## What differs from conventional calibration?

Regularisation is already central to local-volatility calibration. The research question here concerns the **particular spatial penalty**, rather than whether regularisation should be used at all.

| Approach | What controls the fitted surface? | Difference investigated here |
|---|---|---|
| Direct Dupire reconstruction | Derivatives of a fitted option-price surface | Reprice options through the PDE and optimise the volatility correction directly, avoiding explicit second differentiation of noisy market quotes. |
| Reference-centred H1/Tikhonov calibration | Correction magnitude and first derivatives | Add a spatial potential that changes the cost of departing from the reference across the domain. |
| Second-order/H2 smoothing | Higher-order derivative penalties | Compare a first-order Schrödinger-type energy with a second-order penalty in the same correction coordinates. |
| **This project's full operator** | **Magnitude + directional smoothness + quote-density and wing potentials** | **Test whether the chosen spatial shrinkage helps beyond H1, H2 and their individual-potential variants.** |

Reference-centred Tikhonov calibration has established predecessors, including [Crépey (2003)](https://math.maths.univ-evry.fr/crepey/papers/40020SIMA.pdf). The distinguishing hypothesis is the construction and use of the potentials within this shared calibration framework. The Schrödinger analogy alone does not establish mathematical novelty. [Detailed literature positioning](reports/Literature_Comparison.md).

## 1. The quantum-mechanical idea

In quantum mechanics, a Schrödinger Hamiltonian combines a kinetic operator with a potential. Its quadratic energy balances variation in a function against the cost of its location. This project adapts that structure to the **real-valued volatility correction**. [Hamiltonian background](https://farside.ph.utexas.edu/teaching/qm/Quantumhtml/node34.html).

Let $h$ be the correction amplitude on nondimensional log-moneyness and maturity coordinates $(\xi,s)$. The proposed energy is

$$
\mathcal E_V(h)=\int_\Omega\left[\alpha_x|\partial_\xi h|^2+\alpha_T|\partial_s h|^2+(\beta+V(\xi,s))|h|^2\right]\,d\xi\,ds.
$$

With suitable boundary conditions, its associated differential operator has the Schrödinger form

$$
\mathcal H=-\alpha_x\partial_{\xi\xi}-\alpha_T\partial_{ss}+\beta+V(\xi,s).
$$

- **Kinetic/smoothness terms:** discourage rapid changes across strikes and maturities.
- **Positive magnitude term $\beta$:** penalises large corrections and supplies a positive lower bound for the discrete energy.
- **Spatial potential $V$:** makes departures from the reference more expensive in selected regions.

The implementation builds $V$ from two components. A Gaussian-kernel estimate of **training-quote location density** gives a confidence surface $c\in[0,1]$, with penalty $V_{\mathrm{density}}=0.2(1-c)$. A quadratic **wing potential**, rising outside the central strike region, contributes up to 0.5. Together they provide stronger local shrinkage where the surface is weakly supported. Quote density is an information-location proxy, not a measured bid–ask noise variance.

The potential is bounded and nonnegative. It changes the **spatial weighting within an H1-type energy**; it does not create a higher Sobolev smoothness order. [Energy equivalence and assumptions](reports/Regularizer_Theory.md).

## 2. How it enters Dupire calibration

The model starts from an SSVI reference and fits a correction to **log local variance**:

$$
u=u_{\mathrm{ref}}+Qh,\qquad\sigma_{\mathrm{loc}}=\exp(u/2).
$$

Here $Q$ interpolates and smoothly tapers the grid correction. The exponential preserves positive volatility; the taper returns to the reference outside the correction's support. The regularisation acts on **$h$**, not on the tapered correction $Qh$.

For $n$ training quotes and $p$ correction parameters, the implemented objective is

$$
\min_{h\in[-2.5,2.5]^p}\frac1n\left\|W\left(F(u_{\mathrm{ref}}+Qh)-C_{\mathrm{obs}}\right)\right\|_2^2+\frac\lambda p h^\top Rh.
$$

$F$ reprices options through the Dupire PDE, $W$ contains the declared quote scales, and $R$ discretises the energy above using derivative and quadrature weights. Each penalty matrix is normalised to the same mean diagonal scale before its own $\lambda$ is selected.

The workflow is:

1. **Fit the reference** using training prices.
2. **Construct the potentials** from training-quote locations and the declared wing rule.
3. **Build the operator matrix** on nondimensional coordinates with consistent nodal and edge quadrature.
4. **Calibrate with damped Gauss–Newton steps**, backtracking and amplitude bounds. The normal equations include $\lambda R/p$ in the curvature matrix and $-\lambda Rh/p$ on the right-hand side.
5. **Select regularisation inside the training fold**, including the unchanged reference as a candidate, then evaluate on withheld quotes or expiries.

The same reference, correction coordinates, price scales, PDE settings and optimisation budget are shared by the controlled comparisons. Density-only and wing-only variants test what each added component contributes. [Exact benchmark workflow](src/evaluation/literature_benchmark.py).

## 3. Lipschitz continuity: what is established

The useful stability question is: **how much can the estimated correction move when the observed prices change slightly?**

Fix the reference, interpolation, weights, regulariser and pricing linearisation. Let $A$ be the weighted pricing Jacobian and $b$ the corresponding weighted price residual. For $\lambda>0$, write $\tau=n\lambda/p$. The unconstrained linearised estimator is

$$
\widehat h(b)=(A^\top A+\tau R)^{-1}A^\top b.
$$

Nonnegative potentials, positive quadrature weights and $\beta>0$ make $R$ positive definite. If $\mu=\lambda_{\min}(R)>0$, then

$$
\left\|\widehat h(b+\Delta b)-\widehat h(b)\right\|_2\le\frac{\|A\|_2}{\tau\mu}\,\|\Delta b\|_2.
$$

**Why:** the difference equals $(A^\top A+\tau R)^{-1}A^\top\Delta b$, and the smallest eigenvalue of the matrix being inverted is at least $\tau\mu$. This gives a finite bound on noise amplification: the fixed linearised inverse is **Lipschitz continuous**.

This result motivates the regularised Gauss–Newton systems used in the implementation. Its stated assumptions matter: it does not establish global Lipschitz continuity or uniqueness of the full nonlinear, refitted and tuned calibration procedure. Nor does stability alone imply lower estimation error—stronger shrinkage can add bias. Each comparison is also trace-normalised and tuned separately, so the added potential does not automatically improve its stability constant. [Proof, risk decomposition and counterexample](reports/Regularizer_Theory.md).

## 4. What the experiments show

The study covers **50 outer folds**. The table compares held-out pricing RMSE with the SSVI-based PDE baseline; lower RMSE means more accurate option prices.

| Data | What was withheld | Full operator's pricing error versus baseline |
|---|---|---:|
| Euro Stoxx market options | Individual quotes | **34.18% lower** |
| Euro Stoxx market options | Entire expiries | **46.81% higher** |
| Author-supplied simulated Bates prices | Individual quotes | **75.30% lower** |
| Author-supplied simulated Bates prices | Entire expiries | **75.44% lower** |

On ten new noise draws with known volatility, the full operator won five times but had **higher mean recovery error** than SSVI: 0.09679 versus 0.04096. Some unstable fits dominate that average. The results support useful pricing gains in particular settings, while showing that the chosen quantum-inspired energy does not consistently recover the latent surface more accurately. [Full H1/H2 comparisons and diagnostics](docs/RESULTS.md).

![Known-volatility recovery across ten noise draws](docs/assets/recovery-comparison.png)

The comparisons use finite optimisation budgets and many fits reached iteration limits. Market data provide no true local-volatility surface. Numerical quote weights are not measured bid–ask uncertainties, and the H1/H2 comparators are controlled adaptations rather than exact reproductions of each paper's implementation. These are calibration results; trading profitability is not evaluated.

## Code and reproduction

| Component | Implementation |
|---|---|
| Density and wing potentials | [potentials.py](src/regularization/potentials.py) |
| Grid scaling and energy matrix | [scaling.py](src/regularization/scaling.py) |
| Tapered reference correction | [reference_correction.py](src/inverse/reference_correction.py) |
| PDE pricing and sensitivities | [calibration_pde.py](src/pricing/calibration_pde.py) |
| Penalties, optimisation and nested comparison | [literature_benchmark.py](src/evaluation/literature_benchmark.py) |

Saved tables and figures are browsable immediately. Python 3.12.14 was used for the audit. From the repository root:

```bash
python -m pip install -r requirements-audit.txt
python -m pytest -q
OPENBLAS_NUM_THREADS=1 python scripts/23_literature_benchmarks.py --inputs data/literature/benchmark_inputs --jobs 6
OPENBLAS_NUM_THREADS=1 python scripts/24_literature_diagnostics.py --stage outputs/stage_23_literature_audit_final --jobs 3
OPENBLAS_NUM_THREADS=1 python scripts/25_parameter_grid_sensitivity.py
OPENBLAS_NUM_THREADS=1 python scripts/26_analyze_literature_results.py
```

The full benchmark is computationally expensive. [Input provenance](data/literature/benchmark_inputs/README.md) and [tests](tests/) document the inputs and implementation checks. For archived arrays, extract `calibration-arrays.zip` at the repository root with `python -m zipfile -e calibration-arrays.zip .`; [ARTIFACTS.json](ARTIFACTS.json) lists paths and hashes. Earlier outputs remain historical evidence, separate from the current audit. The archived Laplace module is outside this study.

**Author:** Fakhruddin Hakim Hussain · [Portfolio](https://github.com/Fakhruddin12)
