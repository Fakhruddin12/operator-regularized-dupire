# Schrödinger-Type Regularisation: Construction, Stability and Risk

## 0. Quantum-mechanical construction and its calibration meaning

A Schrödinger Hamiltonian has a kinetic differential term and a location-dependent potential. Its energy form motivates the present regulariser on a real-valued correction amplitude. On nondimensional log-moneyness and maturity coordinates $(\xi,s)$, define

$$
\mathcal E_V(h)=\int_\Omega\left[\alpha_x|\partial_\xi h|^2+\alpha_T|\partial_s h|^2+(\beta+V)|h|^2\right]d\xi\,ds.
$$

Under boundary conditions that remove the integration-by-parts boundary terms, the associated operator is

$$
\mathcal H=-\alpha_x\partial_{\xi\xi}-\alpha_T\partial_{ss}+\beta+V.
$$

The derivative terms penalise variation; the potential penalises correction amplitude differently across the domain. The benchmark uses a density potential of at most 0.2 and a quadratic wing potential of at most 0.5. Density is constructed only from fitting-quote locations, with Gaussian kernels normalised to $[0,1]$. It is an information-location proxy, not an estimated observation-noise covariance. The wing term increases away from the central strike region.

The discrete implementation uses positive nodal and edge quadrature:

$$
R_{\rm raw}=\beta M+\alpha_xD_\xi^\top M_\xi D_\xi+\alpha_TD_s^\top M_sD_s+\operatorname{diag}(m_iV_i),\qquad R=\frac{R_{\rm raw}}{\operatorname{tr}(R_{\rm raw})/p}.
$$

Here $M=\operatorname{diag}(m_i)$ is the nodal mass matrix; the edge matrices weight the finite differences. With $\beta>0$, $m_i>0$ and nonnegative potential/smoothness coefficients, $R$ is positive definite. The mean-diagonal normalisation removes arbitrary overall scale before each method's regularisation strength is tuned. Consequently, adding a positive potential before normalisation does not imply a matrix-order improvement after normalisation and separate tuning.

For an eigendecomposition $R=U\operatorname{diag}(r_j)U^\top$, the penalty is $h^\top Rh=\sum_jr_j(U^\top h)_j^2$: high-energy modes are penalised more strongly. This is the useful spectral interpretation of the quantum-mechanical analogy. It remains a classical regularised inverse problem, and the proposed potentials must earn their usefulness through assumptions and empirical comparison. The bounded-potential energy has the same H1 function class as its baseline, as section 5 establishes.

The operator is built in [scaling.py](../src/regularization/scaling.py), its potentials in [potentials.py](../src/regularization/potentials.py), and its normalisation and use in [literature_benchmark.py](../src/evaluation/literature_benchmark.py). For Hamiltonian background, see [Fitzpatrick's quantum-mechanics notes](https://farside.ph.utexas.edu/teaching/qm/Quantumhtml/node34.html); for the nearest calibration predecessors, see [the literature comparison](Literature_Comparison.md).

This note concerns the audited model

$$
u=u_{\rm SSVI}+Qh,\qquad
\widehat h=\arg\min_h\frac1n\|W(F(h)-y)\|^2+
\frac\lambda p h^\top R h.
$$

Here $Q$ interpolates and tapers the latent amplitude. The penalty is on $h$, not on $Qh$. A superior estimate must specify a loss: held-out prices, latent amplitude, actual log variance, volatility, or hedging error are different targets.

## 1. Exact risk for a fixed linearization

Hold the SSVI reference, observation locations, scales and regularizer fixed. Write the standardized linearized experiment as

$$
b=A h_*+\epsilon,\quad E\epsilon=0,\quad
E\epsilon\epsilon^\top=\Sigma.
$$

With $\tau=n\lambda/p$, $G=A^\top A+\tau R$, and $R\succ0$, the unique unconstrained linearized estimator is

$$
\widehat h=K b,\qquad K=G^{-1}A^\top.
$$

For any fixed evaluation map $B$, its exact mean squared error is

$$
\mathcal R_B(R,h_*)
=\|B(KA-I)h_*\|^2+
\operatorname{tr}(BK\Sigma K^\top B^\top).
$$

The first term is squared shrinkage bias; the second is variance. Use $B=I$ for latent amplitude, $B=Q$ for actual log-variance correction, or a test-price Jacobian for first-order price risk. For local volatility, the first-order map additionally multiplies by $\sigma_*/2$. Exact nonlinear volatility risk is not given by this linear formula.

Let $K_f$ and $K_c$ denote full and conventional penalties. Subtract their risk formulas. A guarantee over an explicitly chosen truth class $\mathcal H$ follows if that difference is nonpositive for every $h_*\in\mathcal H$, with the claimed noise covariance. This is a precise mathematical comparison, but the ordering need not hold. The present data-based reference, tuning and box constraints require additional analysis; the formula alone does not prove their risk ordering.

## 2. A counterexample to universal benefit from extra shrinkage

Take one coordinate: $b=h_*+\epsilon$, $\operatorname{Var}(\epsilon)=s^2$. Ridge strength $t\ge0$ gives

$$
\widehat h_t=\frac{b}{1+t},\qquad
E(\widehat h_t-h_*)^2=\frac{t^2h_*^2+s^2}{(1+t)^2}.
$$

Increasing strength from $t_1$ to $t_2>t_1$ reduces variance at $h_*=0$, while increasing the coefficient of $h_*^2$. It therefore loses for sufficiently large $|h_*|$. For example, $t_1=1,t_2=2,s^2=1$: the stronger penalty is better exactly when $h_*^2<5/7$, equal at $5/7$, and worse above it. This disproves an assumption-free claim that adding a nonnegative potential always improves estimation. Both values $h_*=0$ and $h_*=2$ illustrate the reversal inside the implementation's amplitude bound. Taking noise equal to +1 or -1 with equal probability also keeps both estimates inside that box for these two truths, so the reversal need not rely on inadmissible amplitudes.

Trace normalization and separately selected lambdas mean the fitted full penalty is not simply a larger positive-semidefinite matrix than the fitted H1 penalty. No matrix-order superiority is asserted for that comparison.

## 3. A useful conditional optimality theorem

In a correctly specified *linear Gaussian* model with independent noise covariance $s^2I$, known fixed reference and prior

$$
h_*\sim N(0,C),\qquad C^{-1}\succ0,
$$

the posterior mean minimizes integrated squared error among all estimators. It is exactly the Tikhonov estimator when

$$
\tau R=s^2 C^{-1}.
$$

It also minimizes integrated squared error for any fixed linear target $Bh_*$. Thus an operator describing the true correction covariance has a Bayesian optimality justification. Calling a density-and-wing operator the true prior precision is an assumption, not a consequence of low quote density. The theorem does not establish pointwise improvement for every fixed truth, nor exact optimality for the nonlinear PDE estimator.

## 4. Stability and existence are provable, with narrower meaning

For nonnegative potentials and positive quadrature weights,

$$
R=\beta M+\alpha_xD_x^\top M_xD_x+
\alpha_TD_T^\top M_TD_T+M V
$$

is symmetric positive definite when $\beta>0$. Consequently each regularized linearized system has a unique solution. If $\mu=\lambda_{\min}(R)>0$,

$$
\|\widehat h(b+\Delta b)-\widehat h(b)\|
\le\frac{\|A\|}{\tau\mu}\|\Delta b\|.
$$

This is a conservative sensitivity bound. A larger lower eigenvalue improves this bound, but need not improve condition number, exact estimator variance in every direction, or total estimation risk. In the implemented finite box $[-2.5,2.5]^p$, a continuous finite PDE objective attains a minimum by compactness. Nonlinear uniqueness and convergence of a local optimizer do not follow from a positive quadratic regularizer.

For completeness, the Lipschitz argument is a direct resolvent bound. In the fixed linear experiment of section 1, take $\lambda>0$, $\tau=n\lambda/p$, $G=A^\top A+\tau R$ and $\mu=\lambda_{\min}(R)>0$. Then

$$
\widehat h(b+\Delta b)-\widehat h(b)=G^{-1}A^\top\Delta b,\qquad\lambda_{\min}(G)\ge\tau\mu.
$$

Taking Euclidean operator norms yields the displayed bound. The exact Lipschitz constant is $\|G^{-1}A^\top\|_2$; $\|A\|_2/(\tau\mu)$ is an upper bound. For perturbations in unweighted prices, $\Delta b=W\Delta C$, so an additional factor $\|W\|_2$ converts it to a bound in the original price norm.

This establishes data-to-correction Lipschitz continuity for the **fixed, unconstrained linearised estimator**. It is not a global theorem for the nonlinear pipeline with refitted reference, changing Jacobian, constraints and selected hyperparameters. It also does not assert that the continuum surface itself is globally Lipschitz merely because an H1 energy is finite. Those are separate regularity questions.

In the benchmark, the Gauss–Newton curvature is $A_k^\top A_k/n+\lambda R/p$, with positive numerical damping added for the step solve. Thus the same positive quadratic term stabilises the local systems, while backtracking and box projection control the nonlinear updates. This connection motivates the implementation without promoting the fixed-linear theorem into an unsupported nonlinear accuracy guarantee.

The taper preserves the reference outside its support and the exponential ensures positive volatility. Neither property by itself guarantees a more accurate surface. If the true log-variance correction lies outside the tapered interpolation space, there is unavoidable approximation error even with noiseless data.

## 5. The added bounded potential preserves the H1 function class

Before trace normalization, let

$$
E_0(h)=\beta\|h\|_{L^2}^2+\alpha_x\|\partial_xh\|_{L^2}^2+
\alpha_T\|\partial_Th\|_{L^2}^2,\qquad
E_V(h)=E_0(h)+\int Vh^2.
$$

For fixed positive $\beta,\alpha_x,\alpha_T$ and $0\le V\le V_{\max}<\infty$,

$$
E_0(h)\le E_V(h)\le(1+V_{\max}/\beta)E_0(h).
$$

The first inequality follows from nonnegativity; the second uses $\int Vh^2\le V_{\max}\|h\|^2\le(V_{\max}/\beta)E_0(h)$. Thus these are equivalent anisotropic H1 norms on the same function class. The same argument holds for the positive quadrature-weighted discrete energies. Positive scalar trace normalization preserves equivalence, with rescaled constants.

The implemented confidence potential is at most 0.2 and the wing potential at most 0.5, so one can take $V_{\max}=0.7$ before normalization in this benchmark. This bounded-potential modification changes spatial shrinkage, but does not by itself introduce a higher Sobolev smoothness order or prove an improved asymptotic convergence rate over H1. The equivalence constants may be large; this statement is not a quantitative accuracy guarantee. It is not uniform in a limit where $\beta$ tends to zero or the potential becomes unbounded.

## 6. A defensible theorem-oriented research direction

Specify a correction class (for example, an ellipsoid $h^\top R h\le c$), the noise and numerical-error bounds, and the target norm. Prove stability and consistency as noise, grid error and regularization strength decrease together. A comparative rate claim then needs assumptions under which the proposed operator aligns with truth more favorably than H1 or H2. A better constant or rate on that class is meaningful; a universal claim across unknown truths is not justified by the current model.

Relevant primary work already proves regularization convergence results for local-volatility inverse problems: Albani, De Cezaro and Zubelli, *Convex Regularization of Local Volatility Estimation in a Discrete Setting*, [arXiv:1308.2659](https://arxiv.org/abs/1308.2659), and Albani and Zubelli, *Online Local Volatility Calibration by Convex Regularization with Morozov's Principle and Convergence Rates*, [arXiv:1211.0170](https://arxiv.org/abs/1211.0170). Their theorems must be checked against the proposed operator's domain, parameterization and assumptions before being extended. The finite-dimensional derivations in sections 1–5 above are supplied here directly; they are not a claim of a new published theorem.

