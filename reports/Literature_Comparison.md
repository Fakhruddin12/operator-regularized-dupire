# Closest literature and what this benchmark actually compares

Reviewed 22 October 2026. The computational study isolates the regularizer inside one shared SSVI-reference, tapered log-local-variance correction model. It is not a replication of every author's model or published optimizer.

| Work | Specific overlap | Difference from this project | Data used here |
|---|---|---|---|
| Crépey, *Calibration of the Local Volatility in a Generalized Black–Scholes Model Using Tikhonov Regularization* (2003) | Nonlinear option-price inversion with an H1 penalty on deviation from a reference; stability and convergence theory. | Its diffusion-coefficient coordinates and functional assumptions differ from the tapered latent log-variance parameterization and density/wing potentials. | A conventional H1 penalty adapted to our coordinates; not a published-data replication. |
| Albani, Ascher, Yang and Zubelli, *Data driven recovery of local volatility surfaces* (2017; 2015 preprint) | Sparse observations calibrated directly through a PDE, with magnitude and directional smoothness penalties. | Their spot/location adjustment and data-completion comparison are not reproduced. | Published synthetic truth and observation layout, newly generated prices and ten explicitly recorded noise seeds. |
| Geng, Navon and Chen, *Non-parametric calibration ... using a second-order Tikhonov regularization* (2014) | Nonparametric PDE inversion with second-order smoothing. | H2 is applied to our latent correction, not their original volatility representation; lambda selection differs. | Public 155-quote Euro Stoxx benchmark transcription; controlled H2 comparator. |
| Ackerer, Tagasovska and Vatter, *Deep Smoothing of the Implied Volatility Surface* (2020) | A classical implied-volatility model plus a learned multiplicative correction. | Their correction is a neural implied-volatility model with arbitrage penalties; ours changes log local variance and reprices via Dupire. | Their unchanged 324-row synthetic Bates training file. Their neural model is not trained in this audit. |

The nearest mathematical predecessor is therefore reference-centered H1/Tikhonov regularization; the reference-plus-correction architecture also has clear precedent. The added spatial potential changes where the correction is shrunk. Its particular construction and its empirical benefit are separate questions. Describing the quadratic operator as Schrödinger-like does not alone establish a new estimation method.

No exact published match for this complete configuration was established by this review. That is not proof of novelty or an exhaustive priority search. The controlled H1, H2, density-only, wing-only and full-operator ablations measure whether the specific added potentials help under the declared model and tuning procedure.

## Theory with direct relevance

Crépey's paper already proves stability and convergence under its stated hypotheses; a convergence theorem alone would not distinguish this project. Albani, De Cezaro and Zubelli's *Convex Regularization of Local Volatility Estimation in a Discrete Setting* explicitly separates noise and discretization effects. Albani and Zubelli's online-calibration work supplies discrepancy-based parameter-choice and convergence-rate theory. These are useful starting points for checking assumptions, rather than evidence of comparative dominance. The separate `Regularizer_Theory.md` gives a direct finite-dimensional risk comparison and counterexample.

## More recent adjacent work

The abstract of Chen, Dai, Yang and Yang's *Calibration of Local Volatility Models under the Implied Volatility Criterion* (2024) describes PDE/Tikhonov calibration using an implied-volatility criterion. Full text could not be retrieved in this check; it has not been inspected for an exact operator match or reproduced. This is relevant to interpreting the weighting: our vega scale approximates an IV loss locally, but its price floor and finite residuals make it a different criterion.

Yoo et al., *Calibration of Local Volatility Surfaces from Observed Market Call and Put Option Prices* (2025), optimize scattered-interpolant values and locations without a Tikhonov penalty, according to the authors' institutional abstract. This is an adjacent adaptive-parameterization comparator rather than the closest regularizer. Its numerical experiments include Euro Stoxx among other indices; access to the exact dataset and an implementation was not established here.

## Primary sources

- Crépey: [author-hosted full text](https://math.maths.univ-evry.fr/crepey/papers/40020SIMA.pdf), especially section 2.3; [DOI](https://doi.org/10.1137/S0036141001400202).
- Albani et al.: [arXiv:1512.07660](https://arxiv.org/abs/1512.07660); [2017 journal article](https://doi.org/10.3934/ipi.2017038).
- Geng et al.: [DOI](https://doi.org/10.1080/14697688.2013.819988); [author-hosted paper](https://people.sc.fsu.edu/~inavon/pubs/localpaper2.pdf).
- Ackerer et al.: [NeurIPS paper](https://proceedings.neurips.cc/paper/2020/hash/858e47701162578e5e627cd93ab0938a-Abstract.html); [official supplement](https://papers.neurips.cc/paper_files/paper/2020/file/858e47701162578e5e627cd93ab0938a-Supplemental.zip).
- Discrete convergence theory: [arXiv:1308.2659](https://arxiv.org/abs/1308.2659).
- Online convergence theory: [arXiv:1211.0170](https://arxiv.org/abs/1211.0170).
- Chen et al.: [author-submitted SSRN abstract](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4801520).
- Yoo et al.: [author institution's abstract and bibliography](https://pure.korea.ac.kr/en/publications/calibration-of-local-volatility-surfaces-from-observed-market-cal-2/); [DOI](https://doi.org/10.1007/s10614-024-10590-9).

The input files' source URLs, retrieval date, exact observation counts and SHA-256 hashes are retained under `data/literature/benchmark_inputs/provenance.json`.
