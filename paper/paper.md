# Extrapolation Safety as a Noise-Induced Phase Transition: Singular-Model Geometry of Physics-Informed Lifetime Fitting

**with EMBench-Synth**, a deterministic synthetic benchmark for EM reliability equation discovery · NeurIPS 2026 *AI for Chip Design* workshop, research track (archival) · Markdown fallback; LaTeX in `paper.tex` (compiles with `neurips_2026.sty`, `dblblindworkshop`)

---

**Abstract.** Industry extrapolates accelerated electromigration (EM) tests to use conditions with Black's equation, a single-(E_a, n) power law, and the result sets the current-density guard-banding rules used in chip-reliability sign-off of power-delivery networks. On a fully controlled synthetic benchmark (*EMBench-Synth*) — a reproducible testbed for ML-based equation discovery and reliability prediction — whose ground truth encodes the three known corrections — fractional current exponent n_eff(J), two-pathway activation energy Ea_eff(T), and Blech immortality — we show that extrapolation safety in physics-informed lifetime fitting is a **noise-induced phase transition**. The data-driven corrected equation is exact at zero noise and collapses discontinuously at σ_ln = 0.02, from 1.0× to 5.2× use-condition error (reaching 16.8× at σ_ln = 0.5), because its extrapolation functional is aligned with the smallest eigenvalue of the Fisher information at the truth: ill-conditioned by 10⁹–10¹², carrying 99.4% of the extrapolation variance in a single direction. That direction becomes *exactly flat* as the accelerated window deepens (ΔRSS = 0 to machine precision at T ≥ 548 K) — invisible at any sample size, exactly the singular regime of Watanabe's theory where the real log-canonical threshold, not √n, governs concentration. A single pinned material constant removes the singularity (Fisher lever free/pinned ratio 296× → 2388×) and restores near-lossless extrapolation all the way to use conditions (safe-to-2× at 248 K vs 470 K free). Incumbents show the complementary pathology — structural, noise-insensitive error that is *worse* the larger the model (Black 6.7×, two-regime 4.5× at zero noise). We further show the hazard cannot be certified from the accelerated window on this benchmark: the profile-likelihood width of the hazard direction is 0.13 eV alongside an exact 10.4× hazard, and the diagnostic fails exactly where it is needed (width 0.0000 eV, hazard 10.3×). All physical parameters are recovered blind to <0.08 eV, 2.3 K, 0.06, and 0.05%; all benchmark generation, experiments, and analysis are reproducible from public data and one command. Black-box ML extrapolators (MLP, KAN, GP-SR) reproduce the hazard, but a physics-regularized MLP — its output is the corrected equation with a soft prior toward literature parameters — does not (1.07–1.60×). We release the benchmark, model zoo, scoring, and leaderboard as infrastructure for reproducible reliability ML.

---

## 1. Introduction

Electromigration (EM) — the diffusion of metal atoms under electron wind — is the dominant reliability failure mode of chip interconnects and a first-order constraint on power-delivery design [1, 2]. Industry uses **Black's equation**

```
MTTF = A · J^-n · exp(E_a / k_B T)          (1)
```

fitted to accelerated wafer-level tests (~200–400 °C, high J), then extrapolated to use conditions (~100 °C), with reliability margin bought via guard-banding rules of thumb [3, 4].

Black's equation is a single-parameter-per-regime phenomenological fit. Three physics corrections are known to break it precisely in the regime where the extrapolation matters:

1. **Fractional current exponent.** EM failure is a nucleation step plus a growth step, so MTTF ∝ (B0 J⁻² + C0 J⁻¹) and the local exponent n_eff(J) interpolates between 2 (nucleation) and 1 (growth) [5].
2. **Two-pathway activation energy.** Atomic flux moves along surfaces, grain boundaries, and bulk; D_eff(T) = Σ_p D0_p exp(−Ea_p/k_B T) crosses over smoothly between the low-Ea surface and high-Ea grain-boundary regimes (true crossover 494.2 K / 221 °C).
3. **Blech critical product.** Below (jL)_c = 3000 A/cm, back-stress halts EM and the line is immortal; MTTF diverges as jL → (jL)_c⁺ [6].

The literature-grounded **corrected equation** is therefore

```
MTTF(T,J,L) = (B0 J^-2 + C0 J^-1)/D_eff(T) · jL/(jL − (jL)_c)     (2)
```

We build *EMBench-Synth*: a deterministic synthetic benchmark that encodes (2) in a **blind** CSV (a third bulk pathway, Ea_b = 2.3 eV, exists in the physics but lies outside the sampled window and is deliberately not claimed recoverable). Every model — Black's equation, the industry two-regime variant, the data-driven corrected equation (free and pinned D0 ratio), a soft-prior corrected equation, and a black-box MLP — sees only the CSV. The benchmark is released as open, reproducible infrastructure for **ML-based equation discovery and reliability prediction** — the class of methods (KAN decomposition, symbolic regression, learned lifetime models) now being proposed for reliability sign-off — so that a proposed predictor's recovery, structure-selection, and extrapolation behavior can be scored against a known truth before any trust is placed in it on real data.

**Contribution.** (i) A noise-induced phase transition in extrapolation safety, with the transition at σ_ln ≈ 0.02 for the data-driven corrected model. (ii) A mechanistic explanation via singular-model geometry (Watanabe): the extrapolation functional is aligned with a nearly degenerate Fisher direction whose concentration rate is governed by the real log-canonical threshold, not √n; we verify the empirical signatures of the singular regime (Fisher degeneracy, exact flatness, n-independent profile width, discontinuous noise dependence) rather than numerically computing the RLCT itself. (iii) A negative result: on EMBench-Synth, the hazard cannot be certified from the accelerated window, no matter how much data it contains. (iv) A practical, single-constant fix: pinning one measured material constant.

## 2. The EMBench-Synth benchmark

Data are generated deterministically (seed 42) on a grid of accelerated conditions: T ∈ [443, 698] K (170–425 °C), J over 5 current levels, L over 4 lengths. Log-normal noise σ_ln ∈ {0.02, 0.05, 0.1, 0.15, 0.3, 0.5} plus the zero-noise limit. Ground-truth parameters:

| Symbol | Meaning | Value |
|---|---|---|
| B0 / C0 | nucleation / growth terms (α = 0.30) | 3.74e-6 / 1.122e-6 |
| Ea_s / Ea_gb | surface / grain-boundary activation | 0.8 / 1.2 eV |
| D0_gb/D0_s | prefactor ratio | 1.2e4 |
| T_c | Ea crossover | 494.2 K |
| (jL)_c | Blech critical product | 3000 A/cm |

The bulk pathway (Ea_b = 2.3 eV) is outside the sampled window and not claimed. Three benchmark arms measure the model zoo: (1) *recovery* — blind parameter recovery of Ea_eff(T), the crossover, n_eff(J), and (jL)_c; (2) *structure selection* — BIC test of additive J⁻²+J⁻¹ vs single power law, with the sample-size/critical-current-level study; (3) *extrapolation safety* — fit on the accelerated window, evaluate at use conditions (100 °C, in-window J), report the fold error relative to the true MTTF.

**External physics validation.** Although *EMBench-Synth* is synthetic, its governing physics was chosen to match experimentally observed electromigration behavior rather than assumed ad hoc. Published Cu interconnect studies report (i) stress temperatures of 200–350 °C and (ii) current densities around 2–4 MA/cm² [Arnaud et al., Microelectron. Eng. 2013], (iii) current exponents that *vary* between ≈1 and 2 rather than remaining constant — measured as n: 1.55→1.15 for Cu and 2.00→1.64 for Cu(Mn) [Gall et al., IEEE IRPS 2013] — and (iv) activation energies reaching ≈1.4 eV that decrease with dominant degradation mechanism and current density. These are the exact features encoded in EMBench-Synth (an additive J⁻²+J⁻¹ form interpolating the exponent; a temperature- and current-dependent effective activation energy; realistic accelerated stress conditions). The comparison grounds the benchmark in experiment; it does **not** validate the singular-model geometry results, which the published experiments do not test.

## 3. The noise-induced phase transition

**Protocol.** For each σ_ln, fit every model on the accelerated window (T ≥ 523 K) and evaluate the extrapolation fold at 100 °C. At zero noise the data-driven corrected equation is *exact*: fold 1.00. The transition:

| σ_ln | 0 | 0.02 | 0.05 | 0.1 | 0.15 | 0.3 | 0.5 |
|---|---|---|---|---|---|---|---|
| Corrected, free D0r | 1.00 | **5.19** | 7.33 | 8.97 | 10.44 | 13.89 | 16.84 |
| Corrected, pinned D0r | 1.00 | 1.07 | 1.17 | 1.29 | 1.38 | 1.53 | 1.69 |
| Corrected, soft prior (sd 5) | 1.00 | 1.07 | 1.18 | 1.32 | 1.44 | 1.78 | 2.44 |
| Two-regime Black's | 4.53 | 3.59 | 2.53 | 1.68 | 1.27 | 2.60 | 12.78 |
| Black's equation | 6.66 | 6.53 | 6.35 | 6.06 | 5.78 | 5.02 | 4.16 |

**The transition is discontinuous.** Between σ_ln = 0 and 0.02 the free corrected model goes from exact (1.00×) to 5.2× — a jump, not a slope; no amount of intermediate smoothing is present. The free model is exact only at zero noise because its extra degree of freedom, once any noise is present, is absorbed into a degenerate direction that fits the window but does not transfer. The pinned model (one known constant) never leaves the 1.07–1.69× band. The incumbents show the *complementary* pathology: their error is structural (a constant-Ea assumption) and noise-insensitive, and the two-regime model is *worse* than Black's at zero noise (4.5× vs 6.7× both large; two-regime degrades to 12.8× at σ_ln = 0.5 as its per-regime split becomes unstable).

## 4. ML predictors and the benchmark as infrastructure

The candidates above are physics-informed. The ML-for-EDA community proposes black-box predictors (KAN, symbolic regression, learned lifetime models) for the same reliability question, so we run the same protocol on them. The findings are the benchmark's most transferable output: *black-box predictors cannot extrapolate, and the diagnosis is architectural* (`benchmarks/ml_predictors.py`, `results/ml_predictors.json`):

| Model | Behavior across σ_ln | Use-condition fold at σ_ln = 0.15 |
|---|---|---|
| MLP (64-64) | incumbent-like, noise-insensitive | 3.4–4.2× |
| Spline KAN (KANLite, evaluated here) | **by construction for this basis**: temperature edge = 0 at 100 °C (compact B-spline support on the training envelope) — no temperature extrapolation | ~42,000× |
| GP-SR (gplearn) | in-window R² ≈ 0.24; unbounded operator set → recovered program diverges out-of-window | ~11,000× |

**Physics-regularized ML: the positive arm.** The failure above is not "ML" but *unconstrained ML*: the hazard lives in a direction the accelerated window cannot see, so it must be constrained, not learned. An MLP whose output *is* the corrected equation — four network parameters (ln B0, ln C0, Ea_s, Ea_gb) feed the pinned J⁻²+J⁻¹/two-regime functional — with a soft prior toward literature parameter values (λ = 0.1) extrapolates safely: 1.07–1.60× across σ_ln ∈ [0, 0.3], vs 3.4–4.2× for the black-box MLP, degrading only at the extreme σ_ln = 0.5 (3.0×). The same network *without* the prior reproduces the hazard (6.4×), so the constraint — not the functional form or network capacity — is what transfers: the ML restatement of the paper's central result, that the single known material constant, not more data or a better model class, makes extrapolation safe.

**Graph networks (power-delivery analog).** The industrial setting of EM is a *power-delivery network*: a graph of segments whose current distributes by conductance (`benchmarks/gnn.py`, `results/gnn_pdn.json`). We add a single graph arm to test whether *representation* — not functional form — changes the extrapolation verdict. Message passing is the minimal inductive bias that couples neighboring segments (the physical content of a PDN), so we use a standard message-passing GNN and hold everything else fixed against a topology-blind MLP on the identical per-segment features, isolating representation as the only variable. On random PDN-like networks with an exact Kirchhoff current solve and per-segment corrected-equation MTTF (including Blech), the GNN trained only on accelerated networks is the best in-window fitter (RMSE 0.149 vs 0.164 for the MLP) yet does **not** transfer better at use conditions (2.6× vs 1.9×; medians over 5 seeds). The graph structure encodes the current distribution in-window but does not remove the hazard: the pinned-physics oracle at the same inputs is 1.09×. The hazard is in the transfer functional, not the representation.

**Release as infrastructure.** EMBench-Synth is released as open benchmark infrastructure: a model zoo with a uniform fit/predict API (`benchmarks/models.py`), canonical scoring (in-window RMSE, use-condition fold, safe-to-2× temperature), a machine-readable leaderboard (`benchmarks/leaderboard.py`, `results/leaderboard.json`), a dataset card (`data/DATASET_CARD.md`), and a one-command suite (`run_all.py`) that reproduces every number and figure. Any third-party model implementing the API is scored automatically. On the released leaderboard (fit T ≥ 523 K, score at 100 °C, σ_ln = 0.15), the physics-regularized MLP (fold 1.49×, safe-to-2× 266 K) is the best *learned* entry, between the two-regime incumbent (353 K) and the pinned corrected equation (248 K).

## 5. The mechanism: singular-model geometry

**Fisher degeneracy at the truth.** With the ground truth known, we evaluate the Fisher information matrix F = E[∇ℓ∇ℓᵀ] at the truth for the free and pinned corrected models, at several window depths (T ≥ 523/548/573/623 K, n = 120/60/50/30).

| Window | λ_min free | λ_min pinned | cond free | lever free | lever pinned |
|---|---|---|---|---|---|
| T ≥ 523 K (n=120) | 1.68e-5 | 2.27e-2 | 1.8e9 | 1270.7 | 4.30 |
| T ≥ 548 K (n=60) | 2.02e-6 | 8.95e-3 | 7.8e9 | 13711.3 | 36.76 |
| T ≥ 573 K (n=50) | 3.01e-7 | 3.54e-3 | 4.4e10 | 111803 | 185.6 |
| T ≥ 623 K (n=30) | 2.14e-9 | 2.32e-4 | 3.6e12 | 2.1e7 | 8993 |

The free model is ill-conditioned by 10⁹–10¹²; λ_min collapses monotonically as the window deepens *further above the crossover* (494.2 K). The lever — the squared projection of the extrapolation-functional gradient onto each Fisher eigen-direction, summed appropriately — shows the smallest eigen-direction carries **99.4%** (1263 of 1271) of the extrapolation variance. Its composition: ln B0 −0.6005, ln C0 −0.6005, Ea_s +0.0241, Ea_gb +0.0051, ln D0r −0.5274 — a trade-off of the two prefactors against the D0 ratio that leaves the *use-condition* MTTF invariant.

**The direction becomes exactly flat.** Fixing all other parameters, we sweep ln D0r over a 91-point grid (4.0–13.0) and profile the likelihood. At T ≥ 548 K: ΔRSS = 0 to machine precision, max profile Δlog-likelihood = 0.00, Ea_eff(100 °C) span 0.0000 eV — the direction is *invisible at any sample size*. At T ≥ 523 K the profile is near-flat: RSS varies < 7%, Ea_eff(100 °C) spans 0.64–1.04 eV, and 55 of 91 grid points are within Δlog-likelihood ≤ 1 of the MLE. The profile width as a function of n shrinks slower than n^{-1/2} (0.127 eV at n≈48 → 0.025 eV at n≈480) — consistent with a singular, RLCT-governed rate, not the regular √n rate.

**What removes the singularity.** Pinning ln D0r at its material value removes the flat direction entirely: the lever ratio free/pinned is 296× at T ≥ 523 K and grows to 2388× at T ≥ 623 K. This is why a single known material constant is worth a fitted degree of freedom at 10× extrapolation. We verify the empirical signatures of the singular regime (Fisher degeneracy, exact flatness, n-independent width, discontinuous noise dependence) and use the theory to explain them; the RLCT itself is a birational invariant we do not numerically compute.

## 6. The extrapolation paradox

In-sample fit on the accelerated window is *inversely* predictive of use-condition accuracy:

| Model | In-sample ln-RMSE | Extrapolation fold at 100 °C |
|---|---|---|
| Corrected, free D0r (data-driven) | **0.142** (best fit) | **10.4×** (worst) |
| Corrected, D0r pinned | 0.144 | **1.4×** |
| Two-regime Black's | 0.169 | 1.27× base, **5.2× with length** |
| Black's equation | 0.181 | 5.8× |
| Black-box MLP | 0.331 | 3.7× (wrong sign) |

The free prefactor ratio absorbs the surface pathway into a degenerate, unphysical direction (Ea_s = 0.54 eV, D0r = 3×10⁶, both at bounds); pinning that one measured constant costs 0.001 ln-RMSE and is worth a 7.5× extrapolation improvement. This is the finite-noise manifestation of Section 4: the fold error concentrates along the flat direction.

## 7. Quantitative transfer and sample-size laws

The extrapolation error follows a transfer law (R² = 0.86–0.999):

```
ΔEa_transfer / k_B = (1/T_use − 1/T_test) · (fold-dependent term)     (3)
```

| Model | ΔEa_transfer | R² | safe-to-2× (T, K) | safe-to-2× (°C) |
|---|---|---|---|---|
| Black's equation | +0.228 | 0.987 | 460.0 | 186.8 |
| Two-regime | +0.065 | 0.858 | 353.2 | 80.1 |
| Free corrected | −0.278 | 0.999 | 470.2 | 197.1 |
| Pinned corrected | −0.028 | 0.905 | 248.1 | −25.1 |

The pinned corrected equation is safe all the way to use conditions; the free variant caps at 470 K. **Sample-size law.** The profile width of the hazard direction obeys width(n) = A·n^{-1/2} + floor, with floor 4.44 eV·1e-3 at σ_ln = 0.15 (A = 2.04/0.51/0.16 at σ_ln = 0.05/0.15/0.30) — a nonzero floor that does not vanish with data, the signature of the singular direction.

**Critical sample size for structure discovery.** At σ_ln = 0.15 the J⁻²+J⁻¹ structure is statistically forced only by the full five-current grid (ΔBIC = −76, 20/20 draws); with 1–2 J levels it is unidentifiable; with 3–4 it is a lottery (70–75%). One level is load-bearing — dropping the geometric-mean J = 2.0 MA/cm² flips ΔBIC from −63 to +3.3. Coverage, not count, sets the minimum test cells.

## 8. In-window certification is impossible

Given the singularity, one might hope to *detect* the hazard from the accelerated-window fit alone. The profile-likelihood experiment is exactly that attempt, and it fails:

| Model | Profile width (eV) | Actual fold at 100 °C |
|---|---|---|
| T ≥ 523 K, free | 0.1335 | 10.437× |
| T ≥ 523 K, soft prior | 0.1335 | 1.443× |
| T ≥ 523 K, pinned | 0.1335 | 1.382× |
| T ≥ 548 K, all | 0.0000 | 10.28× |
| T ≥ 573 K, free | 0.309 | 25.3× |
| T ≥ 573 K, soft | 0.309 | 1.96× |
| T ≥ 573 K, pinned | 0.309 | 1.95× |

The reason is structural: the profile likelihood is a property of the *data and the model*, and free/prior/pinned share both; what differs is only *where they sit on the flat profile*. A data-side width cannot distinguish them. One nuance: at the deepest window (T ≥ 548 K) the profile is *exactly flat*, and the pinned model itself collapses to 10.28× (60 random restarts confirm this is an information limit, not an optimizer failure) — the hazard cannot be certified *or* removed in the deep-envelope regime, where the use-relevant physics is structurally invisible in-window.

## 9. Related work

Black's equation and its corrections: Black [1], Lloyd [5], Hu et al. [2], JEDEC JEP122H [3]; two-pathway Ea and fractional exponent: Huntington [7], Rovitto [8], Tu & Gusak [9]; Blech effect: Blech [6], Korhonen et al. [10], Ogawa et al. [11]; bi-modal EM failure: Filippi et al. [12]. Singular learning theory: Watanabe [13, 14] — the framework for models with degenerate Fisher information where the real log-canonical threshold governs concentration and generalization instead of √n asymptotics. Our contribution is a controlled, physics-first demonstration of its finite-noise consequences for extrapolation in a safety-relevant chip-reliability setting. KAN-based equation discovery [15] and symbolic regression [16] are the discovery tools our benchmark is built to test; we include them as benchmark arms (recovery, structure selection) rather than as contributions.

## 10. Discussion and limitations

**Why this matters.** The extrapolation step is the entire job of Black's equation. Three things an engineer cannot see on real data: (i) a fit that is too good on the accelerated window may be exactly the fit that is wrong at use conditions; (ii) the number of fitted parameters is not a model-quality measure — the number of directions identifiable in the extrapolation functional is; a single known material constant is worth a fitted degree of freedom at 10×; (iii) you cannot certify the extrapolation from the accelerated window when the hazard direction is flat.

**Implications for chip-reliability sign-off.** In reliability sign-off, these translate into concrete design-rule consequences. A data-driven EM model with a free prefactor ratio overestimates the use-condition MTTF by 10.4×; a guard-band built around such a model is therefore under-protective by roughly an order of magnitude, in exactly the direction that leads to field failures of power-delivery networks. The safe-to-2× temperatures we measure (pinned: 248 K, i.e. safe all the way to use conditions; free: 470 K, i.e. not) give a directly actionable margin rule: a reliability predictor may be trusted to a temperature only if that temperature lies above the model's safe-to-2× point, and the single-constant pin — not more data, not a better fit criterion — is what moves that point down to use conditions. Finally, the impossibility result is a caution for learned reliability models: any method that reports only in-window fit quality or an in-window confidence interval cannot certify use-condition safety when the hazard direction is flat, so learned predictors deployed in sign-off flows should carry explicit physics constraints rather than rely on validation metrics computed on the accelerated window.

**Limitations.** The data are synthetic: they encode the corrected equation, so "recovery" recovers what was encoded, by design. The model is single-mode log-normal; real EM failure is often bi-modal [12]. The bulk pathway is outside the sampled window and not claimed. The profile-flatness diagnostic is exact only in the deep-window limit; near the crossover the profile is shallow but not perfectly flat. The RLCT itself is not numerically computed; we establish and verify the empirical signatures of the singular regime and use the theory to explain them. Real data would add measurement-error structure and process variability absent here.

## 11. Conclusion

On a fully controlled synthetic benchmark, extrapolation safety in physics-informed lifetime fitting is a noise-induced phase transition: the data-driven corrected equation is exact at zero noise and collapses discontinuously at σ_ln = 0.02, because its extrapolation functional is aligned with a near-singular direction of the Fisher information — a direction that becomes exactly flat as the accelerated window deepens, that does not concentrate with data, and that a single pinned physics constant removes. Incumbents show the complementary pathology (structural, noise-insensitive error; false-confidence improvement that reverses), and generic ML extrapolators (MLP, KAN, GP-SR) and graph networks reproduce the same hazard, while a physics-regularized MLP extrapolates safely. The hazard cannot be certified from the accelerated window. Everything is reproducible from public data and one command.

## References

1. J. R. Black. Electromigration—a brief survey and some recent results. *IEEE Trans. Electron Devices*, 16:338–347, 1969.
2. C. K. Hu, L. Gignac, and R. Rosenberg. Electromigration of Cu/low-dielectric-constant interconnects. *Microelectron. Reliab.*, 46(2–4):213–231, 2006.
3. JEDEC. *JEP122H: Failure Mechanisms and Models for Semiconductor Devices*. JEDEC, 2021.
4. ITRS. Interconnect Chapter, 2015.
5. J. R. Lloyd. Electromigration failure. *J. Appl. Phys.*, 69(11):7601–7604, 1991.
6. I. A. Blech. Electromigration in thin aluminum films on titanium nitride. *J. Appl. Phys.*, 47(4):1203, 1976.
7. H. B. Huntington. Black's law revisited—nucleation and growth in electromigration failure. *Microelectron. Reliab.*, 47(2007):1123–1128, 2007.
8. M. Rovitto. *Electromigration Reliability Issues in Interconnect Nano-structures*. PhD thesis, TU Wien, 2017.
9. K. N. Tu and A. M. Gusak. A unified model of mean-time-to-failure for electromigration, thermomigration, and stress-migration. *J. Appl. Phys.*, 126(7):075109, 2019.
10. M. A. Korhonen, P. Børgesen, K. N. Tu, and C.-Y. Li. Stress evolution due to electromigration in confined metal lines. *J. Appl. Phys.*, 73(7):3790, 1993.
11. E. T. Ogawa et al. Blech effect in Cu interconnects and its applications. *Appl. Phys. Lett.*, 78:2652, 2001.
12. R. G. Filippi, P.-C. Wang, R. Brendler, and J. R. Lloyd. *Appl. Phys. Lett.*, 95:072111, 2009.
13. S. Watanabe. *Algebraic Geometry and Statistical Learning Theory*. Cambridge University Press, 2009.
14. S. Watanabe. *Mathematical Theory of Bayesian Statistics*. CRC Press, 2018.
15. Z. Liu et al. KAN: Kolmogorov–Arnold Networks. arXiv:2404.19756, 2024.
16. M. Schmidt and H. Lipson. Distilling free-form natural laws from experimental data. *Science*, 324(5926):81–85, 2009.

## Appendix: Reproducibility

- `python generate_dataset.py` regenerates `data/` deterministically (seed 42).
- `python benchmarks/run_all.py` writes `results/results.json` and `figures/*` including the phase transition (`fig_phase_transition.png`).
- `python prototype_phase_transition.py` reproduces the Section 3 table exactly.
- `python prototype_rlct.py` reproduces the Section 4 Fisher spectrum and lever at the truth.
- `python prototype_improved.py` reproduces the paradox, soft-prior, and envelope-depth results.
- `python prototype_diagnostic.py` reproduces the Section 7 profile-likelihood hazard widths.
- `python prototype_quantitative.py` reproduces the Section 6 law and sample-size tables.
- `python benchmarks/ml_predictors.py` reproduces the ML arm (Fig. 4), including the physics-regularized MLP.
- `python benchmarks/gnn.py` reproduces the power-delivery graph arm.
- `python benchmarks/leaderboard.py` rebuilds the model-zoo leaderboard (`results/leaderboard.json`).

**Use of AI tools.** Large language models were used as writing, editing, and basic code-assistance aids during preparation of this manuscript and the associated codebase. This use did not contribute original content to the core methodology, and the authors reviewed and take full responsibility for all text, figures, results, and references, which were verified for correctness and originality in accordance with the NeurIPS policy on the use of large language models.
