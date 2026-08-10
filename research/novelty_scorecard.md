# Novelty scorecard

This file is a decision document, not a claim of novelty. A result may be
described as a paper contribution only after it passes the prior-art and
validation gates below.

## Required dimensions

We use the supplied framework:

\[
N = D \times I \times O \times V, \qquad
\mathrm{ResearchValue}=N/C.
\]

- **D — difference:** distance from the closest paper, not the average paper.
- **I — impact:** relevance, expected gain, and general usefulness for chip
  reliability/sign-off.
- **O — non-obviousness:** whether the result would have been predicted from
  standard parameter-fitting or experiment-design practice.
- **V — validation:** seeds, ablations, distribution shifts, external data,
  statistics, and reproducibility.
- **C — complexity:** additional implementation and experimental burden.

Target gate for a strong workshop contribution: provisional D/I/O/V values of
at least 0.6/0.7/0.6/0.8, with the caveat that these numbers are only a
structured review aid, not an objective novelty measurement.

## Candidate discoveries

| Candidate | Current status | Required evidence |
|---|---|---|
| Accelerated EM data are deployment-non-identifying | **Strongest current candidate**: across three pathway settings and T≥623 K, 167 parameterizations differ by ≤0.15 log units in-window yet disagree by up to 15.97× at deployment | Analytical construction across windows/pathway settings, prior-art separation, external/digitized validation |
| Deployment-functional test design differs from parameter-information design | **Not established**: current six-temperature experiment selects 443 K for both policies | Broader candidate designs, constrained test budgets, distinct parameter-recovery objective, repeated statistical comparison |
| Safe-identifiability frontier over temperature/current/length coverage | Hypothesis | Phase diagram, worst-case error guarantee, cost-vs-safety curve |
| PDN topology improves interpolation but not deployment sign-off | Preliminary supporting result | Network-level guard-band metrics, unseen topologies, physics-aware baselines |

## Initial prior-art findings

The broad idea of optimizing electromigration accelerated tests is not new.
NASA's *Designing Accelerated Tests of Electromigration* explicitly studies the
tradeoff between high stress (shorter tests) and lower stress (lower
extrapolation uncertainty):
[NASA NTRS record](https://ntrs.nasa.gov/citations/19910000377).
Recent EM reliability work also reports optimized data collection and
calibration for network-level lifetime checking, including ML-based design-rule
checking: [Milor and Ghosh, 2023](https://doi.org/10.1016/j.microrel.2023.115163).
PDN reliability and redundancy are active topics, for example the recent
unit-cell/mesh reliability literature:
[EM in nano-interconnects](https://pmc.ncbi.nlm.nih.gov/articles/PMC11356743/).

Therefore, “we use Fisher information to choose an EM test point” is not a
novelty claim. Any acceptable contribution must be narrower and stronger, such
as a formally derived deployment-functional lower bound, a demonstrated
disagreement between conventional calibration and sign-off-risk objectives, or
a new PDN-level decision rule that is absent from these works.

## Hard acceptance gates

1. Search and document the closest prior work in EM reliability, optimal
   experiment design, active learning, uncertainty-aware extrapolation, and PDN
   reliability.
2. Do not use “novel,” “first,” or “discovery” until the closest-work matrix is
   complete.
3. Compare random, parameter-information, and deployment-functional policies
   under equal test cost.
4. Require at least 20 seeds, multiple noise/pathway settings, and confidence
   intervals.
5. Require at least one design-level metric: false-safe rate, vulnerable
   segment recall, network failure AUROC, or guard-band violation rate.
6. Prefer a simple method with a large safety gain; report novelty density
   rather than rewarding complexity.

## Current evidence

The 20-seed controlled experiment currently reports:

- accelerated-only median fold error: 5.35x;
- deployment-functional selection: 1.47x median;
- parameter-information selection: 1.47x median;
- random selection: 1.94x median;
- 2x-safe rates: 85%, 85%, and 50%, respectively.

The optimizer-independent frontier experiment adds a stronger result: with a
0.15 log-lifetime compatibility threshold, the maximum hidden deployment fold
disagreement is 1.53× for T≥523 K, 1.84× for T≥548 K, 2.75× for T≥573 K, and
15.97× for T≥623 K. The construction is repeated over low-, benchmark-, and
high-prefactor-ratio pathway settings. This is the current candidate discovery
because it is an analytical tradeoff family rather than an optimizer artifact;
it remains provisional until the literature search and external validation
gates pass.

The equality of the first two policies means the proposed distinction between
parameter-optimal and deployment-optimal design is **not yet a finding** on the
current candidate set. This is an intentional falsification result: the
candidate must be redesigned or replaced before it becomes the headline claim.

The robust certificate arm adds a scoped, falsifiable sign-off result. With a
fixed prediction-space excess budget of (5\sigma^2), the 20-seed median
compatible deployment disagreement is 91.8x after the accelerated fit and
88.4x after the certificate-selected single batch; all certificate, Fisher,
and random policies abstain under a 2x target. This supports the claim that
one additional batch is insufficient for this synthetic family, not a
universal claim about EM testing. It remains an empirical benchmark finding
until an external or digitized published test matrix confirms it.
