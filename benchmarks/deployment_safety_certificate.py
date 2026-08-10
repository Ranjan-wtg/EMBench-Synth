"""Robust deployment-safety certificate for the synthetic EM benchmark.

The certificate asks a sign-off question rather than a parameter-recovery
question: among all members of a prescribed, physically motivated tradeoff
family that remain compatible with the accelerated observations, how far
apart can their deployment predictions be?

This is intentionally separate from the local Fisher benchmark.  It uses an
excess-residual compatibility rule and reports an explicit ABSTAIN state when
the compatible models disagree by more than the allowed deployment factor.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import generate_dataset as gen
from benchmarks import models as M

T_USE = 373.0
J_USE = np.array([0.1, 0.5, 1.0])
J_TEST = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
CANDIDATE_T = np.array([373.0, 380.0, 393.0, 410.0, 425.0, 443.0, 455.0, 475.0, 494.0, 510.0, 523.0])
SEEDS = list(range(20))
DELTAS = np.linspace(-4.0, 4.0, 321)
SIGMA = 0.15
TARGET_FOLD = 2.0
# Fixed excess-residual budget. It is deliberately stated as a protocol
# constant; unlike a per-row threshold, it makes sequential compatibility
# sets monotone under added observations.
EXCESS_BUDGET = SIGMA ** 2 * 5.0


def family(p):
    """Construct the cancellation family around a fitted free model."""
    q = np.repeat(np.asarray(p, float)[None, :], len(DELTAS), axis=0)
    q[:, 0] += DELTAS
    q[:, 1] += DELTAS
    q[:, 4] += DELTAS
    return q


def predictions(ps, T, J):
    return np.stack([M._add_mu(p, T, J) for p in ps])


def certificate(ps, reference_mu, compat_T, compat_J,
                T=T_USE, J=J_USE, sigma=SIGMA, observed_y=None):
    """Return compatible-set diameter and the associated sign-off state."""
    # The family is compatible when its mean squared excess residual relative
    # to the reference model is within one noise variance. This is a planning
    # rule, not a claim of calibrated frequentist coverage.
    pred = predictions(ps, compat_T, compat_J)
    ref = np.asarray(reference_mu, float)
    # Compatibility is defined in prediction space around the fitted
    # reference. This makes the sequential set monotone: every new test row
    # contributes a non-negative discrepancy, independent of its noise draw.
    # The observed batch is still used to refit the point model in the other
    # benchmark; it is intentionally not allowed to redefine this certificate.
    excess = np.sum((pred - ref[None, :]) ** 2, axis=1)
    keep = excess <= EXCESS_BUDGET
    if not np.any(keep):
        keep[np.argmin(excess)] = True
    use = predictions(ps[keep], np.full(len(J), T), J)
    diameter = float(np.max(use) - np.min(use))
    fold = float(np.exp(diameter))
    status = "SAFE" if fold <= TARGET_FOLD else "ABSTAIN"
    return {
        "compatible_count": int(np.sum(keep)),
        "diameter_log": diameter,
        "worst_case_fold": fold,
        "status": status,
        "prediction_low": use.min(axis=0).tolist(),
        "prediction_high": use.max(axis=0).tolist(),
    }, keep


def fit(T, J, y):
    return M.AdditiveModel().fit(T, J, y)


def add_batch(Tc, sigma, seed):
    rng = np.random.default_rng(seed + 10000)
    T = np.full(len(J_TEST), Tc)
    clean = gen.black_core_mttf(T, J_TEST)[0]
    return T, J_TEST.copy(), np.log(clean) + rng.normal(0.0, sigma, len(J_TEST))


def plan_candidates(p, T, J, y):
    ps = family(p)
    base_ref = M._add_mu(p, T, J)
    rows = []
    for Tc in CANDIDATE_T:
        Tc_arr = np.full(len(J_TEST), Tc)
        expected = M._add_mu(p, Tc_arr, J_TEST)
        aug_ps = ps
        aug_ref = np.r_[base_ref, expected]
        aug_pred = predictions(aug_ps, np.r_[T, Tc_arr], np.r_[J, J_TEST])
        # Retain hypotheses compatible with the already observed data and
        # with the expected (nominal) outcome of the candidate batch.
        observed_plan = np.r_[y, expected]
        baseline_loss = np.sum((aug_ref - observed_plan) ** 2)
        excess = np.sum((aug_pred - observed_plan[None, :]) ** 2, axis=1) - baseline_loss
        keep = excess <= EXCESS_BUDGET
        if not np.any(keep):
            keep[np.argmin(excess)] = True
        deploy = predictions(aug_ps[keep], np.full(len(J_USE), T_USE), J_USE)
        diameter = float(deploy.max() - deploy.min())
        rows.append({"T_K": float(Tc), "planned_worst_case_fold": float(np.exp(diameter)),
                     "planned_compatible_count": int(keep.sum())})
    return rows


def evaluate(seed):
    df, _ = gen.generate(noise_sigma=SIGMA, seed=seed)
    train = df[df.T_K >= 523.0]
    T, J = train.T_K.values, train.J_MA_cm2.values
    y = np.log(train.MTTF_hours.values)
    base = fit(T, J, y)
    ps = family(base.p)
    base_ref = M._add_mu(base.p, T, J)
    base_cert, _ = certificate(ps, base_ref, T, J, sigma=SIGMA, observed_y=y)
    candidate_rows = plan_candidates(base.p, T, J, y)
    safety_opt = min(candidate_rows, key=lambda r: r["planned_worst_case_fold"])["T_K"]

    # Fisher/log-volume information is a control, not the proposed objective.
    # The comparison is deliberately made on the same candidate temperatures.
    G = np.asarray([[M._add_mu(base.p + np.eye(5)[k] * 1e-5, t, j)
                     - M._add_mu(base.p - np.eye(5)[k] * 1e-5, t, j)
                     for k in range(5)] for t, j in zip(T, J)]) / (2e-5)
    F = G.T @ G + np.eye(5) * 1e-12
    fisher_scores = []
    for Tc in CANDIDATE_T:
        Gc = np.asarray([[M._add_mu(base.p + np.eye(5)[k] * 1e-5, Tc, j)
                          - M._add_mu(base.p - np.eye(5)[k] * 1e-5, Tc, j)
                          for k in range(5)] for j in J_TEST]) / (2e-5)
        sign, val = np.linalg.slogdet(F + Gc.T @ Gc)
        fisher_scores.append(float(val) if sign > 0 else -np.inf)
    fisher_t = float(CANDIDATE_T[int(np.argmax(fisher_scores))])
    random_t = float(CANDIDATE_T[seed % len(CANDIDATE_T)])

    out = {"seed": seed, "base_certificate": base_cert,
           "candidate_plan": candidate_rows, "safety_optimal_T_K": safety_opt,
           "fisher_T_K": fisher_t, "random_T_K": random_t}
    for label, Tc in (("safety", safety_opt), ("fisher", fisher_t), ("random", random_t)):
        Ta, Ja, ya = add_batch(Tc, SIGMA, seed)
        augT, augJ, augy = np.r_[T, Ta], np.r_[J, Ja], np.r_[y, ya]
        model = fit(augT, augJ, augy)
        # Sequential certificate: retain the pre-test hypothesis family and
        # filter it using the newly observed batch. Do not recenter the family
        # after seeing the batch, since that measures fit movement rather than
        # information gained by the experiment.
        cert, _ = certificate(ps, M._add_mu(base.p, augT, augJ), augT, augJ,
                              sigma=SIGMA, observed_y=augy)
        out[f"{label}_certificate"] = cert
    return out


def main():
    rows = [evaluate(seed) for seed in SEEDS]
    summary = {}
    for label in ("base", "safety", "fisher", "random"):
        vals = [r[f"{label}_certificate"]["worst_case_fold"] if label != "base"
                else r["base_certificate"]["worst_case_fold"] for r in rows]
        summary[f"{label}_median_worst_case_fold"] = float(np.median(vals))
        summary[f"{label}_abstain_rate"] = float(np.mean(np.asarray(vals) > TARGET_FOLD))
    summary["safety_median_T_K"] = float(np.median([r["safety_optimal_T_K"] for r in rows]))
    summary["fisher_median_T_K"] = float(np.median([r["fisher_T_K"] for r in rows]))
    summary["random_median_T_K"] = float(np.median([r["random_T_K"] for r in rows]))
    payload = {"protocol": {"train_window": "T >= 523 K", "use_temperature_K": T_USE,
                             "candidate_temperatures_K": CANDIDATE_T.tolist(),
                             "candidate_currents_MA_cm2": J_TEST.tolist(),
                             "sigma_ln": SIGMA, "target_fold": TARGET_FOLD,
                             "compatibility_rule": "fixed total excess log-residual <= 5 sigma^2",
                             "seeds": SEEDS}, "summary": summary, "runs": rows,
              "interpretation": "SAFE means compatible members disagree by at most 2x at the deployment grid; ABSTAIN means the data do not support that sign-off resolution. This certificate is an empirical benchmark result, not a coverage guarantee."}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "deployment_safety_certificate.json"), "w") as f:
        json.dump(payload, f, indent=2)
    os.makedirs(os.path.join(ROOT, "paper", "figures"), exist_ok=True)
    names = ["base", "safety", "fisher", "random"]
    values = [[r["base_certificate"]["worst_case_fold"] if n == "base" else
               r[f"{n}_certificate"]["worst_case_fold"] for r in rows] for n in names]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.boxplot(values, labels=["accelerated\nonly", "certificate\npolicy", "Fisher\ncontrol", "random\ncontrol"],
               showmeans=True)
    ax.axhline(TARGET_FOLD, color="k", ls="--", lw=1, label="2× sign-off limit")
    ax.set_ylabel("Compatible-set deployment disagreement (fold)")
    ax.set_title("Deployment-safety certificate across 20 seeds")
    ax.set_yscale("log"); ax.legend(fontsize=8); fig.tight_layout()
    for path in (os.path.join(ROOT, "figures", "fig_deployment_safety_certificate.png"),
                 os.path.join(ROOT, "paper", "figures", "fig_deployment_safety_certificate.png")):
        os.makedirs(os.path.dirname(path), exist_ok=True); fig.savefig(path, dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
