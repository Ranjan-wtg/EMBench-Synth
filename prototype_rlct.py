"""Fisher-geometry theory for the noise-induced phase transition (v2).

Compute the observed Fisher information F at the KNOWN truth (controlled
benchmark) and the extrapolation lever  lever = nabla g_use^T F^{-1} nabla g_use.
The free-D0r model has a near-singular Fisher matrix (a weak/compensation
direction along which Ea_s and ln D0r trade off while in-window predictions
barely change); the extrapolation functional at 100 C is strongly aligned with
that direction. The predicted extrapolation variance is sigma^2 * lever / n, so
the median |log fold| should scale as sigma * sqrt(lever/n) * 0.6745. As the
training envelope deepens (T_min up), the surface pathway becomes invisible and
lambda_min(F) -> 0: the model becomes exactly singular, the regime where
Watanabe's RLCT governs the posterior width instead of sqrt(n) rates.

Writes results/prototype_rlct.json (v2 keys: truth_lever, eigen_at_truth, ...).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_dataset as gen
from benchmarks import eval as E

KB = 8.617333262e-5
ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
MEDIAN_FACTOR = 0.6744897501960817   # median of |N(0,1)|

TRUE_D0R = 1.2e4
JUSE = np.array([0.1, 0.5, 1.0])
TUSE = 373.0


def num_grad(fn, p, eps=1e-6):
    g = np.zeros(len(p))
    for i in range(len(p)):
        dp = np.zeros(len(p)); dp[i] = eps
        g[i] = (fn(p + dp) - fn(p - dp)) / (2 * eps)
    return g


def add_mu_eval(p, T, J):
    """Mean in log space. 5-param (lnB0,lnC0,Ea_s,Ea_gb,lnD0r) or 4-param."""
    if len(p) == 4:
        lnB0, lnC0, Ea_s, Ea_gb = p
        lnD0r = np.log(TRUE_D0R)
    else:
        lnB0, lnC0, Ea_s, Ea_gb, lnD0r = p
    u = 1.0 / KB / np.asarray(T, float)
    lnJ = np.log(np.asarray(J, float))
    lnD = np.logaddexp(-Ea_s * u, lnD0r - Ea_gb * u)
    return np.logaddexp(lnB0 - 2 * lnJ, lnC0 - lnJ) - lnD


def truth_params():
    return np.array([np.log(gen.B0), np.log(gen.C0), 0.8, 1.2,
                     np.log(gen.PATHWAYS[1][1] / gen.PATHWAYS[0][1])])


def fisher_at_truth(T, J, p):
    """Observed Fisher info (per-sample sum of outer products of score) at p."""
    rows = []
    for T_i, J_i in zip(T, J):
        rows.append(num_grad(lambda q: add_mu_eval(q, T_i, J_i), p))
    return np.array(rows).T @ np.array(rows)   # sum_i grad g_i grad g_i^T


def lever_at_truth(F, p):
    """Extrapolation lever: grad g_use^T F^{-1} grad g_use (per use point)."""
    out = []
    for J_use in JUSE:
        grad = num_grad(lambda q: add_mu_eval(q, TUSE, J_use), p)
        out.append(float(grad @ np.linalg.pinv(F) @ grad))
    return out


def main():
    res = {}
    for tmin in (523.0, 548.0, 573.0, 623.0):
        db, _ = gen.generate(noise_sigma=0.15, seed=42)
        tr = db[db.T_K >= tmin]
        T, J = tr.T_K.values, tr.J_MA_cm2.values
        p5 = truth_params()

        F_free = fisher_at_truth(T, J, p5)
        F_pin = fisher_at_truth(T, J, p5[:4])   # lnD0r held fixed at truth
        w_f = np.linalg.eigvalsh(F_free)
        w_p = np.linalg.eigvalsh(F_pin)

        # smallest-eigenvalue direction of the free model's F
        w, V = np.linalg.eigh(F_free)
        vmin = V[:, 0]
        grad_use = num_grad(lambda q: add_mu_eval(q, TUSE, JUSE[0]), p5)
        align = float(abs(vmin @ grad_use) / (np.linalg.norm(vmin) *
                                              np.linalg.norm(grad_use)))

        lev_free = lever_at_truth(F_free, p5)
        lev_pin = lever_at_truth(F_pin, p5[:4])

        # lever decomposition by eigen-direction: lever = sum_i (c_i^2 / w_i),
        # c_i = v_i . grad_use. Share carried by the smallest eigen-direction.
        w_v, V_v = np.linalg.eigh(F_free)
        grad_use_mid = num_grad(lambda q: add_mu_eval(q, TUSE, JUSE[1]), p5)
        c_mid = V_v.T @ grad_use_mid
        terms = c_mid ** 2 / w_v
        vmin_share = terms[0] / terms.sum()
        res[f"T>=_{int(tmin)}_K"] = {
            "n": int(len(T)),
            "lambda_min_free": float(w_f[0]),
            "lambda_min_pinned": float(w_p[0]),
            "cond_free": float(w_f[-1] / w_f[0]),
            "cond_pinned": float(w_p[-1] / w_p[0]),
            "vmin_grad_use_alignment": round(align, 4),
            "lever_free": [round(float(v), 3) for v in lev_free],
            "lever_pinned": [round(float(v), 3) for v in lev_pin],
            "lever_ratio_free_over_pinned":
                round(float(np.mean(lev_free) / np.mean(lev_pin)), 2),
            "lever_vmin_share": round(float(vmin_share), 4),
            "lever_vmin_term": round(float(terms[0]), 1),
        }
        print(f"T>={int(tmin):3d}K n={len(T):3d}  lam_min_free={w_f[0]:.2e} "
              f"lam_min_pin={w_p[0]:.2e}  cond_free={w_f[-1] / w_f[0]:.1e} "
              f"align={align:.3f}  lever_free={np.mean(lev_free):.1f} "
              f"lever_pin={np.mean(lev_pin):.2f} "
              f"ratio={np.mean(lev_free) / np.mean(lev_pin):.0f}x")

    # Predicted vs actual fold for the free model across sigma at T>=523
    db, _ = gen.generate(noise_sigma=0.15, seed=42)
    tr = db[db.T_K >= 523.0]
    T, J = tr.T_K.values, tr.J_MA_cm2.values
    p5 = truth_params()
    F = fisher_at_truth(T, J, p5)
    lev = np.mean(lever_at_truth(F, p5))
    pred = {}
    for sigma in (0.0, 0.02, 0.05, 0.1, 0.15, 0.3, 0.5):
        db2, _ = gen.generate(noise_sigma=sigma, seed=42)
        tr2 = db2[db2.T_K >= 523.0]
        y = np.log(tr2.MTTF_hours.values)
        pred_logerr = MEDIAN_FACTOR * sigma * np.sqrt(lev / len(T))
        true_log = np.log(gen.black_core_mttf(np.full(3, TUSE), JUSE)[0])
        # actual fold using the true-parameter model (perfect fit at truth)
        actual = E.fold_error(add_mu_eval(p5, np.full(3, TUSE), JUSE),
                              true_log)
        pred[repr(sigma)] = {
            "predicted_fold": round(float(np.exp(pred_logerr)), 3),
            "predicted_logerr": round(float(pred_logerr), 3),
            "actual_fold_truthmodel": round(actual["fold_factor"], 3),
        }
    res["free_sigma_prediction_at_truth_lever"] = pred

    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, "prototype_rlct.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
