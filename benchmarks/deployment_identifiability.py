"""Deployment-identifiability and adaptive-test-design benchmark.

This arm formalizes the paper's main finding.  A model can fit the accelerated
window well while leaving the deployment functional poorly identified.  We
measure that ambiguity with a likelihood-compatible parameter envelope and
test whether an additional temperature batch selected from the fitted Fisher
geometry improves use-condition safety.

The experiment is deliberately modest and reproducible:
  * fit the free corrected equation on T >= 523 K;
  * construct a local 95%-style parameter ellipsoid from the accelerated
    Jacobian and propagate it to 373 K;
  * choose one candidate temperature by expected reduction of deployment
    variance;
  * refit after adding five current levels at that temperature;
  * compare adaptive selection with random candidate selection.

Results are written to results/deployment_identifiability.json and figures.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import generate_dataset as gen
from benchmarks import eval as E
from benchmarks import models as M

KB = 8.617333262e-5
T_USE = 373.0
J_USE = np.array([0.1, 0.5, 1.0])
J_TEST = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
CANDIDATE_T = np.array([443.0, 455.0, 475.0, 494.0, 510.0, 523.0])
SEEDS = list(range(20))


def numerical_gradient(fun, p):
    p = np.asarray(p, dtype=float)
    g = np.empty(len(p), dtype=float)
    for i, value in enumerate(p):
        h = 1e-5 * max(1.0, abs(value))
        q, z = p.copy(), p.copy()
        q[i] += h
        z[i] -= h
        g[i] = (float(fun(q)) - float(fun(z))) / (2.0 * h)
    return g


def model_mu(p, T, J):
    return M._add_mu(np.asarray(p, dtype=float), T, J)


def fit_free(T, J, y):
    m = M.AdditiveModel()
    m.fit(np.asarray(T), np.asarray(J), np.asarray(y))
    return m


def fisher_and_deployment_lever(model, T, J):
    p = np.asarray(model.p, dtype=float)
    G = np.stack([numerical_gradient(lambda q: model_mu(q, t, j), p)
                  for t, j in zip(T, J)])
    F = G.T @ G
    Finv = np.linalg.pinv(F, rcond=1e-12)
    H = np.stack([numerical_gradient(lambda q: model_mu(q, T_USE, j), p)
                  for j in J_USE])
    # Sum of deployment-functional variances over the requested current levels.
    lever = float(np.trace(H @ Finv @ H.T))
    eig = np.linalg.eigvalsh(F)
    return F, H, lever, float(eig[0]), float(eig[-1] / max(eig[0], 1e-30))


def envelope(model, F, H, radius=2.0, n_draws=3000, seed=0):
    """Propagate a local likelihood-compatible ellipsoid to use conditions."""
    rng = np.random.default_rng(seed)
    p = np.asarray(model.p, dtype=float)
    # Regularize only for sampling; the reported lever uses the raw pseudoinverse.
    w, V = np.linalg.eigh(F)
    floor = max(float(w[-1]) * 1e-12, 1e-14)
    cov_sqrt = V @ np.diag(1.0 / np.sqrt(np.maximum(w, floor))) @ V.T
    z = rng.normal(size=(n_draws, len(p)))
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    z *= rng.uniform(0.0, radius, size=(n_draws, 1))
    ps = p[None, :] + z @ cov_sqrt.T
    preds = np.stack([model_mu(q, T_USE, J_USE) for q in ps])
    return {
        "log_lifetime_low": preds.min(axis=0).tolist(),
        "log_lifetime_high": preds.max(axis=0).tolist(),
        "ea_use_low": float(np.min(ps[:, 2])),
        "ea_use_high": float(np.max(ps[:, 2])),
        "n_draws": int(n_draws),
    }


def candidate_scores(model, F, H):
    p = np.asarray(model.p, dtype=float)
    Finv = np.linalg.pinv(F, rcond=1e-12)
    rows = []
    for Tc in CANDIDATE_T:
        Gc = np.stack([numerical_gradient(lambda q: model_mu(q, Tc, j), p)
                       for j in J_TEST])
        Fnew = F + Gc.T @ Gc
        invnew = np.linalg.pinv(Fnew, rcond=1e-12)
        old = float(np.trace(H @ Finv @ H.T))
        new = float(np.trace(H @ invnew @ H.T))
        rows.append({"T_K": float(Tc), "lever_after": new,
                     "relative_reduction": float(1.0 - new / old)})
    return rows


def add_batch(df, Tnew, sigma, seed):
    rng = np.random.default_rng(seed + 10000)
    T = np.full(len(J_TEST), float(Tnew))
    clean = gen.black_core_mttf(T, J_TEST)[0]
    y = np.log(clean) + rng.normal(0.0, sigma, len(J_TEST))
    return T, J_TEST.copy(), y


def evaluate_one(seed, sigma=0.15):
    df, _ = gen.generate(noise_sigma=sigma, seed=seed)
    train = df[df.T_K >= 523.0]
    T, J, y = train.T_K.values, train.J_MA_cm2.values, np.log(train.MTTF_hours.values)
    true = np.log(gen.black_core_mttf(np.full(3, T_USE), J_USE)[0])
    base = fit_free(T, J, y)
    F, H, lever, lam_min, cond = fisher_and_deployment_lever(base, T, J)
    scores = candidate_scores(base, F, H)
    adaptive = min(scores, key=lambda r: r["lever_after"])["T_K"]
    random_t = float(CANDIDATE_T[seed % len(CANDIDATE_T)])

    base_fold = E.fold_error(base.predict(np.full(3, T_USE), J_USE), true)["fold_factor"]
    out = {"seed": seed, "sigma": sigma, "base_fold": float(base_fold),
           "deployment_lever": lever, "lambda_min": lam_min,
           "condition_number": cond, "candidate_scores": scores,
           "adaptive_T_K": adaptive, "random_T_K": random_t}
    for label, Tc in (("adaptive", adaptive), ("random", random_t)):
        Ta, Ja, ya = add_batch(df, Tc, sigma, seed)
        augmented = fit_free(np.r_[T, Ta], np.r_[J, Ja], np.r_[y, ya])
        pred = augmented.predict(np.full(3, T_USE), J_USE)
        out[f"{label}_fold"] = float(E.fold_error(pred, true)["fold_factor"])
        Fa, Ha, la, lma, ca = fisher_and_deployment_lever(augmented, np.r_[T, Ta], np.r_[J, Ja])
        out[f"{label}_lever"] = la
    return out


def make_figures(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.scatter(d["deployment_lever"], d["base_fold"], c=d["seed"], cmap="viridis", s=28)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Deployment-conditioned lever (accelerated fit)")
    ax.set_ylabel("Use-condition fold error")
    ax.set_title("Good accelerated fits can leave deployment risk unresolved")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_deployment_identifiability.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    x = np.arange(len(d))
    ax.plot(x, d["base_fold"], "o-", label="accelerated only")
    ax.plot(x, d["adaptive_fold"], "o-", label="adaptive extra temperature")
    ax.plot(x, d["random_fold"], "o-", label="random extra temperature", alpha=.7)
    ax.axhline(2.0, color="k", ls="--", lw=1, label="2× safety target")
    ax.set_xlabel("Seed"); ax.set_ylabel("Use-condition fold error")
    ax.set_title("Deployment-aware test selection reduces extrapolation risk")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_adaptive_test_design.png"), dpi=180)
    plt.close(fig)


def main():
    rows = [evaluate_one(seed) for seed in SEEDS]
    payload = {
        "protocol": {
            "train_window": "T >= 523 K",
            "use_temperature_K": T_USE,
            "candidate_temperatures_K": CANDIDATE_T.tolist(),
            "candidate_currents_MA_cm2": J_TEST.tolist(),
            "sigma_ln": 0.15,
            "seeds": SEEDS,
        },
        "runs": rows,
        "summary": {
            "base_fold_median": float(np.median([r["base_fold"] for r in rows])),
            "adaptive_fold_median": float(np.median([r["adaptive_fold"] for r in rows])),
            "random_fold_median": float(np.median([r["random_fold"] for r in rows])),
            "adaptive_T_median_K": float(np.median([r["adaptive_T_K"] for r in rows])),
            "adaptive_lever_reduction_median": float(np.median([
                1.0 - r["adaptive_lever"] / r["deployment_lever"] for r in rows])),
            "adaptive_safe_rate_2x": float(np.mean([r["adaptive_fold"] <= 2.0 for r in rows])),
            "random_safe_rate_2x": float(np.mean([r["random_fold"] <= 2.0 for r in rows])),
        },
    }
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "deployment_identifiability.json"), "w") as f:
        json.dump(payload, f, indent=2)
    make_figures(rows, os.path.join(ROOT, "paper", "figures"))
    make_figures(rows, os.path.join(ROOT, "figures"))
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
