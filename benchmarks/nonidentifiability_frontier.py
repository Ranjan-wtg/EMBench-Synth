"""Analytical deployment-non-identifiability frontier.

Constructs a family of corrected-equation parameterizations that trade the
current prefactors against the high-temperature pathway prefactor.  When the
grain-boundary pathway dominates the accelerated envelope, this transformation
nearly cancels in-window, but it changes the low-temperature deployment
prediction where the surface pathway reappears.

This is a falsifiable, optimizer-independent result.  It asks how much
deployment disagreement can coexist with a training-envelope discrepancy below
the measurement noise.  It is the analytical candidate for the paper's novel
discovery; it must still pass external prior-art review.
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
from benchmarks.models import _add_mu

KB = 8.617333262e-5
T_USE = 373.0
J_USE = np.array([0.1, 0.5, 1.0])
DELTAS = np.linspace(-4.0, 4.0, 161)
WINDOWS = [523.0, 548.0, 573.0, 623.0]
SCENARIOS = {
    "low_ratio": {"Ea_s": 0.7, "Ea_g": 1.1, "D0r": 3.0e3},
    "benchmark": {"Ea_s": 0.8, "Ea_g": 1.2, "D0r": 1.2e4},
    "high_ratio": {"Ea_s": 0.9, "Ea_g": 1.4, "D0r": 5.0e4},
}


def truth_params():
    return np.array([
        np.log(gen.B0), np.log(gen.C0), gen.PATHWAYS[0][2],
        gen.PATHWAYS[1][2],
        np.log(gen.PATHWAYS[1][1] / gen.PATHWAYS[0][1]),
    ], dtype=float)


def evaluate_window(tmin, scenario_name, scenario):
    df, _ = gen.generate(noise_sigma=0.0, seed=42)
    tr = df[df.T_K >= tmin]
    T, J = tr.T_K.values, tr.J_MA_cm2.values
    p = truth_params()
    p[2] = scenario["Ea_s"]
    p[3] = scenario["Ea_g"]
    p[4] = np.log(scenario["D0r"])
    train_rows = []
    for delta in DELTAS:
        # Shift both current prefactors and the dominant pathway prefactor.
        # In the GB-dominated limit this leaves accelerated predictions fixed.
        q = p.copy()
        q[0] += delta
        q[1] += delta
        q[4] += delta
        train_diff = np.max(np.abs(_add_mu(q, T, J) - _add_mu(p, T, J)))
        use_diff = np.abs(_add_mu(q, np.full(3, T_USE), J_USE)
                          - _add_mu(p, np.full(3, T_USE), J_USE))
        train_rows.append({"scenario": scenario_name, "Tmin_K": tmin, "delta": float(delta),
                           "max_train_abs_logdiff": float(train_diff),
                           "median_use_abs_logdiff": float(np.median(use_diff)),
                           "max_use_fold": float(np.exp(np.max(use_diff)))})
    return train_rows


def main():
    rows = sum((evaluate_window(t, name, scenario)
                for name, scenario in SCENARIOS.items()
                for t in WINDOWS), [])
    d = pd.DataFrame(rows)
    # Alternatives hidden below a representative log-noise level.
    compatible = d[d.max_train_abs_logdiff <= 0.15]
    summary = {}
    for t in WINDOWS:
        x = compatible[compatible.Tmin_K == t]
        summary[str(int(t))] = {
            "compatible_parameterizations": int(len(x)),
            "max_deployment_fold": float(x.max_use_fold.max()) if len(x) else None,
            "median_deployment_log_error": float(x.median_use_abs_logdiff.max()) if len(x) else None,
            "scenario_max_fold": {name: float(x[x.scenario == name].max_use_fold.max())
                                   for name in SCENARIOS},
        }
    payload = {
        "protocol": {"windows_Tmin_K": WINDOWS, "noise_compatibility_threshold": 0.15,
                     "use_temperature_K": T_USE, "deltas": DELTAS.tolist()},
        "summary": summary,
        "rows": rows,
        "note": "Analytical tradeoff family; no optimizer or noisy fit is used. "
                "This is evidence for practical deployment non-identifiability, "
                "not by itself a prior-art novelty claim.",
    }
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "nonidentifiability_frontier.json"), "w") as f:
        json.dump(payload, f, indent=2)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for t in WINDOWS:
        x = d[d.Tmin_K == t]
        ax.plot(x.max_train_abs_logdiff, x.max_use_fold, label=f"T≥{int(t)} K")
    ax.axvline(0.15, color="k", ls="--", lw=1, label="noise compatibility")
    ax.set_yscale("log")
    ax.set_xlabel("Maximum accelerated-window |Δ log lifetime|")
    ax.set_ylabel("Maximum deployment fold disagreement")
    ax.set_title("Deployment disagreement hidden by the accelerated envelope")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for path in (os.path.join(ROOT, "figures", "fig_nonidentifiability_frontier.png"),
                 os.path.join(ROOT, "paper", "figures", "fig_nonidentifiability_frontier.png")):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
