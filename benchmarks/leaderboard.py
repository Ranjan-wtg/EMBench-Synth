"""Arm 2: EMBench-Synth benchmark release -- model zoo, scoring, leaderboard.

Packages the benchmark as reusable infrastructure:
  - MODEL_ZOO: a registry of models with a uniform fit/predict API
  - score_model(): a small set of canonical scoring functions
  - build_leaderboard(): scores every model in the zoo on the canonical
    extrapolation task (accelerated -> use conditions) and writes a
    machine-readable leaderboard (results/leaderboard.json) plus a compact
    human-readable table.

The zoo and scoring functions are the API a third-party submission would use;
this is the 'reproducible benchmark infrastructure' deliverable referenced in
the paper.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import generate_dataset as gen
from benchmarks import eval as E
from benchmarks import models as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

T_ACCEL = 523.0
T_USE = 373.0
J_USE = np.array([0.1, 0.5, 1.0])


def make_zoo():
    """Registry: name -> factory. Uniform API: fit(T, J, y[, L]) / predict(T, J[, L]).
    A submission adds its model here and is scored by build_leaderboard()."""
    return {
        "Black's equation (industry incumbent)": lambda: M.BlackMLE(),
        "Two-regime Black's (piecewise Ea)": lambda: M.TwoRegimeBlack(),
        "Corrected equation, free D0 ratio": lambda: M.AdditiveModel(),
        "Corrected equation, pinned D0 ratio": lambda: M.AdditiveModel(fix_D0r=1.2e4),
        "Black-box MLP (64-64)": lambda: M.MLPBaseline(extended=False),
        "Physics-regularized MLP (corrected eq.)": lambda: M.PhysicsRegMLP(),
    }


def score_model(model, df):
    """Canonical scores for one fitted model on the accelerated->use task.

    Returns in-window fit RMSE (ln), use-condition fold error, safe-to-2x
    temperature, and the sign (over/under) of the use-condition error.
    """
    train = df[df.T_K >= T_ACCEL]
    Tac, Jac = train.T_K.values, train.J_MA_cm2.values
    yac = np.log(train.MTTF_hours.values)
    model.fit(Tac, Jac, yac)
    fit_rmse = float(np.sqrt(np.mean((model.predict(Tac, Jac) - yac) ** 2)))

    Tuse = np.full(3, T_USE)
    true_log = np.log(gen.black_core_mttf(Tuse, J_USE)[0])
    fo = E.fold_error(model.predict(Tuse, J_USE), true_log)
    fold = fo["fold_factor"]

    # safe-to-2x temperature: the T at which the model's prediction is 2x the
    # true MTTF (solve along the 1/T axis at the use J). Positive sign =>
    # under-protective (overpredicted lifetime) at that temperature.
    T2x = _safe_to_2x(model, true_log)
    return {
        "in_window_fit_rmse_ln": round(fit_rmse, 4),
        "use_condition_fold": round(fold, 3),
        "use_condition_median_logratio": round(fo["median_log_ratio"], 3),
        "safe_to_2x_T_K": round(T2x, 1),
    }


def _safe_to_2x(model, true_log, J=J_USE):
    """Safe-to-2x temperature from the paper's transfer law.

    log(fold) grows linearly in (1/T_use - 1/T_accel); the slope gives an
    effective transfer activation dEa, and safe-to-2x is the extrapolation
    temperature at which the fold error reaches 2x (see the Quantitative
    transfer section of the paper). Matches prototype_improved.main_quantitative.
    """
    import numpy as np
    KB = 8.617333262e-5
    Tmin = 523.0
    slopes, logs = [], []
    for Tu in (373.0, 400.0, 425.0, 450.0, 475.0, 494.0):
        Tl = np.full(3, Tu)
        true = np.log(gen.black_core_mttf(Tl, J)[0])
        fo = E.fold_error(model.predict(Tl, J), true)
        logs.append(fo["median_log_ratio"])
        slopes.append(1.0 / Tu - 1.0 / Tmin)
    dea = np.polyfit(np.asarray(slopes), np.asarray(logs), 1)[0] * KB
    d1t_safe = KB * np.log(2.0) / max(abs(dea), 1e-6)
    t_safe = 1.0 / (1.0 / Tmin + d1t_safe)
    return t_safe


def build_leaderboard():
    df, _ = gen.generate(noise_sigma=0.15, seed=42)
    rows = []
    for name, factory in make_zoo().items():
        scores = score_model(factory(), df)
        rows.append({"model": name, **scores})
        print(f"  {name[:52]:52s} fit {scores['in_window_fit_rmse_ln']:.3f}  "
              f"fold {scores['use_condition_fold']:.2f}x  "
              f"safe-to-2x {scores['safe_to_2x_T_K']:.0f} K")
    rows.sort(key=lambda r: r["use_condition_fold"])
    payload = {
        "leaderboard": rows,
        "task": "accelerated (T>=523 K) -> use (100 C, J 0.1-1 MA/cm^2)",
        "noise_sigma": 0.15, "seed": 42,
        "metrics": ["in_window_fit_rmse_ln", "use_condition_fold",
                    "use_condition_median_logratio", "safe_to_2x_T_K"],
        "zoo_api": "MODEL_ZOO[name] = factory; factory().fit(T, J, y); "
                   "factory().predict(T, J)",
    }
    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, "leaderboard.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    build_leaderboard()
