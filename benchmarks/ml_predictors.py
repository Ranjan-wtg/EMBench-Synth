"""Arm 1: modern ML predictors as *extrapolators* on EMBench-Synth.

The paper's phase-transition study uses physics-informed candidates (corrected
equation, Black's, two-regime). This module asks the ML-for-chip question the
workshop scope is aimed at: how do black-box ML predictors -- spline KAN, GP
symbolic regression, and a neural MLP -- behave when trained on the accelerated
window and asked to extrapolate to use conditions?

Protocol mirrors prototype_phase_transition.py: for each sigma in SIGMAS,
regenerate the benchmark with that noise, fit each ML predictor on T >= 523 K,
predict at 100 C / low J, report median fold error.

Writes results/ml_predictors.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import generate_dataset as gen
from benchmarks import eval as E
from benchmarks.kanlite import KANLite
from benchmarks.models import MLPBaseline, PhysicsRegMLP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

SIGMAS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.3, 0.5]
TUSE = 373.0
JUSE = np.array([0.1, 0.5, 1.0])

# GP-SR: reduced budget for the sweep (kept modest so the full suite stays
# tractable; the recovery arm already runs the large-budget GP on clean data).
GP_POP = 1000
GP_GENS = 20
GP_SEED = 7


def run_gp_predictor(Tac, Jac, yac, Tuse, Juse):
    """GP symbolic regression on [1/T, ln J]; evaluate the program anywhere."""
    from gplearn.genetic import SymbolicRegressor
    from gplearn.functions import make_function

    EXP = make_function(function=lambda x: np.exp(np.clip(x, -20, 20)),
                        name="exp", arity=1)
    X = np.stack([1.0 / Tac, np.log(Jac)], axis=1)
    y = yac
    est = SymbolicRegressor(
        population_size=GP_POP, generations=GP_GENS, function_set=[
            "add", "sub", "mul", "div", "log", "sqrt", "neg", "inv", EXP],
        stopping_criteria=1e-4, p_crossover=0.7, p_subtree_mutation=0.1,
        p_hoist_mutation=0.05, p_point_mutation=0.1, max_samples=0.9,
        verbose=0, random_state=GP_SEED, n_jobs=1,
        parsimony_coefficient=0.003)
    est.fit(X, y)
    Xu = np.stack([1.0 / Tuse, np.log(Juse)], axis=1)
    return est.predict(Xu), float(est.score(X, y)), str(est._program)


def run_kan_predictor(Tac, Jac, yac, Tuse, Juse):
    """Spline KAN on [T, ln J]. The spline basis is compactly supported on the
    training envelope, so this isolates the *zero-extrapolation* limit of
    local basis-function predictors: features outside the accelerated window
    map to the constant bias."""
    kan = KANLite(d_in=2, grid=10, k=3)
    X = np.stack([Tac, np.log(Jac)], axis=1).astype(np.float32)
    kan.fit(X, yac.astype(np.float32), steps=2500, lr=3e-2)
    Xu = np.stack([Tuse, np.log(Juse)], axis=1).astype(np.float32)
    pred = kan(Xu).detach().numpy()
    pred_tr = kan(X).detach().numpy()
    in_r2 = float(1 - np.sum((pred_tr - yac) ** 2) / np.sum((yac - yac.mean()) ** 2))
    return pred, in_r2, None


def main():
    out = {}
    for sigma in SIGMAS:
        dfb, _ = gen.generate(noise_sigma=sigma, seed=42)
        tr = dfb[dfb.T_K >= 523.0]
        Tac, Jac = tr.T_K.values, tr.J_MA_cm2.values
        yac = np.log(tr.MTTF_hours.values)
        Tuse = np.full(3, TUSE)
        true_log = np.log(gen.black_core_mttf(Tuse, JUSE)[0])

        row = {}

        mlp = MLPBaseline(extended=False)
        mlp.fit(Tac, Jac, yac)
        row["MLP (64-64)"] = round(E.fold_error(mlp.predict(Tuse, JUSE),
                                                true_log)["fold_factor"], 3)

        kan_pred, _, _ = run_kan_predictor(Tac, Jac, yac, Tuse, JUSE)
        row["KAN (spline)"] = round(E.fold_error(kan_pred, true_log)["fold_factor"], 3)

        gp_pred, gp_r2, prog = run_gp_predictor(Tac, Jac, yac, Tuse, JUSE)
        row["GP-SR"] = round(E.fold_error(gp_pred, true_log)["fold_factor"], 3)
        if sigma == 0.15:
            row["gp_program_at_0.15"] = prog
            row["gp_r2_at_0.15"] = round(gp_r2, 4)

        prm = PhysicsRegMLP(lam=0.1)
        prm.fit(Tac, Jac, yac)
        row["Phys-reg MLP"] = round(E.fold_error(prm.predict(Tuse, JUSE),
                                                 true_log)["fold_factor"], 3)

        out[repr(sigma)] = row
        print(f"sigma={sigma:.2f}: " + ", ".join(f"{k}={v}" for k, v in row.items()))

    payload = {
        "ml_predictor_phase_transition": out,
        "note": "fit on T>=523, predict 100C. KAN spline basis is "
                "compactly supported on the training envelope (zero "
                "extrapolation). GP-SR budget reduced for the sweep. "
                "Phys-reg MLP: corrected-eq output structure + soft prior "
                "toward literature parameters (lam=0.1); the positive ML arm.",
    }
    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, "ml_predictors.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {path}")

    from benchmarks import figures as fig
    fig.fig_ml_predictors(out, os.path.join(ROOT, "figures",
                                            "fig_ml_predictors.png"))
    print("wrote figures/fig_ml_predictors.png")


if __name__ == "__main__":
    main()
