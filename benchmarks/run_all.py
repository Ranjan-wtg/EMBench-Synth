"""Run the full EMBench-Synth benchmark suite and write results/ + figures/."""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import generate_dataset as gen          # noqa: E402
from benchmarks import recovery as rec   # noqa: E402
from benchmarks import models as M       # noqa: E402
from benchmarks import sr as sr          # noqa: E402
from benchmarks import eval as E         # noqa: E402
from benchmarks import figures as fig    # noqa: E402
from benchmarks.kanlite import KANLite   # noqa: E402

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

T_CROSS_TRUE = float(gen.T_CROSSOVER_K)
J_QUERY = np.array([0.7, 1.4, 2.8, 5.7])


def load_data():
    db = pd.read_csv(os.path.join(DATA, "em_synthetic_full_with_groundtruth.csv"))
    dbb = pd.read_csv(os.path.join(DATA, "em_synthetic_blind.csv"))
    de = pd.read_csv(os.path.join(DATA, "em_synthetic_extended_full_with_groundtruth.csv"))
    deb = pd.read_csv(os.path.join(DATA, "em_synthetic_extended_blind.csv"))
    return db, dbb, de, deb


def run_recovery(dbb, de, deb):
    out = {}
    T, ea, det = rec.recover_ea_eff(dbb, None, half_window=5)
    ea_metrics = rec.evaluate_ea_recovery(T, ea, gen.ea_effective, T_CROSS_TRUE)
    out["ea_eff"] = ea_metrics

    # Incumbent comparison: the constant-Ea error a standard Black's fit makes
    black = M.BlackMLE().fit(dbb.T_K.values, dbb.J_MA_cm2.values,
                             np.log(dbb.MTTF_hours.values))
    ea_const = black.Ea_eV
    out["ea_eff"]["constant_Ea_baseline"] = {
        "Ea_const_eV": ea_const,
        "rmse_vs_true_eV": float(np.sqrt(np.mean(
            (ea_const - gen.ea_effective(np.asarray(T))) ** 2)))}

    J_mid, Jmid_vals, secants = rec.recover_n_eff(dbb, J_QUERY)
    out["n_eff"] = rec.evaluate_n_recovery(J_mid, Jmid_vals, gen.n_effective)

    models = {"AdditiveBlech": M.AdditiveBlech()}
    fitted, boundary = rec.recover_blech_threshold(de, models)
    out["blech"] = {"fitted_jl_c": fitted, "boundary": boundary}

    neg = rec.negative_control(dbb, gen.ea_effective, None,
                               real_mae_eV=ea_metrics["mae_eV"])
    out["negative_control"] = neg
    return out, T, ea, Jmid_vals, secants, fitted["AdditiveBlech"]


def run_kan(dbb):
    """Single-layer spline KAN on ln(MTTF): additive decomposition in [T, J].

    Plain init, physics-initialized (from the fitted Black's + additive fits),
    and Black's-residual (physics-informed) variants.
    """
    X = dbb[["T_K", "J_MA_cm2"]].values.astype(np.float32)
    y = np.log(dbb.MTTF_hours.values).astype(np.float32)

    def run_variant(name, funcs=None):
        kan = KANLite(d_in=2, grid=10, k=3)
        if funcs is not None:
            kan.physics_init(funcs)
        kan.fit(X, y, steps=2500, lr=3e-2)
        pred = kan(X).detach().numpy()
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        r2 = float(1 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2))
        symb = kan.symbolic(X)
        return {"name": name, "rmse_ln": rmse, "r2": r2,
                "edge_families": [f"{s[1] if s[0] > -np.inf else 'none'}" for s in symb],
                "edge_r2": [round(float(s[0]), 4) if s[0] > -np.inf else None for s in symb]}

    # Normalized-variable maps for physics priors. The KAN decomposes
    # ln MTTF = f_T(T) + f_J(J) + bias, so the priors must match that
    # decomposition: f_T = -ln D_eff(T) and f_J = ln(B0 J^-2 + C0 J^-1),
    # both from the fitted corrected equation.
    T0, T1 = dbb.T_K.min(), dbb.T_K.max()
    lnJ0, lnJ1 = np.log(dbb.J_MA_cm2.min()), np.log(dbb.J_MA_cm2.max())
    add = M.AdditiveModel(fix_D0r=1.2e4).fit(dbb.T_K.values, dbb.J_MA_cm2.values, y)
    ap = add.params()
    kB = 8.617333262e-5

    def t_prior(x):
        T = T0 + (T1 - T0) * (x + 1) / 2
        return -np.logaddexp(-ap["Ea_surface_eV"] / (kB * T),
                             np.log(ap["D0_GB_over_surface"]) -
                             ap["Ea_grain_boundary_eV"] / (kB * T))

    def j_prior(x):
        lnJ = lnJ0 + (lnJ1 - lnJ0) * (x + 1) / 2
        return np.logaddexp(np.log(ap["B0"]) - 2 * lnJ, np.log(ap["C0"]) - lnJ)

    variants = {
        "plain": run_variant("plain KAN"),
        "physics_init_black": run_variant("physics-init (Black's + additive priors)",
                                          [t_prior, j_prior]),
    }

    # physics-informed: learn the residual to a fitted Black's equation
    black = M.BlackMLE().fit(dbb.T_K.values, dbb.J_MA_cm2.values, y)
    y_res = y - black.predict(dbb.T_K.values, dbb.J_MA_cm2.values)
    kan = KANLite(d_in=2, grid=10, k=3)
    kan.fit(X, y_res, steps=2500, lr=3e-2)
    pred = kan(X).detach().numpy() + black.predict(dbb.T_K.values, dbb.J_MA_cm2.values)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    r2 = float(1 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2))
    variants["physics_informed_residual"] = {
        "name": "physics-informed KAN (Black's + residual)",
        "rmse_ln": rmse, "r2": r2,
        "edge_families": [s[1] if s[0] > -np.inf else "none"
                          for s in kan.symbolic(X)],
        "edge_r2": [round(float(s[0]), 4) if s[0] > -np.inf else None
                    for s in kan.symbolic(X)]}
    return variants


def run_gp(dbb):
    return sr.run_gp_sr(dbb)


def run_structural(dbb):
    return sr.structural_comparison(dbb)


def main():
    t0 = time.time()
    db, dbb, de, deb = load_data()

    results = {"dataset": {"base_rows": len(dbb), "extended_rows": len(de),
                           "crossover_T_K": T_CROSS_TRUE,
                           "jL_c_A_per_cm": 3000.0}}

    print("[1/12] recovery benchmarks")
    rec_res, T, ea, Jmid, secants, jlc_fit = run_recovery(dbb, de, deb)
    results["recovery"] = rec_res
    print(f"  Ea RMSE {rec_res['ea_eff']['rmse_eV']:.4f} eV, "
          f"Tc rec {rec_res['ea_eff']['Tc_recovered_K']:.0f} K "
          f"(err {rec_res['ea_eff']['Tc_error_K']:.0f} K)")
    print(f"  n RMSE {rec_res['n_eff']['rmse']:.4f}")
    print(f"  (jL)_c recovered {jlc_fit:.0f} A/cm (true 3000)")
    print(f"  negative control: mean|err| {rec_res['negative_control']['mean_abs_err_eV']:.3f} eV")

    print("[2/12] structural comparison (BIC)")
    results["structural"] = run_structural(dbb)
    s = results["structural"]
    print(f"  delta BIC (additive - single) = {s['delta_bic_additive_minus_single']:.1f}")

    print("[3/12] GP symbolic regression")
    results["gp_sr"] = run_gp(dbb)
    print(f"  R^2 {results['gp_sr']['r2']:.3f}, complexity {results['gp_sr']['complexity']}")

    print("[4/12] spline-KAN decomposition")
    results["kan"] = run_kan(dbb)
    for k, v in results["kan"].items():
        print(f"  {v['name']}: R^2 {v['r2']:.3f}, edges {v['edge_families']}")

    print("[5/12] extrapolation to use conditions")
    base_models = M.extrapolation_base_models()
    tr_accel = dbb[dbb.T_K >= 523.0]
    Tac, Jac = tr_accel.T_K.values, tr_accel.J_MA_cm2.values
    yac = np.log(tr_accel.MTTF_hours.values)
    fit_rmse = {}
    for m in base_models:
        m.fit(Tac, Jac, yac)
        fit_rmse[m.name] = float(np.sqrt(np.mean((m.predict(Tac, Jac) - yac) ** 2)))
    results["extrapolation_base"] = E.extrapolation_base(base_models, dbb, gen)
    results["extrapolation_paradox"] = {
        "in_sample_fit_rmse_ln": {k: round(v, 4) for k, v in fit_rmse.items()},
        "extrapolation_fold": {k: round(v["fold_factor"], 3) for k, v in
                               results["extrapolation_base"]["models"].items()}}
    for name, d in results["extrapolation_base"]["models"].items():
        print(f"  {name[:52]:52s} fit {fit_rmse[name]:.4f}  fold {d['fold_factor']:.1f}x")

    ext_models = M.extrapolation_extended_models()
    results["extrapolation_extended"] = E.extrapolation_extended(ext_models, de, gen)
    for name, d in results["extrapolation_extended"]["models"].items():
        print(f"  {name[:52]:52s} fold {d['fold_factor']:.1f}x "
              f"(log-ratio {d['median_log_ratio']:+.2f})")

    print("[6/12] generalization holdouts")
    results["generalization"] = E.generalization(
        db, de, os.path.join(DATA, "splits"),
        {"base": M.base_models(), "extended": M.extended_models()}, gen)
    for split in ["random", "crossover_band", "j_tier"]:
        row = results["generalization"]["base"][split]
        best = min(row["models"].items(), key=lambda kv: kv[1])
        print(f"  base/{split}: best {best[0][:32]} rmse {best[1]:.3f}")
    row = results["generalization"]["extended"]["short_L"]
    best = min(row["models"].items(), key=lambda kv: kv[1])
    print(f"  ext/short_L: best {best[0][:32]} rmse {best[1]:.3f}")

    print("[7/12] robustness sweeps")
    results["robustness_noise"] = E.noise_sweep(gen.generate, gen)
    results["robustness_size"] = E.sample_size_sweep(dbb, gen)
    for r in results["robustness_noise"]:
        print(f"  sigma={r['noise_sigma']:.2f}: Ea RMSE {r['rmse_eV']:.4f} eV")
    for r in results["robustness_size"]:
        print(f"  n={r['n_samples']}: Ea RMSE {r['rmse_eV']:.4f} eV")
    print("  structure discovery (BIC vs single power law):")
    results["structure_discovery_size"] = E.structure_size_sweep(dbb, gen)
    for r in results["structure_discovery_size"]["sweep"]:
        print(f"    n={r['n_samples']:4d} J={r['n_J_levels']} "
              f"d_bic={r['delta_bic_additive_minus_single_mean']:+7.1f} "
              f"p_win={r['p_additive_wins']:.2f} critical={r['critical_reached']}")
    for r in results["structure_discovery_size"]["leave_one_out"]:
        print(f"    leave-one-out J={r['dropped_J']:.1f}: "
              f"d_bic={r['delta_bic_additive_minus_single']:+7.1f}")

    print("[8/12] noise-induced phase transition sweep")
    sys.path.insert(0, ROOT)
    import prototype_phase_transition as pt
    pt.SIGMAS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.3, 0.5]
    results["phase_transition"] = {}
    for sigma in pt.SIGMAS:
        dfb, _ = gen.generate(noise_sigma=sigma, seed=42)
        tr = dfb[dfb.T_K >= 523.0]
        Tac, Jac = tr.T_K.values, tr.J_MA_cm2.values
        yac = np.log(tr.MTTF_hours.values)
        true_log = np.log(gen.black_core_mttf(np.full(3, pt.TUSE), pt.JUSE)[0])
        row = {}
        for name, m in pt.make_models().items():
            m.fit(Tac, Jac, yac)
            row[name] = E.fold_error(m.predict(np.full(3, pt.TUSE), pt.JUSE),
                                     true_log)["fold_factor"]
        results["phase_transition"][repr(sigma)] = row
    fig.fig_phase_transition(results["phase_transition"],
                             os.path.join(FIG, "fig_phase_transition.png"))
    for sigma, row in results["phase_transition"].items():
        print(f"  sigma={sigma}: " + ", ".join(f"{k}={v:.2f}x" for k, v in row.items()))

    print("[9/12] figures")
    fig.fig_arrhenius_kink(dbb, gen, os.path.join(FIG, "fig_arrhenius_kink.png"))
    fig.fig_ea_recovery(T, ea, gen, rec_res["ea_eff"]["Tc_recovered_K"],
                        os.path.join(FIG, "fig_ea_recovery.png"))
    fig.fig_n_recovery(Jmid, secants, gen.n_effective(Jmid),
                       os.path.join(FIG, "fig_n_recovery.png"))
    fig.fig_blech(de, gen, jlc_fit, os.path.join(FIG, "fig_blech.png"))
    fig.fig_extrapolation(results["extrapolation_base"],
                          results["extrapolation_extended"],
                          os.path.join(FIG, "fig_extrapolation.png"))
    fig.fig_kan_edges(dbb, gen, os.path.join(FIG, "fig_kan_edges.png"))
    fig.fig_robustness(results["robustness_noise"], results["robustness_size"],
                       os.path.join(FIG, "fig_robustness.png"))
    fig.fig_paradox(results["extrapolation_paradox"]["in_sample_fit_rmse_ln"],
                    results["extrapolation_paradox"]["extrapolation_fold"],
                    os.path.join(FIG, "fig_paradox.png"))
    fig.fig_structure_size(results["structure_discovery_size"],
                           os.path.join(FIG, "fig_structure_size.png"))

    print("[10/12] ML predictors as extrapolators (KAN, GP-SR, MLP)")
    from benchmarks import ml_predictors as mp
    mp_main = mp.main()
    import json as _json
    with open(os.path.join(RES, "ml_predictors.json")) as f:
        results["ml_predictors"] = _json.load(f)["ml_predictor_phase_transition"]

    print("[11/12] GNN-for-circuits arm (power-delivery networks)")
    from benchmarks import gnn as gnnmod
    gnnmod.main()

    print("[12/12] leaderboard (model zoo + scoring)")
    from benchmarks import leaderboard as lb
    lb.build_leaderboard()

    results["meta"] = {"runtime_s": round(time.time() - t0, 1)}
    with open(os.path.join(RES, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\ndone in {results['meta']['runtime_s']}s -> results/results.json, figures/")


if __name__ == "__main__":
    main()
