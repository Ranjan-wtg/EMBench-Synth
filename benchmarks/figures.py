"""Figure generation for the paper (saved to figures/)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

kB_EV = 8.617333262e-5


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_arrhenius_kink(df_base, phys, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for J in [0.5, 2.0, 8.0]:
        sub = df_base[df_base.J_MA_cm2 == J].sort_values("T_K")
        ax.plot(1000.0 / sub.T_K, np.log(sub.MTTF_hours), "o", ms=4,
                label=f"J = {J:g} MA/cm$^2$")
    ax.set_xlabel("1000/T  [K$^{-1}$]")
    ax.set_ylabel("ln(MTTF)")
    ax.set_title("Blind data: curved Arrhenius plot (two-regime kink)")
    ax.legend(fontsize=8)
    _save(fig, path)


def fig_ea_recovery(T_centers, ea_rec, phys, Tc_rec, path):
    Td = np.linspace(443, 698, 300)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(Td, phys.ea_effective(Td), "-", color="k", lw=2, label="ground truth $E_a^{\\mathrm{eff}}(T)$")
    ax.plot(T_centers, ea_rec, "s", ms=5, color="C0", label="recovered (local Arrhenius slopes)")
    ax.axvline(494.2, color="gray", ls="--", lw=1)
    ax.text(494.2, 1.18, "crossover 494 K", fontsize=8, ha="center")
    if Tc_rec and not np.isnan(Tc_rec):
        ax.axvline(Tc_rec, color="C3", ls=":", lw=1.5)
        ax.text(Tc_rec, 0.87, f"recovered {Tc_rec:.0f} K", fontsize=8, ha="center", color="C3")
    ax.axvspan(443, 698, alpha=0.05, color="green", label="tested envelope (170-425 C)")
    ax.set_xlabel("T [K]")
    ax.set_ylabel("$E_a^{\\mathrm{eff}}$ [eV]")
    ax.set_ylim(0.82, 1.22)
    ax.set_title("Two-regime activation energy: recovery")
    ax.legend(fontsize=8)
    _save(fig, path)


def fig_n_recovery(J_mid, n_rec, n_true, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    Jd = np.logspace(-0.4, 1.1, 100)
    ax.plot(Jd, (2 + 0.3 * Jd) / (1 + 0.3 * Jd), "-", color="k", lw=2,
            label=r"ground truth $n_{\mathrm{eff}}(J) = \frac{2+\alpha J}{1+\alpha J}$")
    ax.plot(J_mid, n_rec, "o", ms=6, color="C0", label="recovered (secant slopes)")
    ax.set_xscale("log")
    ax.set_xlabel("J [MA/cm$^2$]")
    ax.set_ylabel("$n_{\\mathrm{eff}}(J)$")
    ax.set_ylim(1.0, 2.0)
    ax.set_title("Fractional current exponent: recovery (1 < n < 2)")
    ax.legend(fontsize=8)
    _save(fig, path)


def fig_blech(df_ext, phys, fitted_jlc, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    jl = np.linspace(3200, 60000, 300)
    true_boost = jl / (jl - 3000.0)
    ax.loglog(jl, true_boost, "-", color="k", lw=2, label="ground truth $jL/(jL-(jL)_c)$")
    ax.loglog(jl, jl / (jl - fitted_jlc), "--", color="C0", lw=2,
              label=f"fitted (recovered $(jL)_c$ = {fitted_jlc:.0f} A/cm)")
    ax.axvline(3000.0, color="gray", ls="--", lw=1)
    ax.text(3000.0, 3.2, "$(jL)_c$ = 3000 A/cm", fontsize=8, ha="right", rotation=90)
    ax.set_xlabel("$jL$ [A/cm]")
    ax.set_ylabel("Blech boost factor")
    ax.set_title("Blech critical-product recovery")
    ax.legend(fontsize=8)
    _save(fig, path)


def fig_extrapolation(extrap_base, extrap_ext, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))

    def panel(ax, res, title, note):
        d = res["models"]
        names = list(d.keys())
        vals = [d[n]["fold_factor"] for n in names]
        logratio = [d[n]["median_log_ratio"] for n in names]
        colors = ["C3" if v > 2.0 else "C2" for v in vals]
        ax.bar(np.arange(len(names)), vals, color=colors, alpha=0.85)
        for i, (v, lr) in enumerate(zip(vals, logratio)):
            tag = "over-pred." if lr > 0.1 else ("under-pred." if lr < -0.1 else "")
            ax.text(i, v * 1.15, f"{v:.1f}x\n{tag}", ha="center", fontsize=7)
        ax.set_yscale("log")
        ax.set_ylim(0.8, 30)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=16, ha="right", fontsize=7)
        ax.set_title(title)
        ax.set_ylabel("median |fold error|" if note else "")
        ax.grid(True, alpha=0.3)

    panel(axes[0], extrap_base, "(a) base: fit T$\\geq$250$^\\circ$C, use 100$^\\circ$C", True)
    panel(axes[1], extrap_ext, "(b) Blech: + length dependence, use 100$^\\circ$C", False)
    fig.suptitle("Extrapolation accelerated-test $\\rightarrow$ use conditions", y=1.02, fontsize=11)
    _save(fig, path)


def fig_kan_edges(dbb, phys, path):
    """Recovered KAN edge functions vs the corrected-equation decomposition.

    The single-layer KAN writes ln MTTF = f_T(T) + f_J(J) + bias. Each edge is
    mean-aligned with its physics counterpart: f_T = -ln D_eff(T) and
    f_J = ln(B0 J^-2 + C0 J^-1) from the fitted corrected equation.
    """
    from .kanlite import KANLite
    from . import models as M
    X = dbb[["T_K", "J_MA_cm2"]].values.astype(np.float32)
    y = np.log(dbb.MTTF_hours.values).astype(np.float32)
    kan = KANLite(d_in=2, grid=10, k=3)
    kan.fit(X, y, steps=2500, lr=3e-2)
    add = M.AdditiveModel(fix_D0r=1.2e4).fit(dbb.T_K.values, dbb.J_MA_cm2.values, y)
    ap = add.params()

    def phys_T(T):
        return -np.logaddexp(-ap["Ea_surface_eV"] / (kB_EV * T),
                             np.log(ap["D0_GB_over_surface"]) -
                             ap["Ea_grain_boundary_eV"] / (kB_EV * T))

    def phys_J(J):
        return np.logaddexp(np.log(ap["B0"]) - 2 * np.log(J),
                            np.log(ap["C0"]) - np.log(J))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    Td = np.linspace(443, 698, 200)
    Jd = np.logspace(np.log10(0.5), np.log10(8.0), 200)
    Xq = np.stack([Td, np.full(200, 4.0)], 1).astype(np.float32)
    XqJ = np.stack([np.full(200, 560.0), Jd], 1).astype(np.float32)
    ev = kan.edge_values(Xq)
    evJ = kan.edge_values(XqJ)
    eT = ev[0] - ev[0].mean() + phys_T(Td).mean()
    eJ = evJ[1] - evJ[1].mean() + phys_J(Jd).mean()
    axes[0].plot(Td, phys_T(Td), "-", color="k", lw=2, label="physics: $-\\ln D_{\\mathrm{eff}}(T)$")
    axes[0].plot(Td, eT, "--", color="C0", lw=2, label="KAN edge $f_T(T)$")
    axes[0].set_xlabel("T [K]"); axes[0].set_ylabel("edge value (mean-aligned)")
    axes[0].set_title("Temperature edge vs physics")
    axes[0].legend(fontsize=8)
    axes[1].plot(Jd, phys_J(Jd), "-", color="k", lw=2, label=r"physics: $\ln(B_0 J^{-2}+C_0 J^{-1})$")
    axes[1].plot(Jd, eJ, "--", color="C0", lw=2, label="KAN edge $f_J(J)$")
    axes[1].set_xscale("log"); axes[1].set_xlabel("J [MA/cm$^2$]")
    axes[1].set_ylabel("edge value (mean-aligned)")
    axes[1].set_title("Current-density edge vs physics")
    axes[1].legend(fontsize=8)
    fig.suptitle("KAN decomposition recovers the corrected-equation structure", y=1.02, fontsize=11)
    _save(fig, path)


def fig_paradox(fit_rmse, extrap_fold, path):
    """In-sample (accelerated-window) fit vs extrapolation error.

    The extrapolation paradox: models that fit the accelerated window best
    (right) are the ones that extrapolate worst (top). The literature-constrained
    corrected equation is the single bottom-left point.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, x, y in zip(fit_rmse.keys(), fit_rmse.values(), extrap_fold.values()):
        ax.scatter(x, y, s=90, zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 6),
                    fontsize=7.5)
    ax.set_xlabel("in-sample ln-RMSE on accelerated window (lower = better)")
    ax.set_ylabel("extrapolation fold error at 100 C (lower = better)")
    ax.set_yscale("log")
    ax.set_title("The extrapolation paradox: better fit $\\neq$ better extrapolation")
    _save(fig, path)


def fig_structure_size(res, path):
    sweep = res["sweep"]
    n = [r["n_samples"] for r in sweep]
    m = [r["delta_bic_additive_minus_single_mean"] for r in sweep]
    s = [r["delta_bic_additive_minus_single_sd"] for r in sweep]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4),
                                  gridspec_kw={"wspace": 0.3})
    ax.errorbar(n, m, yerr=s, fmt="o-", color="C0", lw=2, capsize=3)
    ax.axhline(0.0, color="k", ls="--", lw=1)
    ax.text(n[-1], 1.5, "single power law favored", fontsize=7.5, ha="right")
    ax.text(n[0], -60, "additive structure favored", fontsize=7.5, ha="left")
    ax.set_xlabel("samples n")
    ax.set_ylabel(r"$\Delta$BIC (additive $-$ single power law)")
    ax.set_title("(a) growth of evidence with n", fontsize=10)

    loo = res.get("leave_one_out", [])
    levels = [r["dropped_J"] for r in loo]
    vals = [r["delta_bic_additive_minus_single"] for r in loo]
    full = max(sweep, key=lambda r: r["n_J_levels"])
    full_db = full["delta_bic_additive_minus_single_mean"]
    colors = ["C3" if v > 0 else "C0" for v in vals]
    ax2.bar([f"{l:g}" for l in levels], vals, color=colors)
    ax2.axhline(full_db, color="k", ls="--", lw=1)
    ax2.text(0.5, full_db + 4, f"full sweep {full_db:+.1f}", fontsize=7.5,
             ha="left", va="bottom")
    ax2.axhline(0.0, color="k", lw=0.8)
    ax2.set_xlabel("dropped J level (MA/cm$^2$)")
    ax2.set_ylabel(r"$\Delta$BIC")
    ax2.set_title("(b) leave-one-level-out verdicts", fontsize=10)
    fig.suptitle("Critical sample size for structure discovery", fontsize=12)
    _save(fig, path)


def fig_phase_transition(sigma_fold, path):
    """Noise-induced phase transition in extrapolation error.

    fold error at 100 C (y, log) vs noise level sigma_ln (x) for the five
    fitted models. The free-D0r corrected equation is exact at sigma=0 and
    collapses discontinuously at sigma>=0.02; pinned/soft-prior stay flat;
    the two-regime incumbent shows a false-confidence minimum; Black's
    equation is uniformly wrong and noise-insensitive.
    """
    sigmas = sorted((float(s) for s in sigma_fold), reverse=False)
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    colors = {"free MLE": "C3", "pinned": "C2", "soft prior sd5": "C0",
              "two-regime": "C1", "Black": "k"}
    lstyles = {"free MLE": "-", "pinned": "-", "soft prior sd5": "--",
               "two-regime": "-.", "Black": ":"}
    for name, c in colors.items():
        ys = [sigma_fold[repr(s)][name] for s in sigmas]
        ax.plot(sigmas, ys, lstyles[name], color=c, marker="o", ms=4,
                lw=2, label=name)
    ax.annotate("phase transition:\n1.00x -> 5.19x\nbetween sigma=0 and 0.02",
                (0.02, 5.19), xytext=(0.06, 7.5),
                arrowprops=dict(arrowstyle="->", color="C3", lw=1.2),
                fontsize=7.5, color="C3")
    ax.set_xlabel(r"measurement noise $\sigma_{\ln}$")
    ax.set_ylabel("median |fold error| at 100 C (predicted/true)")
    ax.set_yscale("log")
    ax.set_ylim(0.8, 30)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Noise-induced phase transition in extrapolation safety")
    _save(fig, path)


def fig_robustness(noise_res, size_res, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    ax = axes[0]
    ax.plot([r["noise_sigma"] for r in noise_res], [r["rmse_eV"] for r in noise_res],
            "o-", color="C0")
    ax.set_xlabel(r"noise $\sigma_{\ln}$")
    ax.set_ylabel(r"Ea recovery RMSE [eV]")
    ax.set_title("Robustness to noise")
    ax = axes[1]
    ax.plot([r["n_samples"] for r in size_res], [r["rmse_eV"] for r in size_res],
            "o-", color="C1")
    ax.set_xlabel("training samples")
    ax.set_ylabel(r"Ea recovery RMSE [eV]")
    ax.set_title("Robustness to sample size")
    _save(fig, path)


def fig_ml_predictors(ml_pred, path):
    """Arm 1: ML predictors on the phase-transition sweep.

    Black-box predictors (MLP incumbent-like; spline KAN and GP-SR with no
    usable extrapolation) fail at orders of magnitude above the physics
    models. The physics-regularized MLP (corrected-eq output structure + soft
    prior toward literature parameters) is the positive arm: it stays with the
    pinned physics reference. Log y.
    """
    sigmas = sorted((float(s) for s in ml_pred), reverse=False)
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for name, c in {"MLP (64-64)": "C0", "KAN (spline)": "C4",
                    "GP-SR": "C5"}.items():
        ys = [ml_pred[repr(s)][name] for s in sigmas]
        ax.plot(sigmas, ys, "-o", color=c, ms=4, lw=2, label=name)
    ys = [ml_pred[repr(s)]["Phys-reg MLP"] for s in sigmas]
    ax.plot(sigmas, ys, "-s", color="C6", ms=5, lw=2, label="physics-regularized MLP")
    ax.axhline(10.44, color="C3", ls=":", lw=1.5, label="free corrected (physics, 10.4x)")
    ax.axhline(1.38, color="C2", ls="--", lw=1.5, label="pinned corrected (1.4x)")
    ax.set_xlabel(r"measurement noise $\sigma_{\ln}$")
    ax.set_ylabel("median |fold error| at 100 C (predicted/true)")
    ax.set_yscale("log")
    ax.set_ylim(0.8, 1e5)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Black-box ML cannot extrapolate; physics-regularized ML can")
    _save(fig, path)


def fig_gnn(gnn_res, path):
    """Arm 3: PDN graph extrapolation, GNN vs topology-blind MLP vs oracle."""
    names = ["GNN", "MLP\n(topology-blind)", "physics oracle\n(pinned)"]
    vals = [gnn_res["GNN"]["use_condition_fold"],
            gnn_res["MLP (topology-blind)"]["use_condition_fold"],
            gnn_res["physics oracle (pinned, true T/J/L)"]["use_condition_fold"]]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    bars = ax.bar(names, vals, color=["C0", "C1", "C2"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.05,
                f"{v:.2f}x", ha="center", fontsize=9)
    ax.set_ylabel("median fold error at use conditions (373 K)")
    ax.set_yscale("log")
    ax.set_ylim(0.8, 10)
    ax.set_title("Graph structure does not fix extrapolation\n"
                 "(power-delivery networks, medians over 5 seeds)")
    _save(fig, path)
