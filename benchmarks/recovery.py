"""Recovery benchmarks: can the blind data reveal the physics we encoded?

We measure the pipeline's ability to recover three known physical quantities
that are NOT exposed to it in the blind CSV:
  1. Ea_eff(T) -- the two-regime activation energy (via local Arrhenius slopes)
  2. n_eff(J)  -- the fractional current exponent (via local log-log slopes)
  3. (jL)_c    -- the Blech critical product (via the divergence model fit)

Plus a negative control (shuffled MTTF must recover nothing).
"""
import numpy as np
import pandas as pd

kB_EV = 8.617333262e-5


def _interp_at(T, ea, Tq):
    return float(np.interp(Tq, T, ea))


def recover_ea_eff(df, T_query, half_window=3):
    """Ea_eff(T) from local slopes of ln(MTTF) vs 1/T at fixed J, median over J.

    df: blind base data with columns T_K, J_MA_cm2, MTTF_hours.
    Returns (T_centers, ea_recovered, per_joint detail).
    """
    T_unique = np.sort(df.T_K.unique())
    T_centers = T_unique[half_window:-half_window]
    recovered = np.empty(len(T_centers))
    detail = []
    for i, T in enumerate(T_centers):
        win = T_unique[max(0, i - half_window):i + half_window + 1]
        slopes = []
        for J in df.J_MA_cm2.unique():
            sub = df[(df.J_MA_cm2 == J) & (df.T_K.isin(win))]
            if len(sub) < 4:
                continue
            x = 1.0 / sub.T_K.values
            y = np.log(sub.MTTF_hours.values)
            slope = np.polyfit(x, y, 1)[0]
            slopes.append(slope)
        recovered[i] = np.median(slopes) * kB_EV   # eV
        detail.append((T, float(recovered[i])))
    return T_centers, recovered, detail


def recover_n_eff(df, J_mid):
    """n_eff(J) from secant slopes of ln(MTTF) vs ln(J), log-averaged over T."""
    J_grid = np.sort(df.J_MA_cm2.unique())
    log_mean = {J: float(np.log(df[df.J_MA_cm2 == J].MTTF_hours).mean())
                for J in J_grid}
    secants = np.zeros(len(J_grid) - 1)
    for k in range(len(J_grid) - 1):
        Ja, Jb = J_grid[k], J_grid[k + 1]
        secants[k] = -(log_mean[Jb] - log_mean[Ja]) / (np.log(Jb) - np.log(Ja))
    J_mid_vals = np.sqrt(J_grid[:-1] * J_grid[1:])
    return np.interp(J_mid, J_mid_vals, secants) if J_mid.size else secants, J_mid_vals, secants


def recover_blech_threshold(df_ext, models):
    """(jL)_c from (a) the fitted divergence model and (b) the immortal boundary."""
    mortal = df_ext[~df_ext.immortal & df_ext.MTTF_hours.notna()]
    fitted = {}
    for name, model in models.items():
        model.fit(mortal.T_K.values, mortal.J_MA_cm2.values,
                  np.log(mortal.MTTF_hours.values), mortal.L_um.values)
        fitted[name] = model.jl_c()
    imm_max = df_ext[df_ext.immortal].jL_product_A_cm.max()
    mor_min = df_ext[~df_ext.immortal].jL_product_A_cm.min()
    boundary = float((imm_max + mor_min) / 2.0)
    return fitted, {"immortal_max_jL": float(imm_max), "mortal_min_jL": float(mor_min),
                    "boundary_estimate": boundary, "true_jLc": 3000.0}


def evaluate_ea_recovery(T_centers, ea_rec, true_func, T_cross_true=494.2):
    T_centers = np.asarray(T_centers, dtype=float)
    ea_rec = np.asarray(ea_rec, dtype=float)
    ea_true = true_func(T_centers)
    rmse = float(np.sqrt(np.mean((ea_rec - ea_true) ** 2)))
    mae = float(np.mean(np.abs(ea_rec - ea_true)))

    # Robust crossover estimate: the steepest part of the recovered Ea_eff(T)
    # curve (max |dEa/dT| after light smoothing). More stable than a full
    # logistic fit, which is ill-conditioned on the partially-sampled curve.
    Tc_rec, Ea1_rec, Ea2_rec = np.nan, np.nan, np.nan
    try:
        d = np.gradient(ea_rec, T_centers)
        k = min(3, len(T_centers) - 1)
        dsm = np.convolve(d, np.ones(k) / k, mode="valid")
        i = int(np.argmax(np.abs(dsm))) + (k - 1) // 2
        Tc_rec = float(T_centers[i])
        lo = ea_rec[T_centers < Tc_rec] if (T_centers < Tc_rec).any() else ea_rec[:2]
        hi = ea_rec[T_centers > Tc_rec] if (T_centers > Tc_rec).any() else ea_rec[-2:]
        Ea1_rec = float(np.median(lo))
        Ea2_rec = float(np.median(hi))
        if Ea1_rec > Ea2_rec:
            Ea1_rec, Ea2_rec = Ea2_rec, Ea1_rec
    except Exception:
        pass
    Tc_err = float(abs(Tc_rec - T_cross_true)) if not np.isnan(Tc_rec) else np.nan
    return {"rmse_eV": rmse, "mae_eV": mae,
            "Tc_recovered_K": Tc_rec, "Tc_error_K": Tc_err,
            "Ea_low_recovered_eV": Ea1_rec, "Ea_high_recovered_eV": Ea2_rec,
            "T_cross_true_K": T_cross_true}


def evaluate_n_recovery(n_rec, J_mid, true_func):
    n_true = true_func(J_mid)
    rmse = float(np.sqrt(np.mean((n_rec - n_true) ** 2)))
    return {"rmse": rmse, "n_recovered": [float(x) for x in n_rec],
            "n_true": [float(x) for x in n_true], "J_mid": [float(x) for x in J_mid]}


def negative_control(df, true_func, T_query, half_window=3, n_shuffles=5,
                     real_mae_eV=None):
    """Shuffle MTTF; recovered Ea_eff should be far from the true physics."""
    errs = []
    for _ in range(n_shuffles):
        dfc = df.copy()
        dfc.MTTF_hours = dfc.MTTF_hours.sample(frac=1, random_state=int(42 + _)).values
        T, ea, _ = recover_ea_eff(dfc, None, half_window)
        errs.append(np.mean(np.abs(ea - true_func(np.asarray(T)))))
    return {"mean_abs_err_eV": float(np.mean(errs)),
            "real_recovery_mae_eV": float(real_mae_eV) if real_mae_eV else None}
