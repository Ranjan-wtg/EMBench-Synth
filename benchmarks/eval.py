"""Extrapolation, generalization, and robustness benchmarks."""
import numpy as np

kB_EV = 8.617333262e-5


def fold_error(pred_log, true_log):
    """Median |ln(pred/true)| and directional median log-ratio."""
    d = np.asarray(pred_log) - np.asarray(true_log)
    return {"median_abs_logerr": float(np.median(np.abs(d))),
            "median_log_ratio": float(np.median(d)),          # >0 => overprediction
            "fold_factor": float(np.exp(np.median(np.abs(d))))}


# ---------------------------------------------------------------------------
# Extrapolation: accelerated test -> use conditions (the industrial job of
# Black's equation). Fit on T >= 523 K (250 C), predict at 100 C / low J.
# ---------------------------------------------------------------------------
def extrapolation_base(models, df, phys):
    train = df[df.T_K >= 523.0]
    Ttr, Jtr, ytr = (train.T_K.values, train.J_MA_cm2.values,
                     np.log(train.MTTF_hours.values))
    Tuse = np.array([373.0, 373.0, 373.0])
    Juse = np.array([0.1, 0.5, 1.0])
    mttf, _, _ = phys.black_core_mttf(Tuse, Juse)
    true_log = np.log(mttf)
    results = {}
    for m in models:
        m.fit(Ttr, Jtr, ytr)
        pred = m.predict(Tuse, Juse)
        results[m.name] = fold_error(pred, true_log)
    return {"fit_n": int(len(train)), "use_conditions": "T=373K, J=0.1-1.0 MA/cm^2",
            "true_use_logMTTF": [float(v) for v in true_log],
            "models": results}


def extrapolation_extended(models, df, phys):
    train = df[(df.T_K >= 523.0) & (~df.immortal) & df.MTTF_hours.notna()]
    Ttr, Jtr, Ltr = train.T_K.values, train.J_MA_cm2.values, train.L_um.values
    ytr = np.log(train.MTTF_hours.values)
    # Use conditions straddle the Blech threshold from above (mortal), at the
    # same jL = {3200, 4000, 4800} A/cm as the accelerated test, but at 100 C.
    # These are deliberately *not* extrapolated to an immortal regime, so a
    # finite MTTF is defined and the jL-agnostic models can be compared.
    Tuse = np.array([373.0, 373.0, 373.0, 373.0])
    Juse = np.array([8.0, 8.0, 8.0, 8.0])
    Luse = np.array([400.0, 500.0, 600.0, 600.0])
    true = []
    for T, J, L in zip(Tuse, Juse, Luse):
        mttf, _, _ = phys.black_core_mttf(T, J)
        true.append(np.log(mttf * phys.blech_boost(phys.jl_product(J, L))))
    results = {}
    for m in models:
        m.fit(Ttr, Jtr, ytr, Ltr)
        results[m.name] = fold_error(m.predict(Tuse, Juse, Luse), np.array(true))
    return {"fit_n": int(len(train)), "jL_use_A_per_cm": [3200.0, 4000.0, 4800.0, 4800.0],
            "models": results}


# ---------------------------------------------------------------------------
# Generalization: held-out temperature band / current tier / length regime
# ---------------------------------------------------------------------------
def _rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def generalization(df_base, df_ext, splits_dir, models, phys):
    import pandas as pd
    res = {"base": {}, "extended": {}}

    def load(split, df):
        tr = pd.read_csv(f"{splits_dir}/{split}_train.csv").row_index.values
        te = pd.read_csv(f"{splits_dir}/{split}_test.csv").row_index.values
        return df.iloc[tr], df.iloc[te]

    for split in ["random", "crossover_band", "j_tier"]:
        dtr, dte = load(split, df_base)
        Ttr, Jtr = dtr.T_K.values, dtr.J_MA_cm2.values
        ytr = np.log(dtr.MTTF_hours.values)
        Tte, Jte, yte = dte.T_K.values, dte.J_MA_cm2.values, np.log(dte.MTTF_hours.values)
        entry = {}
        for m in models["base"]:
            m.fit(Ttr, Jtr, ytr)
            entry[m.name] = _rmse(m.predict(Tte, Jte), yte)
        res["base"][split] = {"test_n": int(len(dte)), "models": entry}

    dtr, dte = load("short_L", df_ext)
    mortal = ~dte.immortal & dte.MTTF_hours.notna()
    dte = dte[mortal]
    Ttr, Jtr, Ltr = dtr.T_K.values, dtr.J_MA_cm2.values, dtr.L_um.values
    ytr = np.log(dtr.MTTF_hours.values)
    entry = {}
    for m in models["extended"]:
        m.fit(Ttr, Jtr, ytr, Ltr)
        entry[m.name] = _rmse(m.predict(dte.T_K.values, dte.J_MA_cm2.values,
                                        dte.L_um.values), np.log(dte.MTTF_hours.values))
    res["extended"]["short_L"] = {"test_n": int(len(dte)), "models": entry}
    return res


# ---------------------------------------------------------------------------
# Robustness: noise level and sample-size sweeps
# ---------------------------------------------------------------------------
def noise_sweep(base_gen, phys, noise_levels=(0.0, 0.05, 0.15, 0.3, 0.5)):
    from . import recovery as rec
    Tq = None
    out = []
    for sigma in noise_levels:
        dfb, _ = base_gen(noise_sigma=sigma)
        T, ea, _ = rec.recover_ea_eff(dfb, Tq, half_window=5)
        metrics = rec.evaluate_ea_recovery(T, ea, phys.ea_effective)
        out.append({"noise_sigma": sigma, **metrics})
    return out


def structure_size_sweep(df, phys, k_levels=(1, 2, 3, 4, 5, 6, 8, 10),
                         repeats=20, seed=0):
    """How many samples are needed before the blind data statistically forces
    the additive (J^-2 + J^-1) structure over a single power law (BIC)?

    Sample size is grown by adding J levels (every T is always swept, as in
    practice). Each draw is stratified to always span the operating range (the
    lowest and highest J are always included), mirroring real test planning;
    `repeats` draws average out which intermediate levels happen to be sampled.
    """
    from . import sr
    J_all = np.sort(df.J_MA_cm2.unique())
    rng = np.random.default_rng(seed)
    out = []
    for k in k_levels:
        if k > len(J_all):
            break
        d_bics, nrows = [], 0
        for rep in range(repeats):
            if k == 1:
                J_sel = np.array([J_all[len(J_all) // 2]])
            elif k == 2:
                J_sel = np.array([J_all[0], J_all[-1]])
            else:
                mid = rng.choice(J_all[1:-1], size=k - 2, replace=False)
                J_sel = np.concatenate([[J_all[0]], mid, [J_all[-1]]])
            sub = df[df.J_MA_cm2.isin(J_sel)]
            T, J = sub.T_K.values, sub.J_MA_cm2.values
            y = np.log(sub.MTTF_hours.values)
            nrows = int(len(sub))
            p1, k1 = sr._fit_model(T, J, y, additive=False)
            p2, k2 = sr._fit_model(T, J, y, additive=True)
            if p2 is None:
                continue
            bic1, _ = sr._bic(y, p1, k1)
            bic2, _ = sr._bic(y, p2, k2)
            d_bics.append(bic2 - bic1)
        if not d_bics:
            continue
        d_bics = np.array(d_bics)
        out.append({
            "n_J_levels": int(k), "n_samples": int(nrows),
            "n_repeats": int(len(d_bics)),
            "delta_bic_additive_minus_single_mean": float(d_bics.mean()),
            "delta_bic_additive_minus_single_sd": float(d_bics.std()),
            "p_additive_wins": float(np.mean(d_bics < 0)),
            "critical_reached": bool(d_bics.mean() < 0 and np.all(d_bics < 0))})

    if len(J_all) > 1:
        from itertools import combinations
        loo = []
        for combo in combinations(J_all, len(J_all) - 1):
            sub = df[df.J_MA_cm2.isin(combo)]
            T, J = sub.T_K.values, sub.J_MA_cm2.values
            y = np.log(sub.MTTF_hours.values)
            p1, k1 = sr._fit_model(T, J, y, additive=False)
            p2, k2 = sr._fit_model(T, J, y, additive=True)
            if p2 is not None:
                bic1, _ = sr._bic(y, p1, k1)
                bic2, _ = sr._bic(y, p2, k2)
                loo.append({"dropped_J": float(set(J_all).difference(combo).pop()),
                            "delta_bic_additive_minus_single": float(bic2 - bic1)})
        out = {"sweep": out, "leave_one_out": loo}
    return out


def sample_size_sweep(df, phys, sizes=(60, 120, 180, 300)):
    from . import recovery as rec
    out = []
    J_all = np.sort(df.J_MA_cm2.unique())
    for n in sizes:
        # Structured subsample: keep ALL T values (so the Ea_eff(T) recovery
        # window spans the same range) and take n/60 current densities. Since
        # Ea_eff(T) is J-independent, each additional J is a replicate that
        # tightens the slope estimate -- a clean "more data" comparison.
        k = int(round(n / len(df.T_K.unique())))
        k = max(1, min(k, len(J_all)))
        J_sel = J_all[np.linspace(0, len(J_all) - 1, k).round().astype(int)]
        sub = df[df.J_MA_cm2.isin(J_sel)]
        T, ea, _ = rec.recover_ea_eff(sub, None, half_window=5)
        metrics = rec.evaluate_ea_recovery(T, ea, phys.ea_effective)
        out.append({"n_samples": int(n), "rows": int(len(sub)),
                    "n_J": int(k), "n_T_unique": int(len(sub.T_K.unique())),
                    **metrics})
    return out
