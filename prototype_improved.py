"""Prototype: improved formulas to resolve the extrapolation paradox.

Three directions, all on the EXISTING deterministic data (no new generation):

  P1  Separated-pathway formula: MTTF = B' J^-2 e^{Ea_s/kT} + C' J^-1 e^{Ea_gb/kT}.
      Each J-regime carries its own activation energy, so Ea_s/Ea_gb are
      identified from the current-density dimension instead of the (truncated)
      temperature window. Is identifiability alone enough for extrapolation?
  P2  Envelope-depth sweep: train window shrinks T >= {523,548,573,623} K, so the
      extrapolation gets longer and the surface pathway is progressively less
      identified. Does the degeneracy worsen with distance?
  P3  Estimator fixes on the joint formula: (a) soft Gaussian prior on ln D0r
      (not a hard pin), (b) physically-grounded box constraints on Ea_s, Ea_gb,
      D0r. Can the gap 10.4x -> 1.4x be closed without knowing D0r exactly?

Writes results/prototype_improved.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_dataset as gen
from benchmarks import eval as E
from benchmarks import models as M

KB = 8.617333262e-5
ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")


# ---------------------------------------------------------------------------
# P1: separated-pathway formula (4 identifiable params; no D0r at all)
# ---------------------------------------------------------------------------
class SeparatedPathway:
    name = "P1 separated: B'J^-2 e(Eas/kT)+C'J^-1 e(Eagb/kT)"

    def __init__(self, eas_bounds=(0.4, 1.6), eagb_bounds=(0.4, 1.6)):
        self.eas_bounds = eas_bounds
        self.eagb_bounds = eagb_bounds

    def fit(self, T, J, y, L=None):
        u = 1.0 / KB / np.asarray(T, float)
        lnJ = np.log(np.asarray(J, float))
        y = np.asarray(y, float)

        def resid(p):
            lb, lc, ea1, ea2 = p
            return np.logaddexp(lb - 2 * lnJ + ea1 * u,
                                lc - lnJ + ea2 * u) - y

        lbnd = [-30, -30, self.eas_bounds[0], self.eagb_bounds[0]]
        ubnd = [-2, -2, self.eas_bounds[1], self.eagb_bounds[1]]
        best = None
        for s in [(-12.5, -23, 0.8, 1.2), (-10, -20, 0.7, 1.3),
                  (-15, -26, 0.9, 1.1), (-12, -16, 1.0, 1.0)]:
            try:
                r = least_squares(resid, s, bounds=(lbnd, ubnd), max_nfev=20000)
            except Exception:
                continue
            if best is None or r.cost < best[0]:
                best = (r.cost, r.x)
        if best is None:
            raise RuntimeError("separated fit failed")
        self.p = best[1]

    def predict(self, T, J, L=None):
        u = 1.0 / KB / np.asarray(T, float)
        lnJ = np.log(np.asarray(J, float))
        lb, lc, ea1, ea2 = self.p
        return np.logaddexp(lb - 2 * lnJ + ea1 * u, lc - lnJ + ea2 * u)


# ---------------------------------------------------------------------------
# P3: joint corrected formula with flexible bounds / soft prior on ln D0r
# ---------------------------------------------------------------------------
class JointFlex:
    """Joint model ln MTTF = ln(B0 J^-2 + C0 J^-1) - ln D_eff(T), with
    D_eff(T) = exp(-Ea_s/kT) + D0r exp(-Ea_gb/kT). Parameters
    (ln B0, ln C0, Ea_s, Ea_gb, ln D0r)."""

    def __init__(self, name, bounds=None, prior=None):
        self.name = name
        self.bounds = bounds  # dict: eas, eagb, lnr as (lo, hi)
        self.prior = prior    # (lnr_mean, lnr_sd) soft Gaussian

    def fit(self, T, J, y, L=None):
        u = 1.0 / KB / np.asarray(T, float)
        lnJ = np.log(np.asarray(J, float))
        y = np.asarray(y, float)
        b = self.bounds or {}
        eas = b.get("eas", (0.4, 1.6))
        eagb = b.get("eagb", (0.4, 1.6))
        lnr = b.get("lnr", (0.0, 20.0))
        lbnd = np.array([-25, -25, eas[0], eagb[0], lnr[0]])
        ubnd = np.array([-4, -4, eas[1], eagb[1], lnr[1]])

        def resid(p):
            lb, lc, ea1, ea2, lnr0 = p
            lnJt = np.logaddexp(lb - 2 * lnJ, lc - lnJ)
            lnD = np.logaddexp(-ea1 * u, lnr0 - ea2 * u)
            r = lnJt - lnD - y
            if self.prior is not None:
                m, sd = self.prior
                r = np.concatenate([r, [(lnr0 - m) / sd]])
            return r

        starts = [(-12, -14, 0.8, 1.2, 9.4), (-10, -16, 0.7, 1.3, 10.0),
                  (-14, -12, 0.9, 1.1, 8.0), (-12, -12, 0.8, 1.2, 12.0)]
        best = None
        for s in starts:
            try:
                r = least_squares(resid, s, bounds=(lbnd, ubnd), max_nfev=20000)
            except Exception:
                continue
            if best is None or r.cost < best[0]:
                best = (r.cost, r.x)
        if best is None:
            raise RuntimeError("joint-flex fit failed")
        self.p = best[1]

    def params(self):
        lb, lc, ea1, ea2, lnr0 = self.p
        return {"B0": float(np.exp(lb)), "C0": float(np.exp(lc)),
                "Ea_s": float(ea1), "Ea_gb": float(ea2),
                "D0r": float(np.exp(lnr0))}

    def predict(self, T, J, L=None):
        u = 1.0 / KB / np.asarray(T, float)
        lnJ = np.log(np.asarray(J, float))
        lb, lc, ea1, ea2, lnr0 = self.p
        lnJt = np.logaddexp(lb - 2 * lnJ, lc - lnJ)
        lnD = np.logaddexp(-ea1 * u, lnr0 - ea2 * u)
        return lnJt - lnD


class JointFlexBlech(JointFlex):
    """Joint model + Blech critical product: ln MTTF = ln(B0 J^-2 + C0 J^-1)
    - ln D_eff(T) + ln(jL) - ln(jL - (jL)_c). Parameter (jL)_c fitted in log."""

    def __init__(self, name, bounds=None, prior=None, jlc_bounds=(np.log(1e3), np.log(1e4))):
        super().__init__(name, bounds, prior)
        self.jlc_bounds = jlc_bounds

    def fit(self, T, J, y, L=None):
        u = 1.0 / KB / np.asarray(T, float)
        lnJ = np.log(np.asarray(J, float))
        jL = np.asarray(J, float) * 1e4 * np.asarray(L, float)
        lnjL = np.log(jL)
        y = np.asarray(y, float)
        b = self.bounds or {}
        eas = b.get("eas", (0.4, 1.6))
        eagb = b.get("eagb", (0.4, 1.6))
        lnr = b.get("lnr", (0.0, 20.0))
        lbnd = np.array([-25, -25, eas[0], eagb[0], lnr[0], self.jlc_bounds[0]])
        ubnd = np.array([-4, -4, eas[1], eagb[1], lnr[1], self.jlc_bounds[1]])

        def resid(p):
            lb, lc, ea1, ea2, lnr0, ljlc = p
            lnJt = np.logaddexp(lb - 2 * lnJ, lc - lnJ)
            lnD = np.logaddexp(-ea1 * u, lnr0 - ea2 * u)
            lnboost = lnjL - ljlc - np.log(np.expm1(lnjL - ljlc))
            r = lnJt - lnD + lnboost - y
            if self.prior is not None:
                m, sd = self.prior
                r = np.concatenate([r, [(lnr0 - m) / sd]])
            return r

        starts = [(-12, -14, 0.8, 1.2, 9.4, np.log(3000.0)),
                  (-10, -16, 0.7, 1.3, 10.0, np.log(2500.0)),
                  (-14, -12, 0.9, 1.1, 8.0, np.log(3500.0))]
        best = None
        for s in starts:
            try:
                r = least_squares(resid, s, bounds=(lbnd, ubnd), max_nfev=20000)
            except Exception:
                continue
            if best is None or r.cost < best[0]:
                best = (r.cost, r.x)
        if best is None:
            raise RuntimeError("joint-flex-Blech fit failed")
        self.p = best[1]

    def params(self):
        p = super().params()
        p["jLc"] = float(np.exp(self.p[5]))
        return p

    def predict(self, T, J, L=None):
        u = 1.0 / KB / np.asarray(T, float)
        lnJ = np.log(np.asarray(J, float))
        jL = np.asarray(J, float) * 1e4 * np.asarray(L, float)
        lnjL = np.log(jL)
        lb, lc, ea1, ea2, lnr0, ljlc = self.p
        lnJt = np.logaddexp(lb - 2 * lnJ, lc - lnJ)
        lnD = np.logaddexp(-ea1 * u, lnr0 - ea2 * u)
        lnboost = lnjL - ljlc - np.log(np.expm1(lnjL - ljlc))
        return lnJt - lnD + lnboost


def main():
    db = pd.read_csv(os.path.join(ROOT, "data", "em_synthetic_blind.csv"))

    # --- reference models (already in the paper) --------------------------
    refs = {
        "joint free D0r (paper)": M.AdditiveModel(),
        "joint D0r pinned (paper)": M.AdditiveModel(fix_D0r=1.2e4),
    }
    constrained = JointFlex(
        "P3b joint + physical box constraints (Eas 0.6-1.0, Eagb 1.0-1.4, D0r 1e2-1e6)",
        bounds={"eas": (0.6, 1.0), "eagb": (1.0, 1.4), "lnr": (np.log(1e2), np.log(1e6))})
    regularized = JointFlex(
        "P3a joint + soft prior lnD0r ~ N(ln 1.2e4, sd 1.5)",
        prior=(np.log(1.2e4), 1.5))
    separated = SeparatedPathway()

    models = {"P1 separated-pathway": separated,
              "P3a regularized joint": regularized,
              "P3b box-constrained joint": constrained}
    models.update({k: v for k, v in refs.items()})

    # --- base extrapolation (the paradox harness) -------------------------
    res = {}
    tr = db[db.T_K >= 523.0]
    Tac, Jac = tr.T_K.values, tr.J_MA_cm2.values
    yac = np.log(tr.MTTF_hours.values)
    Tuse = np.array([373.0, 373.0, 373.0])
    Juse = np.array([0.1, 0.5, 1.0])
    true_log = np.log(gen.black_core_mttf(Tuse, Juse)[0])

    res["base_extrapolation"] = {}
    for name, m in models.items():
        m.fit(Tac, Jac, yac)
        fit = float(np.sqrt(np.mean((m.predict(Tac, Jac) - yac) ** 2)))
        fo = E.fold_error(m.predict(Tuse, Juse), true_log)
        pars = {}
        if isinstance(m, JointFlex):
            pars = {k: (round(v, 4) if k.startswith("Ea") else round(v, 5))
                    for k, v in m.params().items()}
        elif isinstance(m, SeparatedPathway):
            pars = {"Ea_s": round(m.p[2], 4), "Ea_gb": round(m.p[3], 4)}
        elif isinstance(m, M.AdditiveModel):
            p = m.params()
            pars = {"Ea_s": round(p["Ea_surface_eV"], 4),
                    "Ea_gb": round(p["Ea_grain_boundary_eV"], 4),
                    "D0r": round(float(np.exp(p["D0_GB_over_surface"])), 4)}
        res["base_extrapolation"][name] = {
            "in_sample_ln_rmse": round(fit, 4),
            "fold_factor": round(fo["fold_factor"], 3),
            "median_log_ratio": round(fo["median_log_ratio"], 3),
            "fitted": pars,
        }
        print(f"{name[:64]:64s} fit {fit:.4f}  fold {fo['fold_factor']:6.2f}x  "
              f"logr {fo['median_log_ratio']:+.2f}  {pars}")

    # --- P2: envelope-depth sweep -----------------------------------------
    res["envelope_depth"] = {}
    for tmin in (523.0, 548.0, 573.0, 623.0):
        tr = db[db.T_K >= tmin]
        Tac, Jac = tr.T_K.values, tr.J_MA_cm2.values
        yac = np.log(tr.MTTF_hours.values)
        row = {"fit_n": int(len(tr))}
        for name, m in models.items():
            m.fit(Tac, Jac, yac)
            fo = E.fold_error(m.predict(Tuse, Juse), true_log)
            row[name] = round(fo["fold_factor"], 3)
        res["envelope_depth"][f"T>=_{int(tmin)}_K"] = row
        print(f"\nenvelope T>={int(tmin)}K n={len(tr)}: " +
              ", ".join(f"{k.split()[0]}={v}x" for k, v in row.items()
                        if isinstance(v, float)))

    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "prototype_improved.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote results/prototype_improved.json")


def main_extended():
    """Extended (length-dependent) check: does the soft prior work with Blech?"""
    de = pd.read_csv(os.path.join(ROOT, "data", "em_synthetic_extended_full_with_groundtruth.csv"))
    tr = de[(de.T_K >= 523.0) & (~de.immortal) & de.MTTF_hours.notna()]
    Ttr, Jtr, Ltr = tr.T_K.values, tr.J_MA_cm2.values, tr.L_um.values
    ytr = np.log(tr.MTTF_hours.values)
    Tuse = np.array([373.0, 373.0, 373.0, 373.0])
    Juse = np.array([8.0, 8.0, 8.0, 8.0])
    Luse = np.array([400.0, 500.0, 600.0, 600.0])
    true = []
    for T, J, L in zip(Tuse, Juse, Luse):
        mttf, _, _ = gen.black_core_mttf(T, J)
        true.append(np.log(mttf * gen.blech_boost(gen.jl_product(J, L))))

    models = {
        "Blech free D0r (paper)": M.AdditiveBlech(),
        "Blech D0r pinned (paper)": M.AdditiveBlech(fix_D0r=1.2e4),
        "Blech P3a soft prior sd=5": JointFlexBlech(
            "Blech soft prior", prior=(np.log(1.2e4), 5.0)),
    }
    print("\nextended (Blech) extrapolation at 100 C:")
    out = {}
    for name, m in models.items():
        m.fit(Ttr, Jtr, ytr, Ltr)
        fo = E.fold_error(m.predict(Tuse, Juse, Luse), np.array(true))
        out[name] = {"fold_factor": round(fo["fold_factor"], 3),
                     "median_log_ratio": round(fo["median_log_ratio"], 3)}
        print(f"  {name[:56]:56s} fold={fo['fold_factor']:.2f}x "
              f"logr={fo['median_log_ratio']:+.2f}")
    return out


if __name__ == "__main__":
    main()
    out = main_extended()
    path = os.path.join(RES, "prototype_improved.json")
    data = json.load(open(path))
    data["extended_blech"] = out
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"extended results appended to {path}")


def main_quantitative():
    """Quantitative laws.

    A. Extrapolation law: log(fold error) = dEa_transfer * (1/T_use - 1/T_min)/kB
       per model, with the 2x safe-extrapolation temperature.
    B. Sample-size law: per-sample structural evidence A(sigma) (dBIC ~ -A n + ln n
       above the 3-level identifiability floor), with A ~ c/sigma^2.
    """
    from benchmarks import recovery as rec
    db = pd.read_csv(os.path.join(ROOT, "data", "em_synthetic_blind.csv"))
    Tac = db[db.T_K >= 523.0].T_K.values
    Jac = db[db.T_K >= 523.0].J_MA_cm2.values
    yac = np.log(db[db.T_K >= 523.0].MTTF_hours.values)
    mods = {"Black's eq": M.BlackMLE(), "Two-regime": M.TwoRegimeBlack(),
            "free D0r": M.AdditiveModel(), "pinned D0r": M.AdditiveModel(fix_D0r=1.2e4)}
    for m in mods.values():
        m.fit(Tac, Jac, yac)

    Tmin = 523.0
    law = {}
    for name, m in mods.items():
        sl, ls = [], []
        for Tu in (373.0, 400.0, 425.0, 450.0, 475.0, 494.0):
            Tl = np.array([Tu, Tu, Tu]); Jl = np.array([0.1, 0.5, 1.0])
            true = np.log(gen.black_core_mttf(Tl, Jl)[0])
            fo = E.fold_error(m.predict(Tl, Jl), true)
            ls.append(fo["median_log_ratio"])
            sl.append(1.0 / Tu - 1.0 / Tmin)
        sl, ls = np.array(sl), np.array(ls)
        slope = np.polyfit(sl, ls, 1)[0]
        dea = slope * KB
        r2 = np.corrcoef(sl, ls)[0, 1] ** 2
        d1t_safe = KB * np.log(2.0) / max(abs(dea), 1e-6)
        t_safe = 1.0 / (1.0 / Tmin + d1t_safe)
        law[name] = {"dEa_transfer_eV": round(dea, 4),
                     "logfold_linear_R2": round(r2, 3),
                     "safe_to_2x_T_K": round(float(t_safe), 1),
                     "safe_to_2x_T_C": round(float(t_safe) - 273.15, 1)}
        print(f"A. {name:12s} dEa_transfer={dea:+.3f} eV  R2={r2:.3f}  "
              f"safe-to-2x: {t_safe:.0f} K ({t_safe - 273.15:.0f} C)")

    size_law = {}
    for sigma in (0.05, 0.15, 0.30):
        dfb, _ = gen.generate(noise_sigma=sigma)
        res = E.structure_size_sweep(dfb, gen)
        sweep = res["sweep"]
        n = np.array([r["n_samples"] for r in sweep])
        m = np.array([r["delta_bic_additive_minus_single_mean"] for r in sweep])
        k = np.array([r["n_J_levels"] for r in sweep])
        id_floor = m[k <= 2].mean()  # should be ~ +ln n, sigma-independent
        A = np.polyfit(n[k >= 3], m[k >= 3], 1)[0]
        size_law[sigma] = {
            "identifiability_floor_dBIC": round(float(id_floor), 2),
            "per_sample_evidence_A": round(float(-A), 4),
            "A_sigma2": round(float(-A * sigma ** 2), 4),
        }
        print(f"B. sigma={sigma:.2f} floor_dBIC(<=2J)={id_floor:+.2f}  "
              f"A (per-sample evidence)={-A:.4f}  A*sigma^2={-A * sigma ** 2:.4f}")

    out = {"extrapolation_law": law, "sample_size_law": size_law}
    with open(os.path.join(RES, "prototype_quantitative.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote results/prototype_quantitative.json")


if __name__ == "__main__":
    main_quantitative()
