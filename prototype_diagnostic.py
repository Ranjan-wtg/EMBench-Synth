"""Diagnostic-law experiment: can the extrapolation hazard be certified from
the accelerated-window fit ALONE (without knowing the truth)?

Method: for each (model, window) fit, compute the 1-D profile likelihood of the
low-T-relevant parameter ln D0r (the unidentifiable prefactor direction): scan
ln D0r, refit the other 4 parameters, record RSS. The width of the profile in
the extrapolation direction (projected onto Ea_eff at 100 C) is the *predicted*
hazard. If the predicted hazard correlates with the actual use-condition error
across all fits, the hazard is certifiable in-window -- a falsifiable law.

Models: joint formula with (a) free D0r, (b) soft prior on ln D0r,
(c) pinned D0r. Windows: T >= {523, 548, 573, 623} K.
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

KB = 8.617333262e-5
ROOT = os.path.dirname(os.path.abspath(__file__))

TRUE_D0R = 1.2e4
LB = np.array([-25.0, -25.0, 0.4, 0.4, 0.0])
UB = np.array([-4.0, -4.0, 1.6, 1.6, 15.0])
STARTS = [(-12.4, -13.8, 0.8, 1.2, 9.4), (-11.5, -13.1, 0.75, 1.25, 10.0),
          (-13.1, -14.0, 0.85, 1.15, 9.0)]


def fit_joint(T, J, y, prior=None, fix_lnr=None):
    u = 1.0 / KB / np.asarray(T, float)
    lnJ = np.log(np.asarray(J, float))
    y = np.asarray(y, float)
    if fix_lnr is None:
        lb, ub = LB, UB
        starts = STARTS

        def resid(p):
            l0, c0, ea1, ea2, lnr = p
            lnJt = np.logaddexp(l0 - 2 * lnJ, c0 - lnJ)
            lnD = np.logaddexp(-ea1 * u, lnr - ea2 * u)
            r = lnJt - lnD - y
            if prior is not None:
                m, sd = prior
                r = np.concatenate([r, [(lnr - m) / sd]])
            return r
    else:
        lnr = float(fix_lnr)
        lb, ub = LB[:4], UB[:4]
        starts = [s[:4] for s in STARTS]

        def resid(p):
            l0, c0, ea1, ea2 = p
            lnJt = np.logaddexp(l0 - 2 * lnJ, c0 - lnJ)
            lnD = np.logaddexp(-ea1 * u, lnr - ea2 * u)
            return lnJt - lnD - y

    best = None
    for s in starts:
        try:
            r = least_squares(resid, s, bounds=(lb, ub), max_nfev=20000)
        except Exception:
            continue
        if best is None or r.cost < best[0]:
            best = (r.cost, r.x)
    if best is None:
        raise RuntimeError("fit failed")
    p = list(best[1])
    if fix_lnr is not None:
        p.append(lnr)
    return np.array(p)


def ea_eff_params(p, T):
    """Ea_eff at temperature T from fitted params (lnB0,lnC0,Eas,Eagb,lnD0r)."""
    _, _, ea1, ea2, lnr = p
    u = 1.0 / KB / T
    ws = np.exp(-ea1 * u)
    wg = np.exp(lnr) * np.exp(-ea2 * u)
    return (ws * ea1 + wg * ea2) / (ws + wg)


def profile_width(T, J, y, prior, lnr_grid):
    """Profile likelihood of ln D0r -> predicted hazard width of Ea_eff(100C)."""
    out = []
    for lnr in lnr_grid:
        p = fit_joint(T, J, y, prior=prior, fix_lnr=lnr)
        u = 1.0 / KB / np.asarray(T, float)
        lnJ = np.log(np.asarray(J, float))
        lnJt = np.logaddexp(p[0] - 2 * lnJ, p[1] - lnJ)
        lnD = np.logaddexp(-p[2] * u, lnr - p[3] * u)
        rss = float(np.sum((lnJt - lnD - y) ** 2))
        out.append((lnr, rss, ea_eff_params(p, 373.0)))
    out = np.array([(r, rss, ea) for r, rss, ea in out], dtype=float)
    rss_min = out[:, 1].min()
    dll = 0.5 * len(y) * (out[:, 1] - rss_min) / rss_min
    in_range = dll <= 1.0  # ~1-sigma profile region
    ea = out[:, 2]
    return {"hazard_width_std": float(np.ptp(ea[in_range]) / 2.0),
            "ea_eff_100C_range": [float(ea[in_range].min()),
                                  float(ea[in_range].max())],
            "n_profile_points": int(in_range.sum())}


def main():
    db = pd.read_csv(os.path.join(ROOT, "data", "em_synthetic_blind.csv"))
    Tuse = np.array([373.0, 373.0, 373.0])
    Juse = np.array([0.1, 0.5, 1.0])
    true_log = np.log(gen.black_core_mttf(Tuse, Juse)[0])
    true_ea_use = float(gen.ea_effective(373.0))
    lnr_grid = np.linspace(0.0, 15.0, 31)

    rows = []
    for tmin in (523.0, 548.0, 573.0, 623.0):
        tr = db[db.T_K >= tmin]
        T, J = tr.T_K.values, tr.J_MA_cm2.values
        y = np.log(tr.MTTF_hours.values)
        for tag, prior, fix in [("free", None, None),
                                ("soft prior", (np.log(TRUE_D0R), 5.0), None),
                                ("pinned", None, np.log(TRUE_D0R))]:
            p = fit_joint(T, J, y, prior=prior, fix_lnr=fix)
            pred = np.array([np.logaddexp(p[0] - 2 * np.log(jj),
                                          p[1] - np.log(jj))
                             - np.logaddexp(-p[2] / (KB * Tu),
                                            p[4] - p[3] / (KB * Tu))
                             for Tu, jj in zip(Tuse, Juse)])
            fold = E.fold_error(pred, true_log)
            pred_ea = ea_eff_params(p, 373.0)
            pw = profile_width(T, J, y, prior, lnr_grid)
            rows.append({
                "window_Tmin": int(tmin), "n": int(len(tr)), "model": tag,
                "fold_factor": round(fold["fold_factor"], 3),
                "actual_ea_eff_100C_err_eV": round(float(pred_ea - true_ea_use), 3),
                **{k: round(v, 4) if isinstance(v, float) else v
                   for k, v in pw.items()},
            })
            print(f"T>={int(tmin):3d}K n={len(tr):3d} {tag:10s} "
                  f"pred_hazard={pw['hazard_width_std']:.3f} eV  "
                  f"actual_ea_err={pred_ea - true_ea_use:+.3f} eV  "
                  f"fold={fold['fold_factor']:6.2f}x")

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    path = os.path.join(ROOT, "results", "prototype_diagnostic.json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)

    hazard = np.array([r["hazard_width_std"] for r in rows])
    aerr = np.abs(np.array([r["actual_ea_eff_100C_err_eV"] for r in rows]))
    fold = np.array([r["fold_factor"] for r in rows])
    print(f"\ncorr(predicted hazard width, |actual Ea transfer err|) = "
          f"{np.corrcoef(hazard, aerr)[0, 1]:.3f}")
    print(f"corr(predicted hazard width, fold error) = "
          f"{np.corrcoef(hazard, fold)[0, 1]:.3f}")
    print(f"corr(|actual Ea err|, fold) = {np.corrcoef(aerr, fold)[0, 1]:.3f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
