"""Symbolic regression on the blind dataset (gplearn) and structural checks.

Two analyses:
  1. GP-SR in log space: ln(MTTF) ~ f(1/T, ln J) over a physically-motivated
     operator set. We report the recovered program, its complexity, and R^2.
  2. Structural sanity: does the best additive two-power-law (J^-2 + J^-1)
     model beat a single power law by BIC on the *blind* data? This is the
     quantitative test of whether the corrected structure is discoverable.
"""
import numpy as np
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

EXP = make_function(function=lambda x: np.exp(np.clip(x, -20, 20)), name="exp", arity=1)
FUNCTION_SET = ["add", "sub", "mul", "div", "log", "sqrt", "neg", "inv", EXP]
MAX_DEPTH = 7
POPULATION = 2500
GENERATIONS = 40


def run_gp_sr(df, seed=7, tournament_size=12):
    """GP symbolic regression of ln(MTTF) on features [1/T, ln J].

    The operator set includes exp so the additive (B0 J^-2 + C0 J^-1) structure
    is expressible in principle; this is the "do-everything" SR baseline whose
    recovered program we contrast with the interpretable physics-informed fit.
    """
    X = np.stack([1.0 / df.T_K.values, np.log(df.J_MA_cm2.values)], axis=1)
    y = np.log(df.MTTF_hours.values)
    est = SymbolicRegressor(population_size=POPULATION, generations=GENERATIONS,
                            tournament_size=tournament_size, function_set=FUNCTION_SET,
                            stopping_criteria=1e-4, p_crossover=0.7, p_subtree_mutation=0.1,
                            p_hoist_mutation=0.05, p_point_mutation=0.1,
                            max_samples=0.9, verbose=0, random_state=seed,
                            n_jobs=1, parsimony_coefficient=0.003)
    est.fit(X, y)
    program = str(est._program)
    r2 = est.score(X, y)
    complexity = est._program.length_
    return {"program": program, "r2": float(r2), "complexity": int(complexity),
            "seed": seed, "n_samples": int(len(df))}


def _bic(y, pred, k):
    n = len(y)
    rss = float(np.sum((y - pred) ** 2))
    return n * np.log(rss / n) + k * np.log(n), rss


def _fit_model(T, J, y, additive):
    """Fit ln MTTF = g(1/T) + h(J) on the blind data.

    g is a cubic in 1/T (flexible, shared); h is either the single power law
    -n ln J (incumbent Black's) or the additive nucleation+growth structure
    ln(B0 J^-2 + C0 J^-1) (the corrected model). The BIC gap isolates which
    current-density structure the data actually supports.
    """
    u = 1.0 / T
    G = np.stack([np.ones_like(u), u, u ** 2, u ** 3], axis=1)
    lnJ = np.log(J)

    if not additive:
        X = np.hstack([G, lnJ[:, None]])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        return X @ b, 5

    from scipy.optimize import least_squares
    def resid(p):
        b0, c0, g = p[0], p[1], p[2:]
        return G @ g + np.logaddexp(np.log(b0) - 2 * lnJ, np.log(c0) - lnJ) - y
    best = None
    for init in ([3e-6, 1e-6], [1e-3, 1e-4], [1e-7, 1e-7]):
        g0 = np.linalg.lstsq(G, y, rcond=None)[0]
        p0 = np.concatenate([init, g0])
        lb = np.concatenate([[1e-9, 1e-9], np.full(4, -1e12)])
        ub = np.concatenate([[1e3, 1e3], np.full(4, 1e12)])
        try:
            r = least_squares(resid, p0, bounds=(lb, ub), max_nfev=10000)
        except Exception:
            continue
        if best is None or r.cost < best[0]:
            best = (r.cost, r.x)
    if best is None:
        return None, None
    b0, c0, g = best[1][0], best[1][1], best[1][2:]
    return G @ g + np.logaddexp(np.log(b0) - 2 * lnJ, np.log(c0) - lnJ), 6


def structural_comparison(df):
    """BIC comparison on blind data: single power law vs additive two-power-law."""
    T = df.T_K.values; J = df.J_MA_cm2.values; y = np.log(df.MTTF_hours.values)
    p1, k1 = _fit_model(T, J, y, additive=False)
    p2, k2 = _fit_model(T, J, y, additive=True)
    bic1, rss1 = _bic(y, p1, k1)
    bic2, rss2 = _bic(y, p2, k2)
    return {"bic_single_powerlaw": bic1, "bic_additive_2term": bic2,
            "delta_bic_additive_minus_single": bic2 - bic1,
            "rss_single": rss1, "rss_additive": rss2,
            "n_single": int(k1), "n_additive": int(k2),
            "bic_weight_additive": float(1.0 / (1.0 + np.exp((bic2 - bic1) / 2)))}
