"""Fitted candidate models used across the benchmarks.

All models predict ln(MTTF) from physical inputs (T in K, J in MA/cm^2,
optionally L in um). They range from the industry incumbent (Black's equation
with constant n and Ea) through the structure-informed "corrected equation"
(additive nucleation+growth with a two-regime activation energy and a Blech
divergence term) to black-box regressors.
"""
import numpy as np
from scipy.optimize import least_squares
from sklearn.neural_network import MLPRegressor

kB_EV = 8.617333262e-5


def _jl(J, L):
    return np.asarray(J) * 1e6 * np.asarray(L) * 1e-4


# ---------------------------------------------------------------------------
# Incumbent: Black's equation with constant n and Ea (linear in log space)
# ---------------------------------------------------------------------------
class BlackMLE:
    name = "Black's equation (constant n, Ea) - industry incumbent"

    def fit(self, T, J, y, L=None):
        T = np.asarray(T, float); J = np.asarray(J, float); y = np.asarray(y, float)
        A = np.stack([np.ones_like(T), np.log(J), 1.0 / T], axis=1)
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        self.beta = beta                     # [lnA, -n, Ea/k]
        self.n = -beta[1]
        self.Ea_eV = beta[2] * kB_EV
        self.n_params = 3
        return self

    def predict(self, T, J, L=None):
        T = np.asarray(T, float); J = np.asarray(J, float)
        return self.beta[0] - self.n * np.log(J) + self.Ea_eV / (kB_EV * T)

    def params(self):
        return {"n": float(self.n), "Ea_eV": float(self.Ea_eV)}


# ---------------------------------------------------------------------------
# Piecewise two-regime Black's equation (what engineers do today)
# ---------------------------------------------------------------------------
class TwoRegimeBlack:
    name = "Two-regime Black's equation (piecewise Ea)"

    def fit(self, T, J, y, L=None):
        T = np.asarray(T, float); J = np.asarray(J, float); y = np.asarray(y, float)
        cands = np.percentile(T, np.arange(15, 86, 5))
        best = None
        for tstar in cands:
            lo, hi = T <= tstar, T > tstar
            rss = 0.0
            fits = []
            for m in (lo, hi):
                A = np.stack([np.ones_like(T[m]), np.log(J[m]), 1.0 / T[m]], axis=1)
                b, *_ = np.linalg.lstsq(A, y[m], rcond=None)
                fits.append(b)
                rss += np.sum((A @ b - y[m]) ** 2)
            if best is None or rss < best[0]:
                best = (rss, tstar, fits)
        self.tstar = best[1]
        self.beta_lo, self.beta_hi = best[2]
        self.n_params = 7   # 3 + 3 + threshold
        return self

    def predict(self, T, J, L=None):
        T = np.asarray(T, float); J = np.asarray(J, float)
        out = np.empty_like(T)
        for mask, beta in ((T <= self.tstar, self.beta_lo), (T > self.tstar, self.beta_hi)):
            out[mask] = beta[0] + beta[1] * np.log(J[mask]) + beta[2] / T[mask]
        return out

    def params(self):
        return {"T_cut_K": float(self.tstar),
                "Ea_lo_eV": float(self.beta_lo[2] * kB_EV),
                "Ea_hi_eV": float(self.beta_hi[2] * kB_EV),
                "n_lo": float(-self.beta_lo[1]),
                "n_hi": float(-self.beta_hi[1])}


# ---------------------------------------------------------------------------
# Structure-informed "corrected equation": additive nucleation+growth with a
# smooth two-regime Ea(T) (and optional Blech divergence term).
# ---------------------------------------------------------------------------
def _ea_logistic(T, Ea1, Ea2, Tc, s):
    return Ea1 + (Ea2 - Ea1) / (1.0 + np.exp((T - Tc) / s))


def _lse2(a, b):
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def _add_mu(p, T, J):
    """ln MTTF = ln(B0 J^-2 + C0 J^-1) - ln D_eff(T).

    D_eff(T) = exp(-Ea_s/kT) + D0r * exp(-Ea_gb/kT)   (surface + grain-boundary
    Arrhenius pathways, surface prefactor fixed to 1 -- overall scale is
    absorbed into B0). This matches the generative model's structure.
    """
    lnB0, lnC0, Ea_s, Ea_gb, lnD0r = p
    lnD = _lse2(-Ea_s / (kB_EV * T), lnD0r - Ea_gb / (kB_EV * T))
    return -lnD + np.logaddexp(lnB0 - 2 * np.log(J), lnC0 - np.log(J))


class AdditiveModel:
    name = "Corrected: additive nucleation+growth, logsumexp two-regime Ea"

    def __init__(self, fix_D0r=None):
        """fix_D0r: pin the D0_GB/D0_surface prefactor ratio at a literature
        value (known material constant). None => fit it (recovery mode)."""
        self.fix_D0r = fix_D0r
        if fix_D0r is None:
            self.name = (self.name + " (free D0r, data-driven)")
        else:
            self.name = self.name + (f" (D0r fixed={fix_D0r:g})")

    def fit(self, T, J, y, L=None):
        T = np.asarray(T, float); J = np.asarray(J, float); y = np.asarray(y, float)
        if self.fix_D0r is not None:
            lnD0r = np.log(self.fix_D0r)
            starts = [
                (np.log(4e-6), np.log(1e-6), 0.8, 1.2),
                (np.log(1e-5), np.log(2e-6), 0.75, 1.25),
                (np.log(2e-6), np.log(8e-7), 0.85, 1.15),
            ]
            lb = np.array([-25.0, -25.0, 0.4, 0.4])
            ub = np.array([-4.0, -4.0, 1.6, 1.6])
            n = 4
        else:
            lnD0r = None
            starts = [
                (np.log(4e-6), np.log(1e-6), 0.8, 1.2, np.log(1.2e4)),
                (np.log(1e-5), np.log(2e-6), 0.75, 1.25, np.log(2e4)),
                (np.log(2e-6), np.log(8e-7), 0.85, 1.15, np.log(8e3)),
            ]
            lb = np.array([-25.0, -25.0, 0.4, 0.4, 0.0])
            ub = np.array([-4.0, -4.0, 1.6, 1.6, 15.0])
            n = 5
        best = None
        for init in starts:
            p0 = np.array(init)
            try:
                if lnD0r is not None:
                    res = least_squares(
                        lambda p: _add_mu(np.concatenate([p, [lnD0r]]), T, J) - y,
                        p0, bounds=(lb, ub), max_nfev=8000)
                else:
                    res = least_squares(lambda p: _add_mu(p, T, J) - y, p0,
                                        bounds=(lb, ub), max_nfev=8000)
            except Exception:
                continue
            if best is None or res.cost < best.cost:
                best = res
        self.p = best.x
        self._lnD0r = lnD0r
        self.n_params = n
        return self

    def predict(self, T, J, L=None):
        if self._lnD0r is not None:
            p = np.concatenate([self.p, [self._lnD0r]])
        else:
            p = self.p
        return _add_mu(p, T, J)

    def ea_eff(self, T):
        if self._lnD0r is not None:
            Ea_s, Ea_gb = self.p[2], self.p[3]
            lnD0r = self._lnD0r
        else:
            Ea_s, Ea_gb, lnD0r = self.p[2], self.p[3], self.p[4]
        T = np.asarray(T, float)
        w = np.exp(-Ea_s / (kB_EV * T))
        wg = np.exp(lnD0r - Ea_gb / (kB_EV * T))
        return (w * Ea_s + wg * Ea_gb) / (w + wg)

    def params(self):
        if self._lnD0r is not None:
            lnB0, lnC0, Ea_s, Ea_gb = self.p
            lnD0r = self._lnD0r
        else:
            lnB0, lnC0, Ea_s, Ea_gb, lnD0r = self.p
        return {"B0": float(np.exp(lnB0)), "C0": float(np.exp(lnC0)),
                "Ea_surface_eV": float(Ea_s), "Ea_grain_boundary_eV": float(Ea_gb),
                "D0_GB_over_surface": float(np.exp(lnD0r))}


def _add_blech_mu(p, T, J, L):
    lnB0, lnC0, Ea_s, Ea_gb, lnD0r, lnjlc = p
    jlc = np.exp(lnjlc)
    jl = _jl(J, L)
    return _add_mu(p[:5], T, J) + np.log(jl) - np.log(jl - jlc)


class AdditiveBlech:
    name = "Corrected+Blech: additive nucleation+growth, two-regime Ea, (jL)_c"

    def __init__(self, fix_D0r=None):
        self.fix_D0r = fix_D0r

    def fit(self, T, J, y, L=None):
        T = np.asarray(T, float); J = np.asarray(J, float); L = np.asarray(L, float)
        y = np.asarray(y, float)
        jl = _jl(J, L)
        assert (jl > 3000 * 0.5).all(), "fit on mortal rows only"
        if self.fix_D0r is not None:
            lnD0r = np.log(self.fix_D0r)
            starts = [
                (np.log(4e-6), np.log(1e-6), 0.8, 1.2, np.log(3000.0)),
                (np.log(1e-5), np.log(2e-6), 0.75, 1.25, np.log(3200.0)),
                (np.log(2e-6), np.log(8e-7), 0.85, 1.15, np.log(2800.0)),
            ]
            lb = np.array([-25.0, -25.0, 0.4, 0.4, np.log(2000.0)])
            ub = np.array([-4.0, -4.0, 1.6, 1.6, np.log(5000.0)])
            n = 5
        else:
            lnD0r = None
            starts = [
                (np.log(4e-6), np.log(1e-6), 0.8, 1.2, np.log(1.2e4), np.log(3000.0)),
                (np.log(1e-5), np.log(2e-6), 0.75, 1.25, np.log(2e4), np.log(3200.0)),
                (np.log(2e-6), np.log(8e-7), 0.85, 1.15, np.log(8e3), np.log(2800.0)),
            ]
            lb = np.array([-25.0, -25.0, 0.4, 0.4, 0.0, np.log(2000.0)])
            ub = np.array([-4.0, -4.0, 1.6, 1.6, 15.0, np.log(5000.0)])
            n = 6
        best = None
        for init in starts:
            p0 = np.array(init)
            try:
                if lnD0r is not None:
                    res = least_squares(
                        lambda p: _add_blech_mu(
                            np.array([p[0], p[1], p[2], p[3], lnD0r, p[4]]),
                            T, J, L) - y, p0, bounds=(lb, ub), max_nfev=10000)
                else:
                    res = least_squares(lambda p: _add_blech_mu(p, T, J, L) - y,
                                        p0, bounds=(lb, ub), max_nfev=10000)
            except Exception:
                continue
            if best is None or res.cost < best.cost:
                best = res
        self.p = best.x
        self._lnD0r = lnD0r
        self.n_params = n
        return self

    def predict(self, T, J, L=None):
        if self._lnD0r is not None:
            p = np.array([self.p[0], self.p[1], self.p[2], self.p[3],
                          self._lnD0r, self.p[4]])
        else:
            p = self.p
        return _add_blech_mu(p, np.asarray(T, float), np.asarray(J, float),
                             np.asarray(L, float))

    def jl_c(self):
        return float(np.exp(self.p[-1]))

    def params(self):
        lnB0, lnC0, Ea_s, Ea_gb = self.p[:4]
        if self._lnD0r is not None:
            lnD0r = self._lnD0r
            lnjlc = self.p[4]
        else:
            lnD0r, lnjlc = self.p[4], self.p[5]
        return {"B0": float(np.exp(lnB0)), "C0": float(np.exp(lnC0)),
                "Ea_surface_eV": float(Ea_s), "Ea_grain_boundary_eV": float(Ea_gb),
                "D0_GB_over_surface": float(np.exp(lnD0r)),
                "jL_c_A_per_cm": float(np.exp(lnjlc))}


# ---------------------------------------------------------------------------
# Black-box baseline: MLP on physically-motivated features
# ---------------------------------------------------------------------------
class MLPBaseline:
    name = "Black-box MLP baseline"

    def __init__(self, extended=False):
        self.extended = extended
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.pipe = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=3000,
                         random_state=0, early_stopping=True, alpha=1e-4))

    def _X(self, T, J, L):
        T = np.asarray(T, float); J = np.asarray(J, float)
        cols = [1.0 / T, np.log(J)]
        if self.extended:
            cols.append(np.log(np.asarray(L, float)))
        return np.stack(cols, axis=1)

    def fit(self, T, J, y, L=None):
        self.pipe.fit(self._X(T, J, L), y)
        return self

    def predict(self, T, J, L=None):
        return self.pipe.predict(self._X(T, J, L))

    def params(self):
        return {"hidden": "64-64"}


def _torch():
    import torch
    return torch


def _batched_mu_torch(p5, T, J):
    """Torch counterpart of _add_mu (batched, logsumexp-stable)."""
    torch = _torch()
    lnB0, lnC0 = p5[:, 0], p5[:, 1]
    Ea_s, Ea_gb, lnD0r = p5[:, 2], p5[:, 3], p5[:, 4]
    a = -Ea_s / (kB_EV * T)
    b = lnD0r - Ea_gb / (kB_EV * T)
    mx = torch.maximum(a, b)
    lnD = mx + torch.log(torch.exp(a - mx) + torch.exp(b - mx))
    return -lnD + torch.logaddexp(lnB0 - 2 * torch.log(J),
                                  lnC0 - torch.log(J))


class PhysicsRegMLP:
    """Physics-regularized MLP: the positive ML arm.

    A 64-64 MLP maps [1/T, ln J] to the four corrected-equation parameters
    (ln B0, ln C0, Ea_s, Ea_gb); its output is the corrected equation with the
    prefactor ratio *pinned* at the known material value, and training adds a
    soft prior pulling the network parameters toward their literature values.
    The physics is inside the predictor, so the extrapolation functional is
    the pinned corrected equation, not the network's free output: this ML
    predictor extrapolates safely, unlike the black-box MLP.

    Prior ``prior`` is (lnB0, lnC0, Ea_s, Ea_gb) literature values; ``lam``
    scales the soft constraint. seed for torch determinism.
    """

    name = "Physics-regularized MLP (corrected equation, soft prior)"

    def __init__(self, lam=0.1, prior=(3.74e-6, 1.122e-6, 0.8, 1.2),
                 fix_D0r=1.2e4, hidden=64, seed=0, steps=5000, lr=1e-3):
        self.lam = lam
        self.prior = np.asarray(prior, float)
        self.prior[:2] = np.log(self.prior[:2])   # (lnB0, lnC0, Ea_s, Ea_gb)
        self.lnD0r = float(np.log(fix_D0r))
        self.hidden = hidden
        self.seed = seed
        self.steps = steps
        self.lr = lr

    def fit(self, T, J, y, L=None):
        torch = _torch()
        torch.manual_seed(self.seed)
        T = torch.tensor(np.asarray(T, float), dtype=torch.float32)
        J = torch.tensor(np.asarray(J, float), dtype=torch.float32)
        y = torch.tensor(np.asarray(y, float), dtype=torch.float32)
        prior = torch.tensor(self.prior, dtype=torch.float32)
        net = torch.nn.Sequential(
            torch.nn.Linear(2, self.hidden), torch.nn.Tanh(),
            torch.nn.Linear(self.hidden, self.hidden), torch.nn.Tanh(),
            torch.nn.Linear(self.hidden, 4))
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        X = torch.stack([T, J], dim=1)
        lnD0r = self.lnD0r
        for _ in range(self.steps):
            opt.zero_grad()
            p = net(X)
            p5 = torch.cat([p, torch.full((len(p), 1), lnD0r)], dim=1)
            mu = _batched_mu_torch(p5, T, J)
            loss = torch.mean((mu - y) ** 2) + self.lam * torch.mean(
                (p - prior) ** 2)
            loss.backward()
            opt.step()
        self._net = net
        return self

    def predict(self, T, J, L=None):
        torch = _torch()
        T = torch.tensor(np.asarray(T, float), dtype=torch.float32)
        J = torch.tensor(np.asarray(J, float), dtype=torch.float32)
        X = torch.stack([T, J], dim=1)
        with torch.no_grad():
            p = self._net(X)
            p5 = torch.cat([p, torch.full((len(p), 1), self.lnD0r)], dim=1)
            return _batched_mu_torch(p5, T, J).numpy()

    def params(self):
        return {"lam": self.lam, "prior": [float(v) for v in self.prior],
                "hidden": f"{self.hidden}-{self.hidden}"}


# ---------------------------------------------------------------------------
# Small model zoo
# ---------------------------------------------------------------------------
def base_models():
    return [BlackMLE(), TwoRegimeBlack(), AdditiveModel(), MLPBaseline(extended=False)]


def extrapolation_base_models():
    """For the extrapolation paradox: the incumbent, its two-regime fix, the
    fully data-driven corrected equation (free D0r), its literature-constrained
    twin (fixed D0r), and a black-box MLP. The free-vs-constrained pair isolates
    the effect of parameter identifiability on extrapolation accuracy."""
    return [BlackMLE(), TwoRegimeBlack(),
            AdditiveModel(), AdditiveModel(fix_D0r=1.2e4),
            MLPBaseline(extended=False)]


def extended_models():
    return [BlackMLE(), AdditiveBlech(), MLPBaseline(extended=True)]


def extrapolation_extended_models():
    return [BlackMLE(), TwoRegimeBlack(), AdditiveBlech(fix_D0r=1.2e4),
            MLPBaseline(extended=True)]
