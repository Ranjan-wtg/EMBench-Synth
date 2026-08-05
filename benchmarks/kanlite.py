"""kanlite: a compact single-layer spline KAN for interpretable decomposition.

ln(MTTF) = f_1(x_1) + f_2(x_2) + c,  with each edge f_i a B-spline.

This is a legitimate (depth-1) Kolmogorov-Arnold network: learnable univariate
functions on edges, additive composition at the output node. Our target
ln MTTF = Ea_eff(T)/(kT) + ln(B0 J^-2 + C0 J^-1) is exactly additive in the
feature space [T, J], so the two recovered edges should split cleanly into the
activation-energy term and the current-density term. Supports a "physics
initialization" that pre-seeds an edge from a supplied function (e.g. the
Arrhenius prior Ea/kT or the -2 ln J power law), and a symbolic extraction step
that fits each edge from a physics-motivated library.

Features are normalized to [-1, 1] per variable; a public transform maps back
to physical units for symbolic reporting.
"""
import numpy as np
import torch
import torch.nn as nn


def cox_deboor(t, k, knots):
    """Evaluate order-k B-spline basis functions at t (1-D float tensor).

    Standard Cox-de Boor recursion on a knot vector of length m produces
    m-k-1 final basis functions; the intermediate levels start from the m-1
    order-0 step functions.
    """
    t = t.reshape(-1)
    m = len(knots)
    N = torch.zeros(t.shape[0], m - 1)
    for j in range(m - 1):
        N[:, j] = ((t >= knots[j]) & (t < knots[j + 1])).float()
    for p in range(1, k + 1):
        for j in range(m - 1 - p):
            d1 = knots[j + p] - knots[j]
            d2 = knots[j + p + 1] - knots[j + 1]
            left = (t - knots[j]) / d1 if d1 > 0 else torch.zeros_like(t)
            right = (knots[j + p + 1] - t) / d2 if d2 > 0 else torch.zeros_like(t)
            N[:, j] = left * N[:, j] + right * N[:, j + 1]
        N[:, m - 1 - p:] = 0
    return N[:, :m - 1 - k]


def make_knots(xmin, xmax, g, k):
    dx = (xmax - xmin) / g
    return np.concatenate([np.linspace(xmin - k * dx, xmin - dx, k),
                           np.linspace(xmin, xmax, g + 1),
                           np.linspace(xmax + dx, xmax + k * dx, k)]).astype(float)


class SplineEdge(nn.Module):
    def __init__(self, grid=8, k=3, grid_range=(-1.0, 1.0)):
        super().__init__()
        self.k = k
        self.grid_range = grid_range
        self.knots = torch.tensor(make_knots(*grid_range, grid, k), dtype=torch.float32)
        self.n_basis = len(self.knots) - k - 1
        self.w = nn.Parameter(torch.zeros(self.n_basis))

    def forward(self, x):
        B = cox_deboor(x, self.k, self.knots)
        return B @ self.w

    def set_from_func(self, f, n=400):
        """Least-squares set the spline coefficients to approximate f on [-1,1]."""
        x = torch.linspace(-1.0, 1.0, n)
        B = cox_deboor(x, self.k, self.knots)
        y = torch.tensor(f(x.numpy()), dtype=torch.float32)
        self.w.data = torch.linalg.lstsq(B, y.unsqueeze(1), driver="gelsd").solution[:, 0]


class KANLite(nn.Module):
    """Single-layer KAN: y = bias + sum_i f_i(x_i)."""

    def __init__(self, d_in=2, grid=8, k=3):
        super().__init__()
        self.edges = nn.ModuleList([SplineEdge(grid, k) for _ in range(d_in)])
        self.bias = nn.Parameter(torch.zeros(1))
        self.feat_min = None
        self.feat_scale = None

    def normalize(self, X):
        X = torch.as_tensor(X, dtype=torch.float32)
        if self.feat_min is None:
            self.feat_min = X.min(dim=0).values
            self.feat_scale = X.max(dim=0).values - X.min(dim=0).values
            self.feat_scale[self.feat_scale == 0] = 1.0
        return (X - self.feat_min) / self.feat_scale * 2.0 - 1.0

    def forward(self, X):
        Xn = self.normalize(X)
        return sum(e(Xn[:, i]) for i, e in enumerate(self.edges)) + self.bias

    def physics_init(self, funcs):
        """Pre-seed edge i from funcs[i] (a callable on the normalized var)."""
        for i, f in enumerate(funcs):
            self.edges[i].set_from_func(f)

    def fit(self, X, y, steps=4000, lr=2e-2, verbose=False):
        self.fit_steps = steps
        self.fit_lr = lr
        Xt = torch.as_tensor(X, dtype=torch.float32)
        yt = torch.as_tensor(y, dtype=torch.float32)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        best = (float("inf"), None)
        for i in range(steps):
            opt.zero_grad()
            loss = nn.functional.mse_loss(self(Xt), yt)
            loss.backward()
            opt.step()
            sched.step()
            if i % 250 == 0 and verbose:
                print(f"  step {i}: loss {loss.item():.4f}")
        self.fit_loss = float(loss.item())
        return self

    # -- symbolic extraction -------------------------------------------------
    def edge_values(self, Xref):
        Xn = self.normalize(Xref)
        return [e(Xn[:, i]).detach().numpy() for i, e in enumerate(self.edges)]

    @staticmethod
    def _fit_candidates(xn, y, library):
        """Fit each candidate family c*f(scale*(x+shift)) + offset; return best."""
        best = (-np.inf, None, None)
        for fam in library:
            try:
                if fam == "lin":
                    m, b = np.polyfit(xn, y, 1)
                    pred = m * xn + b
                elif fam == "quad":
                    p = np.polyfit(xn, y, 2); pred = np.polyval(p, xn)
                elif fam == "cubic":
                    p = np.polyfit(xn, y, 3); pred = np.polyval(p, xn)
                elif fam == "logistic":
                    from scipy.optimize import curve_fit
                    def g(x, a, b, x0):
                        return a + b / (1 + np.exp((x - x0)))
                    popt, _ = curve_fit(g, xn, y, p0=[y.min(), y.max() - y.min(), 0.0], maxfev=5000)
                    pred = g(xn, *popt)
                elif fam == "softplus":
                    from scipy.optimize import curve_fit
                    def g(x, a, b, s):
                        return a + b * np.log1p(np.exp(s * x))
                    popt, _ = curve_fit(g, xn, y, p0=[0.0, 1.0, 1.0], maxfev=5000)
                    pred = g(xn, *popt)
                elif fam == "exp_shift":
                    from scipy.optimize import curve_fit
                    def g(x, a, b, s):
                        return a + b * np.exp(s * x)
                    popt, _ = curve_fit(g, xn, y, p0=[0.0, 1.0, 1.0], maxfev=5000)
                    pred = g(xn, *popt)
                else:
                    continue
                r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
                if r2 > best[0]:
                    best = (r2, fam, pred)
            except Exception:
                continue
        return best

    def symbolic(self, xref_grid):
        """Return per-variable best symbolic families (R2, fam, predictions)."""
        xn = self.normalize(xref_grid)
        out = []
        for i, e in enumerate(self.edges):
            y = e(xn[:, i]).detach().numpy()
            out.append(self._fit_candidates(xn[:, i].numpy(), y, LIBRARY))
        return out


LIBRARY = ["lin", "quad", "cubic", "logistic", "softplus", "exp_shift"]
