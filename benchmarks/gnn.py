"""Arm 3: GNN-for-circuits on EMBench-Synth (power-delivery network analog).

The industrial setting for EM reliability is a power-delivery network (PDN):
a graph of wire segments through which current distributes according to
conductances (Kirchhoff). We generate such networks, solve the DC current
distribution exactly, and give every segment a corrected-equation MTTF
(including the Blech length term). The ML task is the one EDA actually does:
given the netlist graph and per-segment geometry/temperature, predict per-
segment ln(MTTF).

Two regimes mirror the paper's protocol:
  - accelerated networks: T in [523, 698] K, high injected current
  - use-condition networks: T = 373 K, low injected current
A message-passing GNN (and a topology-blind MLP baseline) is trained only on
accelerated networks and evaluated at use conditions. We report in-window
RMSE and the use-condition fold error -- i.e. does graph structure let a
learned reliability predictor *extrapolate* safely?

Writes results/gnn_pdn.json
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import generate_dataset as gen

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

SEED = 42
N_JUNCTIONS = 12
RADIUS = 0.42
T_LO, T_HI = 523.0, 698.0
T_USE = 373.0
INJECT_HI, INJECT_LO = 1.0, 0.25
NOISE = 0.15


def _generate_network(rng, use_condition=False, noise_sigma=NOISE):
    """One PDN: junction graph, Kirchhoff solve, per-segment (feat, ln MTTF).

    Returns
      feat   : (N, 4) per mortal segment: [1/T, ln J, ln L, ln A]
      y      : (N,)   ln MTTF (Blech included)
      edges  : (2, E) line-graph adjacency between segments sharing a junction
    """
    n = N_JUNCTIONS
    pos = rng.uniform(0.0, 1.0, (n, 2))
    d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    adj = [set() for _ in range(n)]
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if d[u, v] < RADIUS:
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
    # guarantee connectivity with a spanning chain
    for u in range(n - 1):
        if not any(v in adj[u] for v in range(u + 1, n)):
            adj[u].add(u + 1)
            adj[u + 1].add(u)
            edges.append((u, u + 1))
    E = len(edges)
    L = rng.uniform(20.0, 400.0, E)
    A = rng.lognormal(mean=np.log(0.1), sigma=0.5, size=E)
    g = A / L

    Gmat = np.zeros((n, n))
    for (u, v), ge in zip(edges, g):
        Gmat[u, u] += ge
        Gmat[v, v] += ge
        Gmat[u, v] -= ge
        Gmat[v, u] -= ge
    Gred = Gmat[:-1, :-1]
    I = np.zeros(n - 1)
    I[0] = INJECT_LO if use_condition else INJECT_HI
    V = np.linalg.solve(Gred + 1e-9 * np.eye(n - 1), I)
    Vfull = np.concatenate([V, [0.0]])

    if use_condition:
        T = np.full(E, T_USE)
    else:
        T = rng.uniform(T_LO, T_HI, E)

    feats, ys, seg_nodes = [], [], []
    for k, (u, v) in enumerate(edges):
        Ie = g[k] * (Vfull[u] - Vfull[v])
        Jk = np.abs(Ie) / A[k]
        jlk = gen.jl_product(Jk, L[k])
        if jlk <= gen.JL_CRITICAL:
            continue
        core, _, _ = gen.black_core_mttf(T[k], Jk)
        m = core * gen.blech_boost(jlk)
        if noise_sigma > 0:
            m *= np.exp(rng.normal(0.0, noise_sigma))
        feats.append([1.0 / T[k], np.log(Jk), np.log(L[k]), np.log(A[k])])
        ys.append(np.log(m))
        seg_nodes.append(k)

    feat = np.asarray(feats, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    # line graph: two segments adjacent if they share a junction
    seg_to_junc = {k: (edges[k][0], edges[k][1]) for k in seg_nodes}
    from collections import defaultdict
    junc_segs = defaultdict(list)
    for si in seg_nodes:
        for j in edges[si]:
            junc_segs[j].append(si)
    edge_pairs = []
    for j, sers in junc_segs.items():
        for a in range(len(sers)):
            for b in range(a + 1, len(sers)):
                edge_pairs.append((sers[a], sers[b]))
    node_of_seg = {s: i for i, s in enumerate(seg_nodes)}
    if edge_pairs:
        src = [node_of_seg[a] for a, b in edge_pairs]
        dst = [node_of_seg[b] for a, b in edge_pairs]
        eidx = np.stack([np.asarray(src + dst, dtype=np.int64),
                         np.asarray(dst + src, dtype=np.int64)])
    else:
        eidx = np.zeros((2, 0), dtype=np.int64)
    return feat, y, eidx


class GCN(nn.Module):
    """GraphSAGE-style: own-feature branch || neighbor-aggregate branch.

    The node's own features alone determine MTTF, so the network keeps an
    un-averaged own-feature branch and concatenates a neighbor aggregate the
    optimizer is free to (down-)weight. This guarantees the GNN is at least as
    expressive as the topology-blind MLP in-window, isolating the effect of
    graph context on *extrapolation* rather than on raw fit.
    """

    def __init__(self, d_in, hid=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_in, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU())
        self.agg = nn.Linear(hid, hid)
        self.head = nn.Sequential(nn.Linear(2 * hid, hid), nn.ReLU(),
                                  nn.Linear(hid, 1))

    def forward(self, x, eidx):
        N = x.shape[0]
        adj = torch.zeros(N, N, device=x.device)
        if eidx.shape[1] > 0:
            adj[eidx[0], eidx[1]] = 1.0
        deg = adj.sum(1).clamp(min=1.0)
        An = adj / deg[:, None]
        own = torch.relu(self.mlp(x))
        h_nei = An @ own                          # neighbor aggregate (no self)
        h = torch.cat([own, self.agg(h_nei)], dim=1)
        return self.head(h).squeeze(-1)


def _train_model(model, graphs, steps=600, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    allf = torch.cat([torch.as_tensor(g[0]) for g in graphs])
    mu, sd = allf.mean(0), allf.std(0).clamp(min=1e-3)
    for it in range(steps):
        opt.zero_grad()
        loss = 0.0
        for feat, y, eidx in graphs:
            xn = (torch.as_tensor(feat) - mu) / sd
            pred = model(xn, torch.as_tensor(eidx))
            loss = loss + ((pred - torch.as_tensor(y)) ** 2).mean()
        loss = loss / len(graphs)
        loss.backward()
        opt.step()
    return mu, sd


def _evaluate(model, mu, sd, graphs):
    preds, trues = [], []
    for feat, y, eidx in graphs:
        xn = (torch.as_tensor(feat) - mu) / sd
        preds.append(model(xn, torch.as_tensor(eidx)).detach().numpy())
        trues.append(np.asarray(y))
    pred = np.concatenate(preds)
    true = np.concatenate(trues)
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    fold = float(np.exp(np.median(np.abs(pred - true))))
    return rmse, fold


def _physics_oracle(graphs):
    """Pinned corrected equation evaluated at the segment's true (T, J, L).

    The oracle sees the same inputs as the GNN/MLP (T, J, L are node features)
    but carries the correct structure with one pinned material constant. It is
    the in-scope 'what a structure-correct predictor would achieve' reference.
    """
    from benchmarks import models as M
    preds, trues = [], []
    for feat, y, eidx in graphs:
        T = 1.0 / feat[:, 0]
        J = np.exp(feat[:, 1])
        L = np.exp(feat[:, 2])
        m = M.AdditiveBlech(fix_D0r=1.2e4)
        m.fit(T, J, y, L)
        preds.append(m.predict(T, J, L))
        trues.append(y)
    pred = np.concatenate(preds)
    true = np.concatenate(trues)
    return float(np.sqrt(np.mean((pred - true) ** 2))), \
        float(np.exp(np.median(np.abs(pred - true))))


class MLPBaselineGNN(nn.Module):
    """Topology-blind MLP on the same node features (no message passing)."""

    def __init__(self, d_in, hid=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, 1))

    def forward(self, x, eidx=None):
        return self.net(x).squeeze(-1)


def main():
    SEEDS = [42, 7, 123, 2024, 999]
    n_train, n_test = 80, 50
    g_in, g_use, m_in, m_use = [], [], [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        train = []
        while len(train) < n_train:
            g = _generate_network(rng, use_condition=False)
            if g[0].shape[0] >= 4 and g[2].shape[1] > 0:
                train.append(g)
        test = []
        while len(test) < n_test:
            g = _generate_network(rng, use_condition=True)
            if g[0].shape[0] >= 4 and g[2].shape[1] > 0:
                test.append(g)

        d_in = train[0][0].shape[1]
        gnn_ = GCN(d_in)
        mu, sd = _train_model(gnn_, train)
        a, b = _evaluate(gnn_, mu, sd, train)
        _, c = _evaluate(gnn_, mu, sd, test)
        mlp = MLPBaselineGNN(d_in)
        mlp_mu, mlp_sd = _train_model(mlp, train)
        d, _ = _evaluate(mlp, mlp_mu, mlp_sd, train)
        _, e = _evaluate(mlp, mlp_mu, mlp_sd, test)
        g_in.append(a); g_use.append(c); m_in.append(d); m_use.append(e)
        print(f"seed {seed}: GNN in-window {a:.3f}  use {c:.2f}x | "
              f"MLP in-window {d:.3f}  use {e:.2f}x")

    or_rmse_tr, or_fold_tr = _physics_oracle(train)
    or_rmse_te, or_fold_te = _physics_oracle(test)

    payload = {
        "gnn_pdn": {
            "task": "per-segment ln(MTTF) on power-delivery networks; "
                    "train on accelerated (T 523-698 K), test at use (373 K)",
            "seeds": SEEDS, "n_train_networks": n_train, "n_test_networks": n_test,
            "GNN": {"in_window_rmse": round(float(np.median(g_in)), 4),
                    "in_window_rmse_seedwise": [round(x, 3) for x in g_in],
                    "use_condition_fold": round(float(np.median(g_use)), 3),
                    "use_condition_fold_seedwise": [round(x, 3) for x in g_use]},
            "MLP (topology-blind)": {"in_window_rmse": round(float(np.median(m_in)), 4),
                                     "in_window_rmse_seedwise": [round(x, 3) for x in m_in],
                                     "use_condition_fold": round(float(np.median(m_use)), 3),
                                     "use_condition_fold_seedwise": [round(x, 3) for x in m_use]},
            "physics oracle (pinned, true T/J/L)": {
                "in_window_rmse": round(or_rmse_tr, 4),
                "use_condition_fold": round(or_fold_te, 3)},
            "note": "Medians over seeds. MLP sees identical node features but "
                    "no adjacency. fold = exp(median|err|). immortal segments "
                    "excluded (no finite MTTF).",
        },
    }
    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, "gnn_pdn.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {path}")

    from benchmarks import figures as fig
    fig.fig_gnn(payload["gnn_pdn"], os.path.join(ROOT, "figures",
                                                 "fig_gnn.png"))
    print("wrote figures/fig_gnn.png")


if __name__ == "__main__":
    main()
