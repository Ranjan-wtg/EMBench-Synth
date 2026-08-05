# EMBench-Synth Dataset Card

**EMBench-Synth** — a deterministic synthetic benchmark for electromigration (EM)
reliability equation discovery and lifetime prediction.

## Motivation
The industrial job of Black's equation is *extrapolation*: fit accelerated
wafer-level tests, predict use conditions. Real wafer data cannot be used to
test recovery or extrapolation because the ground truth is unknown. EMBench-Synth
provides a controlled, exactly-solvable proxy: the generating process is known,
so recovery, identifiability, and extrapolation error are measurable.

## Task
Predict ln(MTTF) from physical inputs (T in K, J in MA/cm^2, optionally L in um).
- **Base** (T, J): the corrected equation without length dependence.
- **Extended** (T, J, L): includes the Blech critical-product term; rows below
  (jL)_c = 3000 A/cm are immortal (no finite MTTF).

## Generating process
MTTF = (B0 J^-2 + C0 J^-1) / D_eff(T) · jL/(jL - (jL)_c), with:
- two-pathway activation energy D_eff(T) = Σ_p D0_p exp(-Ea_p/kT),
  crossover at T_c = 494.2 K;
- fractional current exponent n_eff(J) ∈ (1, 2);
- Blech divergence at (jL)_c = 3000 A/cm.
Values: B0 = 3.74e-6, C0 = 1.122e-6, Ea_s = 0.8 eV, Ea_gb = 1.2 eV,
D0_gb/D0_s = 1.2e4. See `generate_dataset.py`.

## Data
- `data/em_synthetic_full_with_groundtruth.csv` — base, with ground truth
- `data/em_synthetic_blind.csv` — base, only (T, J, MTTF)
- `data/em_synthetic_extended_full_with_groundtruth.csv` — extended + truth
- `data/em_synthetic_extended_blind.csv` — extended, blind
- `data/splits/` — documented train/test splits
  (random, crossover-band, J-tier, short-L holdouts)

## Bias / limitations
Synthetic by design. The physics is encoded by construction; the benchmark is a
measurement instrument, not a discovery claim. Single-mode log-normal failure.
The bulk pathway (Ea_b = 2.3 eV) is outside the sampled window. Deterministic
seed 42; noise sigma = 0.15 by default.

## Reproducibility
```
python generate_dataset.py
python benchmarks/run_all.py
python benchmarks/ml_predictors.py   # Arm 1: ML predictors as extrapolators
python benchmarks/gnn.py             # Arm 3: GNN-for-circuits (PDN analog)
python benchmarks/leaderboard.py     # Arm 2: model zoo + leaderboard
```

## Scoring (see `benchmarks/leaderboard.py`)
- `in_window_fit_rmse_ln` — fit quality on the accelerated window (T >= 523 K)
- `use_condition_fold` — median |predicted/true| at 100 C, low J
- `safe_to_2x_T_K` — extrapolation temperature at which fold error reaches 2x
  (transfer-law definition; lower is safer)
- `use_condition_median_logratio` — sign of the error (positive = overprediction)

## Model zoo API
```python
from benchmarks import models as M
m = M.AdditiveModel(fix_D0r=1.2e4)   # or any model in benchmarks/models.py
m.fit(T, J, y)                        # accelerated window
m.predict(Tuse, Juse)                 # use conditions
# positive ML arm:
m = M.PhysicsRegMLP(lam=0.1)          # corrected-eq. output + soft prior
```
Any third-party model implementing fit/predict over (T, J[, L]) can be added to
`benchmarks/leaderboard.py` MODEL_ZOO and scored automatically.
