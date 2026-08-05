"""
EMBench-Synth: a physically-grounded synthetic electromigration benchmark dataset
====================================================================================

Generates a (T, J, L) -> MTTF dataset for validating equation-discovery (KAN +
symbolic regression) pipelines in electromigration reliability. Every term in
the generative model is either a directly-citable literature result or an
explicitly-flagged modeling choice. The dataset encodes THREE known EM
phenomena in one controlled testbed:

  1. Multi-pathway diffusion activation energy, Ea_eff(T). Parallel diffusion
     pathways (surface / grain-boundary / bulk) each contribute an Arrhenius
     diffusivity D_p(T) = D0_p exp(-Ea_p / kT). The physically-derived
     effective activation energy is the logsumexp-weighted mean

         Ea_eff(T) = Sum_p w_p(T) Ea_p / Sum_p w_p(T),  w_p(T) = D0_p exp(-Ea_p/kT)

     which produces a smooth, monotone "kink" in the Arrhenius plot between the
     low-T dominant pathway and the high-T dominant pathway. This is the
     textbook explanation of curved / two-regime Arrhenius data (Lienig et al.
     2025, Table 2.1; Lloyd NASA NEPP; Tu & Gusak 2019). D0 ratios are chosen
     so the surface->grain-boundary crossover sits inside the literature-tested
     170-425 C temperature envelope (COIN project; SWEAT; AlCu studies).

  2. Additive void nucleation + growth failure time. Rovitto and the
     nucleation-growth literature (Huntington 2007) decompose MTTF as

         MTTF = t_N + t_E = B(T) J^-2 + C(T) J^-1

     where t_N is the void-nucleation-limited term (n = 2) and t_E the
     void-growth (drift) term (n = 1). Both are driven by the same mass
     transport, so B(T) = B0 / D_eff(T) and C(T) = C0 / D_eff(T), with
     D_eff(T) the total effective diffusivity. The local (log-log) current
     exponent n_eff(J) = (2 B J^-2 + C J^-1) / (B J^-2 + C J^-1) is therefore
     fractional, between 1 and 2 -- exactly the fractional exponents reported
     by Huntington. No ad-hoc n(T) drift is imposed; the observed exponent
     emerges from the additive structure.

  3. Blech (critical product) immortality. Below the critical product
     (jL)_c a line does not fail by EM (back-stress balances the electron-wind
     driving force). Above it, the Korhonen-type failure time diverges as
     (jL) -> (jL)_c^+, giving a genuine immortality threshold:

         MTTF_ext = MTTF * boost(jL),   boost(jL) = jL / (jL - (jL)_c)

     Rows below (jL)_c are flagged immortal (MTTF undefined). (jL)_c is set to
     the single-link 90 nm SiCOH literature value 3000 A/cm (Ogawa et al. 2001;
     large-scale statistics from academia.edu/118806387; range 2000-10000 A/cm).

Form of the temperature dependence. Parallel (independent) diffusion pathways
sum additively in diffusivity space, D_eff(T) = Sum_p D0_p exp(-Ea_p/kT), so
the physically-derived MTTF is proportional to 1/D_eff(T) -- NOT to
exp(Ea_eff/kT). With the latter form the local Arrhenius slope develops a
spurious negative "dip" near the crossover; with the former, the slope equals
the logsumexp-weighted mean Ea_eff(T) exactly, giving a clean two-regime kink
between the low-T dominant pathway and the high-T dominant pathway. This is
the textbook explanation of curved / two-regime Arrhenius data (Lienig et al.
2025, Table 2.1; Lloyd NASA NEPP; Tu & Gusak 2019). D0 ratios are chosen so
the surface->grain-boundary crossover sits inside the literature-tested
170-425 C temperature envelope (COIN project; SWEAT; AlCu studies).

Physical constants (all citable):
    Cu surface diffusion Ea        = 0.8 eV   (Lienig et al. 2025, Table 2.1)
    Cu grain-boundary Ea           = 1.2 eV   (Lienig et al. 2025, Table 2.1)
    Cu bulk/lattice Ea             = 2.3 eV   (Lienig et al. 2025, Table 2.1;
                                               NOT reached in this T-range by design)
    D0 ratios                      = modeling choice, calibrated so the crossover
                                    sits inside the tested 170-425 C envelope.
    B(T), C(T) prefactors          = calibrated fit knobs (the "A" of Black's
                                    equation is always process/geometry-specific).

Explicitly NOT in this dataset:
    - Bulk (2.3 eV) diffusion regime: no literature-documented test envelope
      reaches it. Do not claim bulk-diffusion recovery.
    - Bimodal failure statistics (Filippi 2009): single-mode lognormal only.
    - Joule-heating n > 2 regime: excluded as a documented confound.
"""

import json
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
k_eV = 8.617333262e-5  # Boltzmann constant, eV/K

# Diffusion pathways: (label, D0 relative, Ea eV) -- Ea from Lienig 2025 T2.1.
# D0 ratios are a disclosed modeling choice tuned so the crossover temperature
# (where surface and grain-boundary diffusivities are equal) lands inside the
# literature-tested 170-425 C envelope.
PATHWAYS = [
    ("surface", 1.0e0, 0.8),      # dominant low-T Cu diffusion path
    ("grain_boundary", 1.2e4, 1.2),  # dominant high-T path in this range
    ("bulk", 1.0e0, 2.3),          # included for completeness; negligible here
]

T_CROSSOVER_K = 0.4 / np.log(1.2e4 / 1.0e0) / k_eV  # surface == GB equality point

# Black's-equation-style prefactors for nucleation and growth terms. Units:
# hours * (MA/cm^2)^n * (diffusivity units). Calibrated so MTTF ~ 300 h at the
# literature-typical accelerated-test anchor (200 C, 2 MA/cm^2), consistent
# with wafer-level EM test durations ("lifetime test could last more than
# 500 h", SWEAT study).
B0 = 3.74e-6     # t_N = B0 * J^-2 / D_eff(T)
alpha = 0.30     # C0 = alpha * B0  ->  t_E = C0 * J^-1 / D_eff(T)
C0 = alpha * B0

# Blech critical product, A/cm. Central estimate from single-link 90 nm SiCOH.
JL_CRITICAL = 3000.0  # A/cm

# Lognormal multiplicative noise on MTTF (EM failure times are lognormal).
NOISE_SIGMA_LOG = 0.15


# ---------------------------------------------------------------------------
# Physics kernels (imported by the benchmark suite)
# ---------------------------------------------------------------------------
def ea_effective(T):
    """Logsumexp-weighted effective activation energy for parallel pathways.

    Ea_eff(T) = Sum_p D0_p exp(-Ea_p/kT) Ea_p / Sum_p D0_p exp(-Ea_p/kT).
    """
    T = np.asarray(T, dtype=float)
    shp = np.shape(T)
    w = np.empty((len(PATHWAYS),) + shp)
    for i, (_, D0, Ea) in enumerate(PATHWAYS):
        w[i] = D0 * np.exp(-Ea / (k_eV * T))
    eas = np.array([Ea for _, _, Ea in PATHWAYS])
    return np.tensordot(w.T, eas, axes=1) / w.sum(axis=0)


def n_effective(J):
    """Local log-log current-density exponent of the additive failure model.

    n_eff(J) = (2 B J^-2 + C J^-1) / (B J^-2 + C J^-1) = (2 + alpha J)/(1 + alpha J),
    which lies between 1 (growth-dominated, high J) and 2 (nucleation-dominated, low J).
    """
    J = np.asarray(J, dtype=float)
    return (2.0 + alpha * J) / (1.0 + alpha * J)


def effective_diffusivity(T):
    """Total effective diffusivity from parallel diffusion pathways.

    D_eff(T) = Sum_p D0_p exp(-Ea_p / kT). Parallel (independent) mass-transport
    paths contribute additively to the diffusivity; failure rate is proportional
    to D_eff, so MTTF is inversely proportional to it.
    """
    T = np.asarray(T, dtype=float)
    out = np.zeros(np.shape(T))
    for _, D0, Ea in PATHWAYS:
        out = out + D0 * np.exp(-Ea / (k_eV * T))
    return out


def black_core_mttf(T, J):
    """MTTF without length (no Blech).

    MTTF = (B0 J^-2 + C0 J^-1) / D_eff(T).  The Arrhenius slope of this form
    is exactly Ea_eff(T) -- the logsumexp-weighted mean activation energy --
    so local Arrhenius slopes recover the two-regime Ea curve directly.
    """
    T = np.asarray(T, dtype=float)
    J = np.asarray(J, dtype=float)
    ea = ea_effective(T)
    mttf = (B0 * J ** -2.0 + C0 * J ** -1.0) / effective_diffusivity(T)
    return mttf, ea, n_effective(J)


def jl_product(J_MA_cm2, L_um):
    """Critical-product quantity in A/cm. J in MA/cm^2, L in um.

    jL [A/cm] = J [A/cm^2] * L [cm]; J = J_MA_cm2 * 1e6 A/cm^2 (implicitly
    normalized to the line cross-section the current density refers to).
    """
    return np.asarray(J_MA_cm2, dtype=float) * 1e6 * np.asarray(L_um, dtype=float) * 1e-4


def blech_boost(jl, jl_c=JL_CRITICAL):
    """Multiplicative MTTF boost from the Blech back-stress effect.

    boost(jL) = jL / (jL - (jL)_c)  for  jL > (jL)_c   (diverges at threshold)
                (immortal)                          for  jL <= (jL)_c

    The divergence-at-threshold form follows the Korhonen-type stress-evolution
    result that time-to-failure grows without bound as the critical product is
    approached from above; well above threshold the boost -> 1 and ordinary
    Black's-equation behaviour is recovered.
    """
    jl = np.asarray(jl, dtype=float)
    boost = np.full_like(jl, np.nan)
    mortal = jl > jl_c
    boost[mortal] = jl[mortal] / (jl[mortal] - jl_c)
    return boost


# ---------------------------------------------------------------------------
# Sampling grid -- all inside the literature-tested 170-425 C (443-698 K) window.
# ---------------------------------------------------------------------------
T_dense = np.linspace(455.0, 545.0, 40)   # dense around the ~494 K crossover
T_wide = np.linspace(443.0, 698.0, 20)    # coarse coverage of the tested envelope
T_grid = np.unique(np.concatenate([T_dense, T_wide]))

J_grid = np.array([0.5, 1.0, 2.0, 4.0, 8.0])   # MA/cm^2, literature test currents

# Lengths: span from well below to well above the critical length (jL_c/J ~
# 300 um at J=1 MA/cm^2; ~37.5 um at J=8). Values < 35 um at low J are immortal.
L_grid = np.array([5.0, 15.0, 25.0, 35.0, 40.0, 50.0, 65.0, 90.0, 130.0, 200.0, 350.0, 600.0])

SEED = 42
rng = np.random.default_rng(SEED)


def generate(noise_sigma=NOISE_SIGMA_LOG, seed=SEED):
    """Generate the base + extended dataframes with shared noise draws.

    A single noise draw is made per (T, J, L) row. The BASE dataset is the
    (T, J) projection with the noise of a far-from-threshold reference length
    (600 um, boost ~ 1), so base and extended agree exactly on shared (T, J).
    """
    local_rng = np.random.default_rng(seed)

    rows_ext = []
    for T in T_grid:
        for J in J_grid:
            for L in L_grid:
                jl = jl_product(J, L)
                boost = blech_boost(jl)
                immortal = not (jl > JL_CRITICAL)
                mttf_clean, ea, n_eff = black_core_mttf(T, J)
                noise = local_rng.normal(0, noise_sigma) if not immortal else np.nan
                mttf = mttf_clean * boost * np.exp(noise) if not immortal else np.nan
                rows_ext.append({
                    "T_K": round(float(T), 2),
                    "T_C": round(float(T) - 273.15, 2),
                    "J_MA_cm2": J,
                    "L_um": L,
                    "jL_product_A_cm": round(float(jl), 1),
                    "immortal": bool(immortal),
                    "MTTF_hours": float(mttf),
                    "Ea_eff_eV_ground_truth": round(float(ea), 4),
                    "n_eff_ground_truth": round(float(n_eff), 4),
                    "blech_boost_ground_truth": round(float(boost), 4),
                })
    df_ext = pd.DataFrame(rows_ext)

    # Base dataset: (T, J) only, using the reference-length (600 um) noise draw
    # so MTTF values are identical to the extended set at the same (T, J, L=600).
    df_base = df_ext[df_ext.L_um == 600.0].copy().reset_index(drop=True)
    df_base["MTTF_hours"] = df_base.MTTF_hours / df_base.blech_boost_ground_truth  # undo ~1.0 boost
    df_base = df_base[["T_K", "T_C", "J_MA_cm2", "MTTF_hours",
                       "Ea_eff_eV_ground_truth", "n_eff_ground_truth"]]

    return df_base, df_ext


def write_splits(df_base, df_ext, out_dir):
    """Write documented train/test splits to out_dir/splits.

    Splits (all written as row-index CSV files referencing the *full* data
    CSVs, which are row-order-stable and regenerated from the same seed):
      - random 80/20 (seed 42)
      - crossover-band holdout: hold out T in [480, 510] K
      - J-tier holdout:         hold out J = 8 MA/cm^2
      - short-L holdout:        hold out L < 65 um  (extended only)
    """
    split_dir = os.path.join(out_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)
    r = np.random.default_rng(42)
    n = len(df_base)
    perm = r.permutation(n)
    n_test = int(0.2 * n)

    def _write(name, train_idx, test_idx):
        pd.DataFrame({"row_index": train_idx}).to_csv(
            os.path.join(split_dir, f"{name}_train.csv"), index=False)
        pd.DataFrame({"row_index": test_idx}).to_csv(
            os.path.join(split_dir, f"{name}_test.csv"), index=False)

    _write("random", perm[n_test:], perm[:n_test])

    cross_mask = (df_base.T_K >= 480) & (df_base.T_K <= 510)
    _write("crossover_band", df_base.index[~cross_mask], df_base.index[cross_mask])

    j8_mask = df_base.J_MA_cm2 == 8.0
    _write("j_tier", df_base.index[~j8_mask], df_base.index[j8_mask])

    shortL_mask = (df_ext.L_um < 65.0)
    _write("short_L", df_ext.index[~shortL_mask], df_ext.index[shortL_mask])


def build_metadata(generation_log):
    metadata = {
        "dataset": {
            "name": "EMBench-Synth",
            "version": "1.0",
            "purpose": "Controlled testbed for equation-discovery (KAN + symbolic "
                       "regression) in electromigration reliability. Synthetic, NOT "
                       "a substitute for real experimental data.",
            "crossover_T_K": float(T_CROSSOVER_K),
            "crossover_T_C": float(T_CROSSOVER_K - 273.15),
            "noise_sigma_log": float(generation_log["noise_sigma"]),
            "seed": int(generation_log["seed"]),
            "T_K_range": [float(T_grid.min()), float(T_grid.max())],
            "J_MA_cm2_grid": [float(j) for j in J_grid],
            "L_um_grid": [float(l) for l in L_grid],
        },
        "physical_model": {
            "mttf_base": "MTTF = (B0 J^-2 + C0 J^-1) / D_eff(T),  "
                          "D_eff(T) = Sum_p D0_p exp(-Ea_p/kT)",
            "nucleation_growth": "MTTF = t_N + t_E = B(T) J^-2 + C(T) J^-1; "
                                 "local exponent n_eff(J) = (2 + alpha J)/(1 + alpha J) in [1, 2]",
            "ea_effective": "Apparent Ea = Ea_eff(T) = Sum_p w_p Ea_p / Sum_p w_p, "
                            "w_p = D0_p exp(-Ea_p/kT); the local Arrhenius slope "
                            "of MTTF = const / D_eff(T) equals Ea_eff(T) exactly",
            "pathways": [{"name": n, "D0_relative": d0, "Ea_eV": ea}
                         for n, d0, ea in PATHWAYS],
            "blech": "MTTF_ext = MTTF * boost(jL); boost(jL) = jL/(jL - jL_c) for "
                     "jL > jL_c; immortal (MTTF undefined) for jL <= jL_c",
            "jL_critical_A_per_cm": JL_CRITICAL,
            "prefactors": {"B0": B0, "C0": C0, "alpha": alpha,
                           "note": "Calibrated fit knobs (the 'A' of Black's equation), "
                                   "not universal constants."},
        },
        "sources": {
            "Lienig_2025_Table2.1": {
                "citation": "Lienig, J., Scheible, J., et al., Fundamentals of "
                            "Electromigration-Aware Integrated Circuit Design, Springer 2025, Ch. 2, Table 2.1",
                "values_eV": {"surface": 0.8, "grain_boundary": 1.2, "bulk": 2.3},
            },
            "Huntington_2007_nucleation_growth": {
                "citation": "H. B. Huntington, 'Black's law revisited -- Nucleation and "
                            "growth in electromigration failure', Microelectronics Reliability 47 (2007)",
                "note": "MTTF = t_N + t_E, n = 2 nucleation / n = 1 growth; observed "
                        "fractional exponents (e.g. n = 1.4 +/- 0.1) explained by the additive sum.",
            },
            "Rovitto_thesis": {
                "citation": "M. Rovitto, PhD thesis, TU Wien (iue.tuwien.ac.at/phd/rovitto), eq. 1.3",
                "note": "MTTF = t_N + t_E = B j^-2 + C j^-1.",
            },
            "Tu_Gusak_2019": {
                "citation": "K. N. Tu, A. M. Gusak, 'A unified model of mean-time-to-failure for "
                            "electromigration, thermomigration, and stress-migration based on entropy "
                            "production', J. Appl. Phys. 126, 075109 (2019)",
                "note": "Justifies n = 2 for the nucleation-dominated limit via entropy production.",
            },
            "Korhonen_1993": {
                "citation": "M. A. Korhonen et al., J. Appl. Phys. 73, 3790 (1993)",
                "note": "Stress-evolution PDE underlying the Blech back-stress model; "
                        "time-to-failure diverges as (jL) -> (jL)_c from above.",
            },
            "Ogawa_2001_Blech": {
                "citation": "E. T. Ogawa et al., 'Blech Effect in Cu Interconnects and Its "
                            "Applications', Appl. Phys. Lett. 78, 2652 (2001)",
                "note": "Reports (jL)_c ~ 3700 A/cm in Cu/oxide dual damascene.",
            },
            "Blech_critical_product_range": {
                "citation": "arXiv:1712.05562 (general 2000-10000 A/cm range); "
                            "academia.edu/118806387 (90/65/45nm SiCOH large-scale statistics, "
                            "2900-3000 A/cm single-link)",
                "note": "Central estimate 3000 A/cm used here.",
            },
            "Filippi_2009_bimodal": {
                "citation": "R. G. Filippi, P.-C. Wang, R. Brendler, J. R. Lloyd, "
                            "Appl. Phys. Lett. 95, 072111 (2009)",
                "note": "Real EM failure times are often bimodal (early + late populations); "
                        "this dataset uses single-mode lognormal noise only (documented limitation).",
            },
            "tested_envelopes": {
                "citation": "COIN project Cu 170-250 C; AlCu 200-250 C (MDPI); "
                            "SWEAT up to ~425 C before bimodal onset",
                "note": "Sampling is confined to this 170-425 C envelope.",
            },
        },
        "IMPORTANT_CAVEATS": [
            "SYNTHETIC dataset for pipeline validation, not a substitute for real experimental data.",
            "Bulk (2.3 eV) diffusion is NOT reached in the sampled T-range by design "
            "(no tested envelope reaches it) -- do not claim bulk-diffusion recovery.",
            "Single-mode lognormal failure statistics only; real EM is often bimodal "
            "(Filippi 2009).",
            "B0, C0 (the 'A' prefactors) are calibrated fit knobs, not universal "
            "constants -- report SHAPES (Ea_eff(T), n_eff(J), (jL)_c), not absolute "
            "MTTF magnitudes, as meaningful.",
            "D0 pathway ratios are a disclosed modeling choice calibrating the "
            "crossover temperature into the tested envelope.",
            "The Blech boost form jL/(jL - jL_c) is a Korhonen-type first-order "
            "approximation; the exact transition shape is an open modeling question. "
            "The threshold VALUE (jL)_c is literature-grounded.",
            "Ground-truth columns exist ONLY for your post-hoc validation; never expose "
            "them to the discovery algorithm (use the *blind* CSVs).",
        ],
    }
    return metadata


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)

    df_base, df_ext = generate()
    write_splits(df_base, df_ext, out_dir)

    df_base.to_csv(os.path.join(out_dir, "em_synthetic_blind.csv"), index=False)
    full = df_base[["T_K", "T_C", "J_MA_cm2", "MTTF_hours",
                    "Ea_eff_eV_ground_truth", "n_eff_ground_truth"]]
    full.to_csv(os.path.join(out_dir, "em_synthetic_full_with_groundtruth.csv"), index=False)

    ext_cols = ["T_K", "T_C", "J_MA_cm2", "L_um", "MTTF_hours"]
    df_ext[ext_cols].to_csv(os.path.join(out_dir, "em_synthetic_extended_blind.csv"), index=False)
    df_ext.to_csv(os.path.join(out_dir, "em_synthetic_extended_full_with_groundtruth.csv"), index=False)

    metadata = build_metadata({"noise_sigma": NOISE_SIGMA_LOG, "seed": SEED})
    with open(os.path.join(out_dir, "metadata_and_sources.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"crossover T = {T_CROSSOVER_K:.1f} K ({T_CROSSOVER_K - 273.15:.1f} C)")
    print(f"base:    {len(df_base)} rows  (T x J) = {len(T_grid)} x {len(J_grid)}")
    print(f"extended:{len(df_ext)} rows  (T x J x L) = {len(T_grid)} x {len(J_grid)} x {len(L_grid)}")
    print(f"  immortal rows: {(df_ext.immortal).sum()}")
    print(f"T range: {T_grid.min():.0f}-{T_grid.max():.0f} K "
          f"({T_grid.min() - 273.15:.0f}-{T_grid.max() - 273.15:.0f} C)")
    m = df_ext[~df_ext.MTTF_hours.isna()]
    print(f"mortal MTTF range: {m.MTTF_hours.min():.3g} - {m.MTTF_hours.max():.3g} h")
    print(f"Ea_eff range: {df_base.Ea_eff_eV_ground_truth.min():.3f} - "
          f"{df_base.Ea_eff_eV_ground_truth.max():.3f} eV")
    print(f"n_eff range: {df_base.n_eff_ground_truth.min():.3f} - "
          f"{df_base.n_eff_ground_truth.max():.3f}")
    print(f"boost range: {df_ext.blech_boost_ground_truth.dropna().min():.3f} - "
          f"{df_ext.blech_boost_ground_truth.max():.3f}")
    print("wrote data/ (4 CSVs, metadata, splits)")


if __name__ == "__main__":
    main()
