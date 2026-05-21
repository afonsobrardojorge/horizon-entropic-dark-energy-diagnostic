"""HEDD prototype analysis.

Horizon Entropic Dark Energy Diagnostic (HEDD)

This script reads cosmic chronometer H(z) data, reconstructs H(z) and H'(z)
with a deliberately simple weighted-polynomial bootstrap, and computes:

    S_H / S_H0 = (H0 / H)^2
    Delta_H = d ln S_H / d ln a - 3 Omega_m - 4 Omega_r
    w_geom = -1 + Delta_H / (3 Omega_geom)

The goal is not precision cosmology. It is a reproducible first diagnostic
that can later be replaced by a Gaussian Process or a full likelihood with
covariances and BAO/SN data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import math
import numpy as np
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent if MODULE_DIR.name == "src" else MODULE_DIR
DATA_PATH = PROJECT_ROOT / "data" / "hz_cosmic_chronometers.csv"
FSIGMA8_PATH = PROJECT_ROOT / "data" / "fsigma8_compilation.csv"
OUT_DIR = PROJECT_ROOT / "outputs"


@dataclass(frozen=True)
class Cosmology:
    H0: float = 67.36
    H0_sigma: float = 0.54
    Omega_m0: float = 0.3153
    Omega_r0: float = 9.22e-5
    sigma8_0: float = 0.811

    @property
    def Omega_geom0(self) -> float:
        return 1.0 - self.Omega_m0 - self.Omega_r0


def read_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.sort_values("z").reset_index(drop=True)


def read_growth_data(path: Path = FSIGMA8_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.sort_values("z").reset_index(drop=True)


def weighted_polyfit(
    z: np.ndarray,
    H: np.ndarray,
    sigma_H: np.ndarray,
    H0: float,
    degree: int,
    h0_prior_sigma: float | None = None,
) -> np.poly1d:
    """Fit E(z)=H(z)/H0 with inverse-variance weights."""

    if h0_prior_sigma is not None:
        z = np.concatenate(([0.0], z))
        H = np.concatenate(([H0], H))
        sigma_H = np.concatenate(([h0_prior_sigma], sigma_H))

    E = H / H0
    sigma_E = sigma_H / H0
    weights = 1.0 / np.maximum(sigma_E, 1e-12)
    coeff = np.polyfit(z, E, deg=degree, w=weights)
    return np.poly1d(coeff)


def lcdm_E(z: np.ndarray, cosmo: Cosmology) -> np.ndarray:
    return np.sqrt(
        cosmo.Omega_m0 * (1.0 + z) ** 3
        + cosmo.Omega_r0 * (1.0 + z) ** 4
        + cosmo.Omega_geom0
    )


def dark_energy_factor(
    z: np.ndarray,
    w0: float = -1.0,
    wa: float = 0.0,
) -> np.ndarray:
    """CPL dark-energy density factor relative to z=0."""

    zp1 = 1.0 + z
    return zp1 ** (3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * z / zp1)


def cpl_w(z: np.ndarray, w0: float, wa: float) -> np.ndarray:
    return w0 + wa * z / (1.0 + z)


def cpl_E_and_derivative(
    z: np.ndarray,
    cosmo: Cosmology,
    w0: float = -1.0,
    wa: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zp1 = 1.0 + z
    de_factor = dark_energy_factor(z, w0=w0, wa=wa)
    E2 = (
        cosmo.Omega_m0 * zp1**3
        + cosmo.Omega_r0 * zp1**4
        + cosmo.Omega_geom0 * de_factor
    )
    dln_de_dz = 3.0 * (1.0 + w0 + wa) / zp1 - 3.0 * wa / zp1**2
    dE2_dz = (
        3.0 * cosmo.Omega_m0 * zp1**2
        + 4.0 * cosmo.Omega_r0 * zp1**3
        + cosmo.Omega_geom0 * de_factor * dln_de_dz
    )
    E = np.sqrt(np.maximum(E2, 1e-12))
    dE_dz = 0.5 * dE2_dz / E
    return E, dE_dz, cpl_w(z, w0=w0, wa=wa)


def diagnostic_from_E(
    z: np.ndarray,
    E: np.ndarray,
    dE_dz: np.ndarray,
    cosmo: Cosmology,
) -> pd.DataFrame:
    E2 = np.maximum(E**2, 1e-12)
    Omega_m = cosmo.Omega_m0 * (1.0 + z) ** 3 / E2
    Omega_r = cosmo.Omega_r0 * (1.0 + z) ** 4 / E2
    Omega_geom = 1.0 - Omega_m - Omega_r
    dlnS_dlna = 2.0 * (1.0 + z) * dE_dz / np.maximum(E, 1e-12)
    Delta_H = dlnS_dlna - 3.0 * Omega_m - 4.0 * Omega_r

    with np.errstate(divide="ignore", invalid="ignore"):
        w_geom = -1.0 + Delta_H / (3.0 * Omega_geom)
    w_geom = np.where(Omega_geom > 0.03, w_geom, np.nan)

    return pd.DataFrame(
        {
            "z": z,
            "H": cosmo.H0 * E,
            "dH_dz": cosmo.H0 * dE_dz,
            "S_rel": 1.0 / E2,
            "dlnS_dlna": dlnS_dlna,
            "Omega_m": Omega_m,
            "Omega_r": Omega_r,
            "Omega_geom": Omega_geom,
            "Delta_H": Delta_H,
            "w_geom": w_geom,
        }
    )


def bootstrap_reconstruction(
    df: pd.DataFrame,
    cosmo: Cosmology,
    z_grid: np.ndarray,
    degree: int = 3,
    nboot: int = 1000,
    seed: int = 7,
    h0_prior_sigma: float | None = None,
    label: str = "unanchored",
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.poly1d]:
    rng = np.random.default_rng(seed)
    z = df["z"].to_numpy(float)
    H = df["H_km_s_Mpc"].to_numpy(float)
    sigma_H = df["sigma_H_km_s_Mpc"].to_numpy(float)

    central_poly = weighted_polyfit(
        z, H, sigma_H, cosmo.H0, degree, h0_prior_sigma=h0_prior_sigma
    )
    keys = [
        "H",
        "dH_dz",
        "S_rel",
        "dlnS_dlna",
        "Omega_m",
        "Omega_geom",
        "Delta_H",
        "w_geom",
    ]
    samples = {key: [] for key in keys}

    for _ in range(nboot):
        H_sample = rng.normal(H, sigma_H)
        poly = weighted_polyfit(
            z,
            H_sample,
            sigma_H,
            cosmo.H0,
            degree,
            h0_prior_sigma=h0_prior_sigma,
        )
        E = poly(z_grid)
        dE = np.polyder(poly)(z_grid)
        diag = diagnostic_from_E(z_grid, E, dE, cosmo)
        for key in keys:
            samples[key].append(diag[key].to_numpy(float))

    bands = {}
    for key, arrs in samples.items():
        arr = np.asarray(arrs, dtype=float)
        bands[f"{key}_p16"] = np.nanpercentile(arr, 16, axis=0)
        bands[f"{key}_p50"] = np.nanpercentile(arr, 50, axis=0)
        bands[f"{key}_p84"] = np.nanpercentile(arr, 84, axis=0)

    central = diagnostic_from_E(
        z_grid, central_poly(z_grid), np.polyder(central_poly)(z_grid), cosmo
    )
    for name, values in bands.items():
        central[name] = values
    central["reconstruction"] = label
    central["H0_anchor_sigma"] = np.nan if h0_prior_sigma is None else h0_prior_sigma

    return central, samples, central_poly


def add_bands_from_samples(
    central: pd.DataFrame,
    samples: dict[str, list[np.ndarray]],
) -> pd.DataFrame:
    for key, arrs in samples.items():
        arr = np.asarray(arrs, dtype=float)
        central[f"{key}_p16"] = np.nanpercentile(arr, 16, axis=0)
        central[f"{key}_p50"] = np.nanpercentile(arr, 50, axis=0)
        central[f"{key}_p84"] = np.nanpercentile(arr, 84, axis=0)
    return central


def with_optional_h0_anchor(
    z: np.ndarray,
    H: np.ndarray,
    sigma_H: np.ndarray,
    cosmo: Cosmology,
    h0_prior_sigma: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if h0_prior_sigma is None:
        return z.copy(), H.copy(), sigma_H.copy()
    return (
        np.concatenate(([0.0], z)),
        np.concatenate(([cosmo.H0], H)),
        np.concatenate(([h0_prior_sigma], sigma_H)),
    )


def natural_cubic_second_derivatives(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(x)
    second = np.zeros(n)
    if n <= 2:
        return second

    u = np.zeros(n - 1)
    for i in range(1, n - 1):
        sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])
        p = sig * second[i - 1] + 2.0
        second[i] = (sig - 1.0) / p
        ddydx = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (
            y[i] - y[i - 1]
        ) / (x[i] - x[i - 1])
        u[i] = (6.0 * ddydx / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p

    for k in range(n - 2, -1, -1):
        second[k] = second[k] * second[k + 1] + u[k]
    return second


def natural_cubic_eval(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique_x, unique_idx = np.unique(x, return_index=True)
    x = unique_x
    y = y[unique_idx]
    second = natural_cubic_second_derivatives(x, y)

    y_grid = np.empty_like(x_grid, dtype=float)
    dy_grid = np.empty_like(x_grid, dtype=float)
    for j, xg in enumerate(x_grid):
        i = np.searchsorted(x, xg) - 1
        i = int(np.clip(i, 0, len(x) - 2))
        h = x[i + 1] - x[i]
        if h <= 0:
            y_grid[j] = np.nan
            dy_grid[j] = np.nan
            continue
        A = (x[i + 1] - xg) / h
        B = (xg - x[i]) / h
        y_grid[j] = (
            A * y[i]
            + B * y[i + 1]
            + ((A**3 - A) * second[i] + (B**3 - B) * second[i + 1]) * h**2 / 6.0
        )
        dy_grid[j] = (
            (y[i + 1] - y[i]) / h
            - (3.0 * A**2 - 1.0) * h * second[i] / 6.0
            + (3.0 * B**2 - 1.0) * h * second[i + 1] / 6.0
        )
    return y_grid, dy_grid


def spline_reconstruction(
    df: pd.DataFrame,
    cosmo: Cosmology,
    z_grid: np.ndarray,
    nboot: int = 1000,
    seed: int = 11,
    h0_prior_sigma: float | None = None,
    label: str = "spline_planck_anchor",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = df["z"].to_numpy(float)
    H = df["H_km_s_Mpc"].to_numpy(float)
    sigma_H = df["sigma_H_km_s_Mpc"].to_numpy(float)
    z_fit, H_fit, sigma_fit = with_optional_h0_anchor(z, H, sigma_H, cosmo, h0_prior_sigma)
    E_fit = H_fit / cosmo.H0

    E, dE = natural_cubic_eval(z_fit, E_fit, z_grid)
    central = diagnostic_from_E(z_grid, E, dE, cosmo)
    keys = [
        "H",
        "dH_dz",
        "S_rel",
        "dlnS_dlna",
        "Omega_m",
        "Omega_geom",
        "Delta_H",
        "w_geom",
    ]
    samples = {key: [] for key in keys}
    for _ in range(nboot):
        H_sample = rng.normal(H_fit, sigma_fit)
        if h0_prior_sigma is not None:
            H_sample[0] = rng.normal(cosmo.H0, h0_prior_sigma)
        E_sample = H_sample / cosmo.H0
        E_b, dE_b = natural_cubic_eval(z_fit, E_sample, z_grid)
        diag = diagnostic_from_E(z_grid, E_b, dE_b, cosmo)
        for key in keys:
            samples[key].append(diag[key].to_numpy(float))

    central = add_bands_from_samples(central, samples)
    central["reconstruction"] = label
    central["H0_anchor_sigma"] = np.nan if h0_prior_sigma is None else h0_prior_sigma
    return central


def se_kernel(x1: np.ndarray, x2: np.ndarray, amp: float, length: float) -> np.ndarray:
    d = x1[:, None] - x2[None, :]
    return amp**2 * np.exp(-0.5 * (d / length) ** 2)


def gp_residual_predict(
    z_train: np.ndarray,
    H_train: np.ndarray,
    sigma_train: np.ndarray,
    z_grid: np.ndarray,
    cosmo: Cosmology,
    length_scale: float = 0.65,
) -> tuple[np.ndarray, np.ndarray]:
    y = H_train / cosmo.H0
    sigma_y = sigma_train / cosmo.H0
    mean_train = lcdm_E(z_train, cosmo)
    mean_grid, dmean_grid, _ = cpl_E_and_derivative(z_grid, cosmo, w0=-1.0, wa=0.0)
    residual = y - mean_train
    amp = max(float(np.nanstd(residual)), float(np.nanmedian(sigma_y)), 0.03)

    K = se_kernel(z_train, z_train, amp, length_scale)
    K += np.diag(sigma_y**2 + 1e-8)
    alpha = np.linalg.solve(K, residual)

    Ks = se_kernel(z_grid, z_train, amp, length_scale)
    pred_residual = Ks @ alpha
    dKs = -((z_grid[:, None] - z_train[None, :]) / length_scale**2) * Ks
    dpred_residual = dKs @ alpha
    return mean_grid + pred_residual, dmean_grid + dpred_residual


def gp_reconstruction(
    df: pd.DataFrame,
    cosmo: Cosmology,
    z_grid: np.ndarray,
    nboot: int = 600,
    seed: int = 23,
    h0_prior_sigma: float | None = None,
    length_scale: float = 0.65,
    label: str = "gp_planck_anchor",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = df["z"].to_numpy(float)
    H = df["H_km_s_Mpc"].to_numpy(float)
    sigma_H = df["sigma_H_km_s_Mpc"].to_numpy(float)
    z_fit, H_fit, sigma_fit = with_optional_h0_anchor(z, H, sigma_H, cosmo, h0_prior_sigma)

    E, dE = gp_residual_predict(z_fit, H_fit, sigma_fit, z_grid, cosmo, length_scale)
    central = diagnostic_from_E(z_grid, E, dE, cosmo)
    keys = [
        "H",
        "dH_dz",
        "S_rel",
        "dlnS_dlna",
        "Omega_m",
        "Omega_geom",
        "Delta_H",
        "w_geom",
    ]
    samples = {key: [] for key in keys}
    for _ in range(nboot):
        H_sample = rng.normal(H_fit, sigma_fit)
        if h0_prior_sigma is not None:
            H_sample[0] = rng.normal(cosmo.H0, h0_prior_sigma)
        E_b, dE_b = gp_residual_predict(
            z_fit,
            H_sample,
            sigma_fit,
            z_grid,
            cosmo,
            length_scale,
        )
        diag = diagnostic_from_E(z_grid, E_b, dE_b, cosmo)
        for key in keys:
            samples[key].append(diag[key].to_numpy(float))

    central = add_bands_from_samples(central, samples)
    central["reconstruction"] = label
    central["H0_anchor_sigma"] = np.nan if h0_prior_sigma is None else h0_prior_sigma
    central["gp_length_scale"] = length_scale
    return central


def bayesian_cpl_reconstruction(
    df: pd.DataFrame,
    cosmo: Cosmology,
    z_grid: np.ndarray,
    nsteps: int = 9000,
    burn: int = 2000,
    thin: int = 8,
    seed: int = 404,
    label: str = "bayesian_cpl",
) -> pd.DataFrame:
    """Simple Metropolis-Hastings CPL reconstruction.

    This is a prototype Bayesian block, not a full cosmological likelihood.
    It keeps H0, Omega_m0, and Omega_r0 fixed and samples only (w0, wa) from
    cosmic-chronometer H(z) data under a CPL background.
    """

    rng = np.random.default_rng(seed)
    z_obs = df["z"].to_numpy(float)
    H_obs = df["H_km_s_Mpc"].to_numpy(float)
    sigma = df["sigma_H_km_s_Mpc"].to_numpy(float)

    def logpost(params: np.ndarray) -> float:
        w0, wa = params
        if not (-2.2 < w0 < -0.2 and -4.0 < wa < 4.0):
            return -np.inf
        E, _, _ = cpl_E_and_derivative(z_obs, cosmo, w0=w0, wa=wa)
        H_model = cosmo.H0 * E
        chi2 = np.sum(((H_obs - H_model) / sigma) ** 2)
        # Weak regularizing prior that discourages extreme CPL excursions.
        prior = ((w0 + 1.0) / 0.8) ** 2 + (wa / 2.0) ** 2
        return -0.5 * (chi2 + prior)

    current = np.array([-1.0, 0.0], dtype=float)
    current_lp = logpost(current)
    proposal = np.array([0.045, 0.18])
    chain = []
    accepted = 0
    for step in range(nsteps):
        trial = current + rng.normal(0.0, proposal)
        trial_lp = logpost(trial)
        if np.log(rng.random()) < trial_lp - current_lp:
            current = trial
            current_lp = trial_lp
            accepted += 1
        if step >= burn and (step - burn) % thin == 0:
            chain.append(current.copy())

    chain_arr = np.asarray(chain, dtype=float)
    if chain_arr.size == 0:
        chain_arr = np.array([[-1.0, 0.0]])

    keys = [
        "H",
        "dH_dz",
        "S_rel",
        "dlnS_dlna",
        "Omega_m",
        "Omega_geom",
        "Delta_H",
        "w_geom",
    ]
    samples = {key: [] for key in keys}
    for w0, wa in chain_arr:
        diag = model_diagnostic(z_grid, cosmo, model=label, w0=float(w0), wa=float(wa))
        for key in keys:
            samples[key].append(diag[key].to_numpy(float))

    w0_med = float(np.nanmedian(chain_arr[:, 0]))
    wa_med = float(np.nanmedian(chain_arr[:, 1]))
    central = model_diagnostic(z_grid, cosmo, model=label, w0=w0_med, wa=wa_med)
    central = add_bands_from_samples(central, samples)
    central["reconstruction"] = label
    central["w0_p16"] = float(np.nanpercentile(chain_arr[:, 0], 16))
    central["w0_p50"] = w0_med
    central["w0_p84"] = float(np.nanpercentile(chain_arr[:, 0], 84))
    central["wa_p16"] = float(np.nanpercentile(chain_arr[:, 1], 16))
    central["wa_p50"] = wa_med
    central["wa_p84"] = float(np.nanpercentile(chain_arr[:, 1], 84))
    central["acceptance_rate"] = accepted / max(nsteps, 1)
    return central


def lcdm_diagnostic(z: np.ndarray, cosmo: Cosmology) -> pd.DataFrame:
    E, dE, _ = cpl_E_and_derivative(z, cosmo, w0=-1.0, wa=0.0)
    return diagnostic_from_E(z, E, dE, cosmo)


def model_diagnostic(
    z: np.ndarray,
    cosmo: Cosmology,
    model: str,
    w0: float = -1.0,
    wa: float = 0.0,
) -> pd.DataFrame:
    E, dE, w_true = cpl_E_and_derivative(z, cosmo, w0=w0, wa=wa)
    diag = diagnostic_from_E(z, E, dE, cosmo)
    diag["model"] = model
    diag["w_true"] = w_true
    diag["w_error"] = diag["w_geom"] - diag["w_true"]
    return diag


def lcdm_sanity_check(z: np.ndarray, cosmo: Cosmology) -> dict[str, float]:
    """Verify that exact flat LCDM gives Delta_H=0 and w_geom=-1."""

    diag = model_diagnostic(z, cosmo, model="LCDM", w0=-1.0, wa=0.0)
    mask = diag["Omega_geom"].to_numpy(float) > 0.03
    delta = diag.loc[mask, "Delta_H"].to_numpy(float)
    w = diag.loc[mask, "w_geom"].to_numpy(float)
    return {
        "max_abs_Delta_H": float(np.nanmax(np.abs(delta))),
        "max_abs_w_plus_1": float(np.nanmax(np.abs(w + 1.0))),
        "z_min_checked": float(diag.loc[mask, "z"].min()),
        "z_max_checked": float(diag.loc[mask, "z"].max()),
    }


def synthetic_model_checks(z: np.ndarray, cosmo: Cosmology) -> pd.DataFrame:
    models = [
        ("LCDM", -1.0, 0.0),
        ("wCDM_w=-0.9", -0.9, 0.0),
        ("wCDM_w=-1.1", -1.1, 0.0),
        ("CPL_w0=-0.95_wa=0.30", -0.95, 0.30),
    ]
    rows = []
    for name, w0, wa in models:
        diag = model_diagnostic(z, cosmo, model=name, w0=w0, wa=wa)
        mask = diag["Omega_geom"].to_numpy(float) > 0.03
        if name == "LCDM":
            target_delta = np.zeros(mask.sum())
        else:
            target_delta = (
                3.0
                * (1.0 + diag.loc[mask, "w_true"].to_numpy(float))
                * diag.loc[mask, "Omega_geom"].to_numpy(float)
            )
        rows.append(
            {
                "model": name,
                "w0": w0,
                "wa": wa,
                "max |Delta_H - target|": float(
                    np.nanmax(np.abs(diag.loc[mask, "Delta_H"].to_numpy(float) - target_delta))
                ),
                "max |w_geom - w_true|": float(
                    np.nanmax(np.abs(diag.loc[mask, "w_error"].to_numpy(float)))
                ),
                "z range": f'{diag.loc[mask, "z"].min():.3f} to {diag.loc[mask, "z"].max():.3f}',
            }
        )
    return pd.DataFrame(rows)


def interp_linear(x: np.ndarray, y: np.ndarray, xq: np.ndarray | float) -> np.ndarray:
    return np.interp(xq, x, y, left=y[0], right=y[-1])


def growth_prediction_from_background(
    background: pd.DataFrame,
    cosmo: Cosmology,
    z_eval: np.ndarray,
    sigma8_0: float | None = None,
    n_steps: int = 1800,
) -> pd.DataFrame:
    """Predict f sigma_8(z) for a smooth H(z) background in GR.

    This is a prototype consistency check. It assumes smooth dark energy,
    GR linear growth, and normalization D(z=0)=1.
    """

    sigma8 = cosmo.sigma8_0 if sigma8_0 is None else sigma8_0
    z_bg = background["z"].to_numpy(float)
    H_col = "H_p50" if "H_p50" in background else "H"
    dH_col = "dH_dz_p50" if "dH_dz_p50" in background else "dH_dz"
    E_bg = background[H_col].to_numpy(float) / cosmo.H0
    dE_bg = background[dH_col].to_numpy(float) / cosmo.H0

    z_max = max(float(np.nanmax(z_eval)), float(np.nanmax(z_bg)))
    a_min = 1.0 / (1.0 + z_max)
    x_grid = np.linspace(math.log(a_min), 0.0, n_steps)
    a_grid = np.exp(x_grid)
    z_grid = 1.0 / a_grid - 1.0

    E = interp_linear(z_bg, E_bg, z_grid)
    dE_dz = interp_linear(z_bg, dE_bg, z_grid)
    dlnH_dlna = -(1.0 + z_grid) * dE_dz / np.maximum(E, 1e-12)
    omega_m = cosmo.Omega_m0 * (1.0 + z_grid) ** 3 / np.maximum(E**2, 1e-12)
    dlnH_dlna = np.clip(dlnH_dlna, -10.0, 10.0)
    omega_m = np.clip(omega_m, 0.0, 5.0)

    def rhs(idx: int, state: np.ndarray) -> np.ndarray:
        D, P = state
        return np.array(
            [
                P,
                -(2.0 + dlnH_dlna[idx]) * P + 1.5 * omega_m[idx] * D,
            ]
        )

    D = np.empty_like(x_grid)
    P = np.empty_like(x_grid)
    D[0] = a_grid[0]
    P[0] = a_grid[0]
    for i in range(len(x_grid) - 1):
        h = x_grid[i + 1] - x_grid[i]
        state = np.array([D[i], P[i]])
        k1 = rhs(i, state)
        k2 = rhs(i, state + 0.5 * h * k1)
        k3 = rhs(i, state + 0.5 * h * k2)
        k4 = rhs(i + 1, state + h * k3)
        new_state = state + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        D[i + 1], P[i + 1] = new_state

    D_norm = D / D[-1]
    P_norm = P / D[-1]
    f_growth = P_norm / np.maximum(D_norm, 1e-12)
    fsigma8 = f_growth * sigma8 * D_norm

    # Interpolate from ascending z for output.
    order = np.argsort(z_grid)
    z_sorted = z_grid[order]
    D_sorted = D_norm[order]
    f_sorted = f_growth[order]
    fs8_sorted = fsigma8[order]
    return pd.DataFrame(
        {
            "z": z_eval,
            "D": interp_linear(z_sorted, D_sorted, z_eval),
            "f": interp_linear(z_sorted, f_sorted, z_eval),
            "fsigma8_pred": interp_linear(z_sorted, fs8_sorted, z_eval),
        }
    )


def growth_chi2_stat(prediction: pd.DataFrame, growth_data: pd.DataFrame) -> dict[str, float]:
    model = interp_linear(
        prediction["z"].to_numpy(float),
        prediction["fsigma8_pred"].to_numpy(float),
        growth_data["z"].to_numpy(float),
    )
    obs = growth_data["fsigma8"].to_numpy(float)
    sigma = growth_data["sigma_fsigma8"].to_numpy(float)
    chi2 = float(np.nansum(((obs - model) / sigma) ** 2))
    dof = int(len(obs))
    return {"chi2": chi2, "dof": dof, "chi2_per_point": chi2 / max(dof, 1)}


def reconstruction_summary(
    reconstructions: dict[str, pd.DataFrame],
    z_points: Iterable[float] = (0.0, 0.5, 1.0),
) -> pd.DataFrame:
    rows = []
    for name, rec in reconstructions.items():
        z = rec["z"].to_numpy(float)
        for z0 in z_points:
            row = {"reconstruction": name, "z": z0}
            for key in ["Delta_H", "w_geom", "H"]:
                for suffix in ["p16", "p50", "p84"]:
                    col = f"{key}_{suffix}"
                    if col in rec:
                        row[col] = float(interp_linear(z, rec[col].to_numpy(float), z0))
            rows.append(row)
    return pd.DataFrame(rows)


def finite_range(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if math.isclose(lo, hi):
        lo -= 1.0
        hi += 1.0
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def svg_plot(
    title: str,
    xlabel: str,
    ylabel: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    lines: list[dict],
    bands: list[dict] | None = None,
    points: list[dict] | None = None,
    hlines: list[dict] | None = None,
    width: int = 780,
    height: int = 460,
) -> str:
    bands = bands or []
    points = points or []
    hlines = hlines or []
    ml, mr, mt, mb = 72, 24, 48, 62
    pw, ph = width - ml - mr, height - mt - mb

    def sx(x: float) -> float:
        return ml + (x - xlim[0]) / (xlim[1] - xlim[0]) * pw

    def sy(y: float) -> float:
        return mt + (ylim[1] - y) / (ylim[1] - ylim[0]) * ph

    def path_from_xy(x: np.ndarray, y: np.ndarray) -> str:
        finite = np.isfinite(x) & np.isfinite(y)
        coords = [(sx(float(a)), sy(float(b))) for a, b in zip(x[finite], y[finite])]
        if not coords:
            return ""
        chunks = [f"M {coords[0][0]:.2f} {coords[0][1]:.2f}"]
        chunks += [f"L {a:.2f} {b:.2f}" for a, b in coords[1:]]
        return " ".join(chunks)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#172033}",
        ".axis{stroke:#263142;stroke-width:1.2}",
        ".grid{stroke:#d9dee8;stroke-width:1}",
        ".tick{font-size:12px;fill:#4b5568}",
        ".title{font-size:18px;font-weight:700}",
        ".label{font-size:13px;font-weight:600}",
        ".legend{font-size:12px}",
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text class="title" x="{ml}" y="28">{title}</text>',
    ]

    for frac in np.linspace(0, 1, 6):
        x = xlim[0] + frac * (xlim[1] - xlim[0])
        px = sx(x)
        svg.append(f'<line class="grid" x1="{px:.2f}" y1="{mt}" x2="{px:.2f}" y2="{mt+ph}"/>')
        svg.append(f'<text class="tick" x="{px:.2f}" y="{mt+ph+20}" text-anchor="middle">{x:.2g}</text>')
    for frac in np.linspace(0, 1, 6):
        y = ylim[0] + frac * (ylim[1] - ylim[0])
        py = sy(y)
        svg.append(f'<line class="grid" x1="{ml}" y1="{py:.2f}" x2="{ml+pw}" y2="{py:.2f}"/>')
        svg.append(f'<text class="tick" x="{ml-10}" y="{py+4:.2f}" text-anchor="end">{y:.2g}</text>')

    svg.append(f'<line class="axis" x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}"/>')
    svg.append(f'<line class="axis" x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}"/>')
    svg.append(f'<text class="label" x="{ml+pw/2}" y="{height-18}" text-anchor="middle">{xlabel}</text>')
    svg.append(f'<text class="label" transform="translate(18 {mt+ph/2}) rotate(-90)" text-anchor="middle">{ylabel}</text>')

    for hline in hlines:
        y = hline["y"]
        if ylim[0] <= y <= ylim[1]:
            py = sy(y)
            svg.append(
                f'<line x1="{ml}" y1="{py:.2f}" x2="{ml+pw}" y2="{py:.2f}" '
                f'stroke="{hline.get("color", "#777")}" stroke-dasharray="6 5" stroke-width="1.4"/>'
            )

    for band in bands:
        x = np.asarray(band["x"], dtype=float)
        ylo = np.asarray(band["ylo"], dtype=float)
        yhi = np.asarray(band["yhi"], dtype=float)
        finite = np.isfinite(x) & np.isfinite(ylo) & np.isfinite(yhi)
        if not np.any(finite):
            continue
        xf = x[finite]
        lo = ylo[finite]
        hi = yhi[finite]
        upper = [(sx(float(a)), sy(float(b))) for a, b in zip(xf, hi)]
        lower = [(sx(float(a)), sy(float(b))) for a, b in zip(xf[::-1], lo[::-1])]
        poly = " ".join(f"{a:.2f},{b:.2f}" for a, b in upper + lower)
        svg.append(
            f'<polygon points="{poly}" fill="{band.get("color", "#4f83cc")}" '
            f'opacity="{band.get("opacity", 0.18)}"/>'
        )

    for line in lines:
        x = np.asarray(line["x"], dtype=float)
        y = np.asarray(line["y"], dtype=float)
        path = path_from_xy(x, y)
        if path:
            svg.append(
                f'<path d="{path}" fill="none" stroke="{line.get("color", "#1f77b4")}" '
                f'stroke-width="{line.get("width", 2.2)}" stroke-dasharray="{line.get("dash", "")}"/>'
            )

    for pointset in points:
        xs = np.asarray(pointset["x"], dtype=float)
        ys = np.asarray(pointset["y"], dtype=float)
        yerr = np.asarray(pointset.get("yerr", np.zeros_like(ys)), dtype=float)
        color = pointset.get("color", "#202938")
        for x, y, e in zip(xs, ys, yerr):
            if not np.isfinite([x, y, e]).all():
                continue
            px, py = sx(float(x)), sy(float(y))
            pylo, pyhi = sy(float(y - e)), sy(float(y + e))
            svg.append(f'<line x1="{px:.2f}" y1="{pylo:.2f}" x2="{px:.2f}" y2="{pyhi:.2f}" stroke="{color}" stroke-width="1"/>')
            svg.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3.2" fill="{color}"/>')

    legend_x, legend_y = ml + 14, mt + 18
    legend_items = [item for item in lines if item.get("label")]
    legend_items += [item for item in points if item.get("label")]
    for i, item in enumerate(legend_items):
        y = legend_y + i * 18
        color = item.get("color", "#111")
        svg.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+22}" y2="{y}" stroke="{color}" stroke-width="2.4"/>')
        svg.append(f'<text class="legend" x="{legend_x+30}" y="{y+4}">{item["label"]}</text>')

    svg.append("</svg>")
    return "\n".join(svg)


def make_report(
    df: pd.DataFrame,
    rec: pd.DataFrame,
    lcdm: pd.DataFrame,
    out_path: Path,
    reconstruction_label: str,
    h0_table_html: str,
    sanity_html: str,
    synthetic_html: str,
    robustness_html: str,
    growth_data: pd.DataFrame,
    growth_predictions: dict[str, pd.DataFrame],
    growth_chi2_html: str,
) -> None:
    z_grid = rec["z"].to_numpy()
    xlim = (0.0, float(max(df["z"].max(), z_grid.max())))

    def band_plot(key: str, title: str, ylabel: str, hlines=None, ylim=None) -> str:
        y_values = []
        for suffix in ["_p16", "_p50", "_p84"]:
            if f"{key}{suffix}" in rec:
                y_values.extend(rec[f"{key}{suffix}"].to_numpy())
        if key in lcdm:
            y_values.extend(lcdm[key].to_numpy())
        ylim_local = ylim or finite_range(y_values)
        return svg_plot(
            title=title,
            xlabel="redshift z",
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim_local,
            bands=[
                {
                    "x": z_grid,
                    "ylo": rec[f"{key}_p16"],
                    "yhi": rec[f"{key}_p84"],
                    "color": "#2f80ed",
                    "opacity": 0.18,
                }
            ],
            lines=[
                {"x": z_grid, "y": rec[f"{key}_p50"], "label": "HEDD bootstrap median", "color": "#1657b8"},
                {"x": z_grid, "y": lcdm[key], "label": "flat LCDM reference", "color": "#c2410c", "dash": "7 5"},
            ],
            hlines=hlines or [],
        )

    H_ylim = finite_range(
        list(df["H_km_s_Mpc"] - df["sigma_H_km_s_Mpc"])
        + list(df["H_km_s_Mpc"] + df["sigma_H_km_s_Mpc"])
        + list(rec["H_p16"])
        + list(rec["H_p84"])
    )
    H_svg = svg_plot(
        title="Reconstructed H(z)",
        xlabel="redshift z",
        ylabel="H(z) [km/s/Mpc]",
        xlim=xlim,
        ylim=H_ylim,
        bands=[{"x": z_grid, "ylo": rec["H_p16"], "yhi": rec["H_p84"], "color": "#2f80ed", "opacity": 0.18}],
        lines=[
            {"x": z_grid, "y": rec["H_p50"], "label": "HEDD bootstrap median", "color": "#1657b8"},
            {"x": z_grid, "y": lcdm["H"], "label": "flat LCDM reference", "color": "#c2410c", "dash": "7 5"},
        ],
        points=[
            {
                "x": df["z"],
                "y": df["H_km_s_Mpc"],
                "yerr": df["sigma_H_km_s_Mpc"],
                "label": "cosmic chronometers",
                "color": "#111827",
            }
        ],
    )

    delta_svg = band_plot(
        "Delta_H",
        "HEDD residual Delta_H(z)",
        "Delta_H",
        hlines=[{"y": 0.0, "color": "#111827"}],
        ylim=finite_range(list(rec["Delta_H_p16"]) + list(rec["Delta_H_p84"]) + [0.0]),
    )
    w_svg = band_plot(
        "w_geom",
        "Effective geometric equation of state",
        "w_geom",
        hlines=[{"y": -1.0, "color": "#111827"}],
        ylim=(-2.5, 0.5),
    )
    S_svg = band_plot("S_rel", "Relative apparent-horizon entropy", "S_H / S_H0")
    Om_svg = band_plot("Omega_geom", "Residual geometric density fraction", "Omega_geom")

    growth_lines = []
    growth_values = []
    colors = ["#1657b8", "#047857", "#7c3aed", "#b45309", "#c2410c"]
    for color, (name, pred) in zip(colors, growth_predictions.items()):
        growth_lines.append(
            {
                "x": pred["z"],
                "y": pred["fsigma8_pred"],
                "label": name,
                "color": color,
                "dash": "7 5" if "LCDM" in name else "",
            }
        )
        growth_values.extend(pred["fsigma8_pred"].to_numpy(float))
    growth_values.extend((growth_data["fsigma8"] - growth_data["sigma_fsigma8"]).to_numpy(float))
    growth_values.extend((growth_data["fsigma8"] + growth_data["sigma_fsigma8"]).to_numpy(float))
    growth_svg = svg_plot(
        title="Growth consistency: f sigma_8(z)",
        xlabel="redshift z",
        ylabel="f sigma_8",
        xlim=(0.0, float(max(growth_data["z"].max(), z_grid.max()))),
        ylim=finite_range(growth_values),
        lines=growth_lines,
        points=[
            {
                "x": growth_data["z"],
                "y": growth_data["fsigma8"],
                "yerr": growth_data["sigma_fsigma8"],
                "label": "compiled f sigma_8 data",
                "color": "#111827",
            }
        ],
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>HEDD prototype report</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;max-width:980px;margin:32px auto;color:#172033;line-height:1.45}}
code{{background:#eef2f7;padding:2px 5px;border-radius:4px}}
.note{{background:#f8fafc;border-left:4px solid #2f80ed;padding:12px 16px;margin:16px 0}}
svg{{max-width:100%;height:auto;border:1px solid #e5e7eb;margin:16px 0 28px 0}}
table{{border-collapse:collapse;font-size:13px}}
td,th{{border:1px solid #d6dbe5;padding:5px 8px}}
</style>
</head>
<body>
<h1>HEDD prototype report</h1>
<p><strong>Horizon Entropic Dark Energy Diagnostic.</strong> This is a first-pass
observational diagnostic, not a final cosmological analysis.</p>
<div class="note">
Assumptions: flat FLRW background, GR at background level, conserved matter and
radiation, apparent horizon R_H=c/H, and S_H proportional to H^-2. This report
shows the <strong>{reconstruction_label}</strong> reconstruction.
</div>
<div class="note">
The current output should not be interpreted as evidence for w != -1. It is a
methodological prototype showing how the diagnostic can be computed from H(z)
data. The H'(z) reconstruction is not robust enough for cosmological claims.
</div>
<p>Central equations:</p>
<pre>
S_H / S_H0 = (H0 / H)^2
Delta_H = 2(1+z) H'(z)/H(z) - 3 Omega_m(z) - 4 Omega_r(z)
w_geom = -1 + Delta_H / (3 Omega_geom)
</pre>
<h2>Block 1: synthetic sanity checks</h2>
<p>Before interpreting real data, the diagnostic is checked against an exact
flat LCDM reference. It must return Delta_H = 0 and w_geom = -1 up to numerical
precision. The same recovery check is also run for controlled wCDM and CPL
backgrounds.</p>
{sanity_html}
{synthetic_html}
<h2>Block 2: real H(z) data</h2>
<p>The diagnostic is applied to the cosmic-chronometer H(z) compilation. The
default visible reconstruction is Planck-H0 anchored; the unanchored version is
kept as a stress test.</p>
<h2>Core diagnostic plots</h2>
{H_svg}
{S_svg}
{delta_svg}
{w_svg}
{Om_svg}
<h2>Block 3: robustness</h2>
<p>The unanchored cosmic-chronometer-only polynomial can infer an H(0) that is
in tension with the H0 used in the density normalization. The script therefore
writes both unanchored and Planck-anchored outputs.</p>
{h0_table_html}
<p>Prototype method comparison at selected redshifts:</p>
{robustness_html}
<h2>Block 4: growth consistency</h2>
<p>The reconstructed H(z) is used as a smooth-background input for the GR linear
growth equation. This is a consistency check, not a full growth likelihood.</p>
{growth_svg}
{growth_chi2_html}
<h2>Block 5: statistical conclusion</h2>
<p>Current conclusion: HEDD passes the controlled synthetic identities. With the
current chronometer-only prototype reconstruction, no robust cosmological
claim should be made. In particular, the output is not evidence for w != -1.
A serious claim would require robustness across reconstruction methods,
multiple data sets, strong exclusion of w = -1, growth consistency, and
independent reproduction.</p>
<h2>Data source</h2>
<p>The input CSV contains the 32 cosmic-chronometer H(z) points listed in
Table 1 of Gomez-Valent &amp; Amendola, MNRAS 523, 3406-3422 (2023), with the
original references quoted there. The growth CSV contains 20 f sigma_8 points
from the compilation in EPJC 82, 831 (2022), Table 2.</p>
<h2>Method caveat</h2>
<p>The current derivative H'(z) is estimated from a weighted cubic polynomial
bootstrap. This is intentionally conservative as a prototype. A publication
version should use a GP or a full likelihood with the covariance matrix,
plus BAO/SN constraints and consistency tests with f sigma_8(z).</p>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def h0_summary_table(reconstructions: dict[str, pd.DataFrame], cosmo: Cosmology) -> str:
    table = h0_summary_frame(reconstructions, cosmo)
    return table.to_html(index=False, float_format=lambda x: f"{x:.4g}")


def h0_summary_frame(reconstructions: dict[str, pd.DataFrame], cosmo: Cosmology) -> pd.DataFrame:
    rows = []
    for name, rec in reconstructions.items():
        first = rec.iloc[0]
        rows.append(
            {
                "reconstruction": name,
                "H(0) median": first["H_p50"],
                "H(0) 16-84": f'{first["H_p16"]:.2f} to {first["H_p84"]:.2f}',
                "Delta_H(0) median": first["Delta_H_p50"],
                "w_geom(0) median": first["w_geom_p50"],
            }
        )
    table = pd.DataFrame(rows)
    table.insert(1, "normalization H0", cosmo.H0)
    return table


def sanity_frame(check: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "test": "exact flat LCDM",
                "max |Delta_H|": check["max_abs_Delta_H"],
                "max |w_geom + 1|": check["max_abs_w_plus_1"],
                "z range": f'{check["z_min_checked"]:.3f} to {check["z_max_checked"]:.3f}',
            }
        ]
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cosmo = Cosmology()
    df = read_data()
    growth_df = read_growth_data()
    z_grid = np.linspace(0.0, float(df["z"].max()), 220)
    rec_unanchored, _, _ = bootstrap_reconstruction(
        df,
        cosmo,
        z_grid,
        degree=3,
        nboot=1200,
        seed=42,
        label="unanchored",
    )
    rec_anchor, _, _ = bootstrap_reconstruction(
        df,
        cosmo,
        z_grid,
        degree=3,
        nboot=1200,
        seed=42,
        h0_prior_sigma=cosmo.H0_sigma,
        label="planck_anchor",
    )
    rec_gp = gp_reconstruction(
        df,
        cosmo,
        z_grid,
        nboot=450,
        seed=123,
        h0_prior_sigma=cosmo.H0_sigma,
        length_scale=0.65,
        label="gp_planck_anchor",
    )
    rec_spline = spline_reconstruction(
        df,
        cosmo,
        z_grid,
        nboot=900,
        seed=321,
        h0_prior_sigma=cosmo.H0_sigma,
        label="spline_planck_anchor",
    )
    rec_bayes = bayesian_cpl_reconstruction(
        df,
        cosmo,
        z_grid,
        nsteps=9000,
        burn=2000,
        thin=8,
        seed=404,
        label="bayesian_cpl",
    )
    lcdm = lcdm_diagnostic(z_grid, cosmo)
    sanity = lcdm_sanity_check(z_grid, cosmo)
    synthetic = synthetic_model_checks(z_grid, cosmo)
    sanity_table = sanity_frame(sanity).to_html(
        index=False, float_format=lambda x: f"{x:.3e}"
    )
    synthetic_table = synthetic.to_html(index=False, float_format=lambda x: f"{x:.3e}")
    h0_frame = h0_summary_frame(
        {"unanchored": rec_unanchored, "planck_anchor": rec_anchor}, cosmo
    )
    h0_table = h0_frame.to_html(index=False, float_format=lambda x: f"{x:.4g}")
    robustness = reconstruction_summary(
        {
            "poly_unanchored": rec_unanchored,
            "poly_planck_anchor": rec_anchor,
            "gp_planck_anchor": rec_gp,
            "spline_planck_anchor": rec_spline,
            "bayesian_cpl": rec_bayes,
        }
    )
    robustness_table = robustness.to_html(index=False, float_format=lambda x: f"{x:.4g}")

    growth_lcdm = growth_prediction_from_background(
        lcdm, cosmo, growth_df["z"].to_numpy(float)
    )
    growth_anchor = growth_prediction_from_background(
        rec_anchor, cosmo, growth_df["z"].to_numpy(float)
    )
    growth_gp = growth_prediction_from_background(
        rec_gp, cosmo, growth_df["z"].to_numpy(float)
    )
    growth_spline = growth_prediction_from_background(
        rec_spline, cosmo, growth_df["z"].to_numpy(float)
    )
    growth_bayes = growth_prediction_from_background(
        rec_bayes, cosmo, growth_df["z"].to_numpy(float)
    )
    growth_predictions = {
        "poly anchored HEDD": growth_anchor,
        "GP anchored HEDD": growth_gp,
        "spline anchored HEDD": growth_spline,
        "Bayesian CPL HEDD": growth_bayes,
        "LCDM reference": growth_lcdm,
    }
    growth_stats = pd.DataFrame(
        [
            {"model": name, **growth_chi2_stat(pred, growth_df)}
            for name, pred in growth_predictions.items()
        ]
    )
    growth_chi2_table = growth_stats.to_html(index=False, float_format=lambda x: f"{x:.4g}")

    rec_anchor.to_csv(OUT_DIR / "hedd_reconstruction.csv", index=False)
    rec_anchor.to_csv(OUT_DIR / "hedd_reconstruction_planck_anchor.csv", index=False)
    rec_unanchored.to_csv(OUT_DIR / "hedd_reconstruction_unanchored.csv", index=False)
    rec_gp.to_csv(OUT_DIR / "hedd_reconstruction_gp_planck_anchor.csv", index=False)
    rec_spline.to_csv(OUT_DIR / "hedd_reconstruction_spline_planck_anchor.csv", index=False)
    rec_bayes.to_csv(OUT_DIR / "hedd_reconstruction_bayesian_cpl.csv", index=False)
    lcdm.to_csv(OUT_DIR / "lcdm_reference.csv", index=False)
    synthetic.to_csv(OUT_DIR / "synthetic_model_checks.csv", index=False)
    robustness.to_csv(OUT_DIR / "robustness_summary.csv", index=False)
    growth_stats.to_csv(OUT_DIR / "growth_chi2_summary.csv", index=False)
    growth_lcdm.to_csv(OUT_DIR / "growth_lcdm_reference.csv", index=False)
    growth_anchor.to_csv(OUT_DIR / "growth_poly_planck_anchor.csv", index=False)
    growth_gp.to_csv(OUT_DIR / "growth_gp_planck_anchor.csv", index=False)
    growth_spline.to_csv(OUT_DIR / "growth_spline_planck_anchor.csv", index=False)
    growth_bayes.to_csv(OUT_DIR / "growth_bayesian_cpl.csv", index=False)
    make_report(
        df,
        rec_anchor,
        lcdm,
        OUT_DIR / "hedd_report.html",
        reconstruction_label="Planck H0 anchored",
        h0_table_html=h0_table,
        sanity_html=sanity_table,
        synthetic_html=synthetic_table,
        robustness_html=robustness_table,
        growth_data=growth_df,
        growth_predictions=growth_predictions,
        growth_chi2_html=growth_chi2_table,
    )
    sanity_frame(sanity).to_csv(OUT_DIR / "lcdm_sanity_check.csv", index=False)

    summary = rec_anchor.loc[
        np.isclose(rec_anchor["z"], 0.0)
        | np.isclose(rec_anchor["z"], 0.5, atol=0.005)
        | np.isclose(rec_anchor["z"], 1.0, atol=0.005)
    ][["z", "H_p50", "Delta_H_p50", "w_geom_p50", "Omega_geom_p50"]]
    print("Wrote outputs/hedd_reconstruction.csv")
    print("Wrote outputs/hedd_reconstruction_planck_anchor.csv")
    print("Wrote outputs/hedd_reconstruction_unanchored.csv")
    print("Wrote outputs/hedd_reconstruction_gp_planck_anchor.csv")
    print("Wrote outputs/hedd_reconstruction_spline_planck_anchor.csv")
    print("Wrote outputs/hedd_reconstruction_bayesian_cpl.csv")
    print("Wrote outputs/lcdm_reference.csv")
    print("Wrote outputs/lcdm_sanity_check.csv")
    print("Wrote outputs/synthetic_model_checks.csv")
    print("Wrote outputs/robustness_summary.csv")
    print("Wrote outputs/growth_chi2_summary.csv")
    print("Wrote outputs/hedd_report.html")
    print()
    print("LCDM identity sanity check:")
    print(sanity_frame(sanity).to_string(index=False, float_format=lambda x: f"{x: .3e}"))
    print()
    print("Synthetic model recovery:")
    print(synthetic.to_string(index=False, float_format=lambda x: f"{x: .3e}"))
    print()
    print("H0 consistency:")
    print(h0_frame.to_string(index=False, float_format=lambda x: f"{x: .4g}"))
    print()
    print("Growth chi2 summary:")
    print(growth_stats.to_string(index=False, float_format=lambda x: f"{x: .4g}"))
    print()
    print("Bayesian CPL posterior summary:")
    bayes_first = rec_bayes.iloc[0]
    print(
        f"w0={bayes_first['w0_p50']:.3g} "
        f"[{bayes_first['w0_p16']:.3g}, {bayes_first['w0_p84']:.3g}], "
        f"wa={bayes_first['wa_p50']:.3g} "
        f"[{bayes_first['wa_p16']:.3g}, {bayes_first['wa_p84']:.3g}], "
        f"acceptance={bayes_first['acceptance_rate']:.3g}"
    )
    print()
    print(summary.to_string(index=False, float_format=lambda x: f"{x: .4g}"))


if __name__ == "__main__":
    main()
