"""Create the HEDD v2 notebook without requiring nbformat."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent if ROOT.name == "src" else ROOT
NOTEBOOK = PROJECT_ROOT / "notebooks" / "HEDD_diagnostic_notebook.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(True),
    }


cells = [
    md(
        r"""# HEDD v2: Horizon Entropic Dark Energy Diagnostic

This notebook is a proof-of-concept computational diagnostic, not a discovery
claim.

HEDD tests whether the dark/geometric component reconstructed from `H(z)` is
compatible with geometric equilibrium:

```text
S_H / S_H0 = (H0 / H)^2
Delta_H = 2(1+z) H'(z)/H(z) - 3 Omega_m(z) - 4 Omega_r(z)
w_geom = -1 + Delta_H / (3 Omega_geom)
```

Interpretation:

- `Delta_H = 0`: compatible with Lambda-like equilibrium, `w = -1`.
- `Delta_H != 0`: possible effective evolution, but not automatically new physics.

Evidence for a real deviation would require synthetic recovery tests,
method robustness, multiple datasets, strong exclusion of `w=-1`, consistency
with growth, and independent reproduction.
"""
    ),
    code(
        """from pathlib import Path
import sys

import numpy as np
import pandas as pd
from IPython.display import SVG, display

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from hedd_analysis import (
    Cosmology,
    read_data,
    read_growth_data,
    bootstrap_reconstruction,
    gp_reconstruction,
    spline_reconstruction,
    bayesian_cpl_reconstruction,
    lcdm_diagnostic,
    lcdm_sanity_check,
    synthetic_model_checks,
    reconstruction_summary,
    growth_prediction_from_background,
    growth_chi2_stat,
    svg_plot,
    finite_range,
)

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

cosmo = Cosmology()
hz = read_data(ROOT / "data" / "hz_cosmic_chronometers.csv")
growth = read_growth_data(ROOT / "data" / "fsigma8_compilation.csv")
z_grid = np.linspace(0.0, hz["z"].max(), 220)
"""
    ),
    md(
        r"""## Block 1: Sanity checks

The diagnostic must recover controlled backgrounds before it is trusted on
real data:

- Lambda-CDM must give `Delta_H = 0`, `w_geom = -1`.
- wCDM with fixed `w` must recover that fixed `w`.
- CPL must recover `w(z)=w0+wa z/(1+z)`.
"""
    ),
    code(
        """sanity = lcdm_sanity_check(z_grid, cosmo)
synthetic = synthetic_model_checks(z_grid, cosmo)

display(pd.DataFrame([sanity]))
display(synthetic)

assert sanity["max_abs_Delta_H"] < 1e-12
assert sanity["max_abs_w_plus_1"] < 1e-12
assert synthetic["max |w_geom - w_true|"].max() < 1e-12
"""
    ),
    md(
        r"""## Block 2: Real H(z) data

Now apply HEDD to the cosmic-chronometer data. The default reconstruction is
Planck-H0 anchored to avoid mixing a fixed density normalization with an
unanchored `H(0)`.
"""
    ),
    code(
        """lcdm = lcdm_diagnostic(z_grid, cosmo)

rec_anchor, _, _ = bootstrap_reconstruction(
    hz,
    cosmo,
    z_grid,
    degree=3,
    nboot=1200,
    seed=42,
    h0_prior_sigma=cosmo.H0_sigma,
    label="poly_planck_anchor",
)

rec_unanchored, _, _ = bootstrap_reconstruction(
    hz,
    cosmo,
    z_grid,
    degree=3,
    nboot=1200,
    seed=42,
    label="poly_unanchored",
)

rec = rec_anchor
rec[["z", "H_p50", "Delta_H_p50", "w_geom_p50", "Omega_geom_p50"]].head()
"""
    ),
    code(
        """def display_svg(svg):
    display(SVG(svg))


def band_plot(rec, key, title, ylabel, hlines=None, ylim=None):
    y_values = []
    for suffix in ["_p16", "_p50", "_p84"]:
        col = f"{key}{suffix}"
        if col in rec:
            y_values.extend(rec[col].to_numpy())
    if key in lcdm:
        y_values.extend(lcdm[key].to_numpy())
    ylim_local = ylim or finite_range(y_values)
    return svg_plot(
        title=title,
        xlabel="redshift z",
        ylabel=ylabel,
        xlim=(0.0, float(hz["z"].max())),
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
            {"x": z_grid, "y": rec[f"{key}_p50"], "label": "HEDD median", "color": "#1657b8"},
            {"x": z_grid, "y": lcdm[key], "label": "flat LCDM reference", "color": "#c2410c", "dash": "7 5"},
        ],
        hlines=hlines or [],
    )
"""
    ),
    code(
        """H_ylim = finite_range(
    list(hz["H_km_s_Mpc"] - hz["sigma_H_km_s_Mpc"])
    + list(hz["H_km_s_Mpc"] + hz["sigma_H_km_s_Mpc"])
    + list(rec["H_p16"])
    + list(rec["H_p84"])
)

display_svg(
    svg_plot(
        title="Reconstructed H(z)",
        xlabel="redshift z",
        ylabel="H(z) [km/s/Mpc]",
        xlim=(0.0, float(hz["z"].max())),
        ylim=H_ylim,
        bands=[{"x": z_grid, "ylo": rec["H_p16"], "yhi": rec["H_p84"], "color": "#2f80ed", "opacity": 0.18}],
        lines=[
            {"x": z_grid, "y": rec["H_p50"], "label": "HEDD median", "color": "#1657b8"},
            {"x": z_grid, "y": lcdm["H"], "label": "flat LCDM reference", "color": "#c2410c", "dash": "7 5"},
        ],
        points=[
            {
                "x": hz["z"],
                "y": hz["H_km_s_Mpc"],
                "yerr": hz["sigma_H_km_s_Mpc"],
                "label": "cosmic chronometers",
                "color": "#111827",
            }
        ],
    )
)

display_svg(band_plot(rec, "S_rel", "Relative apparent-horizon entropy", "S_H / S_H0"))
display_svg(
    band_plot(
        rec,
        "Delta_H",
        "HEDD residual Delta_H(z)",
        "Delta_H",
        hlines=[{"y": 0.0, "color": "#111827"}],
        ylim=finite_range(list(rec["Delta_H_p16"]) + list(rec["Delta_H_p84"]) + [0.0]),
    )
)
display_svg(
    band_plot(
        rec,
        "w_geom",
        "Effective geometric equation of state",
        "w_geom",
        hlines=[{"y": -1.0, "color": "#111827"}],
        ylim=(-2.5, 0.5),
    )
)
"""
    ),
    md(
        r"""## Block 3: Robustness

Compare multiple reconstruction choices:

- polynomial, unanchored;
- polynomial, Planck-H0 anchored;
- GP residual reconstruction around Lambda-CDM;
- natural-cubic spline, Planck-H0 anchored;
- Bayesian-CPL posterior reconstruction.

The spline is intentionally kept as a stress test because it can amplify noisy
derivatives.
"""
    ),
    code(
        """rec_gp = gp_reconstruction(
    hz,
    cosmo,
    z_grid,
    nboot=450,
    seed=123,
    h0_prior_sigma=cosmo.H0_sigma,
    length_scale=0.65,
    label="gp_planck_anchor",
)

rec_spline = spline_reconstruction(
    hz,
    cosmo,
    z_grid,
    nboot=900,
    seed=321,
    h0_prior_sigma=cosmo.H0_sigma,
    label="spline_planck_anchor",
)

rec_bayes = bayesian_cpl_reconstruction(
    hz,
    cosmo,
    z_grid,
    nsteps=9000,
    burn=2000,
    thin=8,
    seed=404,
    label="bayesian_cpl",
)

recs = {
    "poly_unanchored": rec_unanchored,
    "poly_planck_anchor": rec_anchor,
    "gp_planck_anchor": rec_gp,
    "spline_planck_anchor": rec_spline,
    "bayesian_cpl": rec_bayes,
}

robustness = reconstruction_summary(recs)
display(robustness)
"""
    ),
    code(
        """def compare_plot(key, ylabel, hline=None, ylim=None):
    colors = ["#6b7280", "#1657b8", "#047857", "#7c3aed", "#b45309"]
    lines = []
    values = []
    for color, (name, r) in zip(colors, recs.items()):
        y = r[f"{key}_p50"].to_numpy(float)
        values.extend(y)
        lines.append({"x": r["z"], "y": y, "label": name, "color": color})
    hlines = [] if hline is None else [{"y": hline, "color": "#111827"}]
    return svg_plot(
        title=f"Robustness comparison: {key}",
        xlabel="redshift z",
        ylabel=ylabel,
        xlim=(0.0, float(hz["z"].max())),
        ylim=ylim or finite_range(values + ([] if hline is None else [hline])),
        lines=lines,
        hlines=hlines,
    )


display_svg(compare_plot("Delta_H", "Delta_H", hline=0.0))
display_svg(compare_plot("w_geom", "w_geom", hline=-1.0, ylim=(-2.5, 0.5)))
"""
    ),
    md(
        r"""## Block 4: Growth consistency

Use reconstructed `H(z)` as input to the GR linear growth equation and compare
the predicted `f sigma_8(z)` with a small compiled RSD/peculiar-velocity data
set. This is a consistency check, not a full growth likelihood.
"""
    ),
    code(
        """growth_predictions = {
    "poly anchored HEDD": growth_prediction_from_background(rec_anchor, cosmo, growth["z"].to_numpy(float)),
    "GP anchored HEDD": growth_prediction_from_background(rec_gp, cosmo, growth["z"].to_numpy(float)),
    "spline anchored HEDD": growth_prediction_from_background(rec_spline, cosmo, growth["z"].to_numpy(float)),
    "Bayesian CPL HEDD": growth_prediction_from_background(rec_bayes, cosmo, growth["z"].to_numpy(float)),
    "LCDM reference": growth_prediction_from_background(lcdm, cosmo, growth["z"].to_numpy(float)),
}

growth_stats = pd.DataFrame(
    [{"model": name, **growth_chi2_stat(pred, growth)} for name, pred in growth_predictions.items()]
)
display(growth_stats)
"""
    ),
    code(
        """colors = ["#1657b8", "#047857", "#7c3aed", "#b45309", "#c2410c"]
lines = []
values = []
for color, (name, pred) in zip(colors, growth_predictions.items()):
    lines.append(
        {
            "x": pred["z"],
            "y": pred["fsigma8_pred"],
            "label": name,
            "color": color,
            "dash": "7 5" if "LCDM" in name else "",
        }
    )
    values.extend(pred["fsigma8_pred"].to_numpy(float))

values.extend((growth["fsigma8"] - growth["sigma_fsigma8"]).to_numpy(float))
values.extend((growth["fsigma8"] + growth["sigma_fsigma8"]).to_numpy(float))

display_svg(
    svg_plot(
        title="Growth consistency: f sigma_8(z)",
        xlabel="redshift z",
        ylabel="f sigma_8",
        xlim=(0.0, float(growth["z"].max())),
        ylim=finite_range(values),
        lines=lines,
        points=[
            {
                "x": growth["z"],
                "y": growth["fsigma8"],
                "yerr": growth["sigma_fsigma8"],
                "label": "compiled data",
                "color": "#111827",
            }
        ],
    )
)
"""
    ),
    md(
        r"""## Block 5: Statistical conclusion

Decision rule for this prototype:

- Passing mocks means the formula/code is internally consistent.
- Agreement across reconstruction methods would make the diagnostic more
  credible.
- Excluding `Delta_H = 0` or `w=-1` requires robust uncertainties, multiple
  data sets, and systematic checks.
- Growth must agree with the same background under GR if the interpretation is
  smooth geometric/dark energy.

Current status: methodological prototype. No discovery claim.
"""
    ),
    code(
        """rec_anchor.to_csv(OUT / "hedd_reconstruction.csv", index=False)
rec_anchor.to_csv(OUT / "hedd_reconstruction_planck_anchor.csv", index=False)
rec_unanchored.to_csv(OUT / "hedd_reconstruction_unanchored.csv", index=False)
rec_gp.to_csv(OUT / "hedd_reconstruction_gp_planck_anchor.csv", index=False)
rec_spline.to_csv(OUT / "hedd_reconstruction_spline_planck_anchor.csv", index=False)
rec_bayes.to_csv(OUT / "hedd_reconstruction_bayesian_cpl.csv", index=False)
lcdm.to_csv(OUT / "lcdm_reference.csv", index=False)
synthetic.to_csv(OUT / "synthetic_model_checks.csv", index=False)
robustness.to_csv(OUT / "robustness_summary.csv", index=False)
growth_stats.to_csv(OUT / "growth_chi2_summary.csv", index=False)

display(
    rec_anchor.loc[
        (np.abs(rec_anchor["z"] - 0.0) < 0.005)
        | (np.abs(rec_anchor["z"] - 0.5) < 0.005)
        | (np.abs(rec_anchor["z"] - 1.0) < 0.005),
        ["z", "H_p50", "Delta_H_p50", "w_geom_p50", "Omega_geom_p50"],
    ]
)
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Wrote {NOTEBOOK}")
