# Horizon Entropic Dark Energy Diagnostic (HEDD)

HEDD is a proof-of-concept computational diagnostic for testing whether the
dark component reconstructed from cosmic expansion is compatible with
horizon-entropy equilibrium.

Mission statement:

> HEDD tests whether the dark component reconstructed from cosmic expansion is
> compatible with horizon-entropy equilibrium (`w = -1`), or whether it shows
> effective non-equilibrium evolution detectable through `H(z)` and structure
> growth.

This repository does not claim a discovery of dynamical dark energy. It is a
prototype diagnostic pipeline.

## Core diagnostic

For a flat FLRW background with apparent horizon entropy `S_H proportional to
H^-2`:

```text
Delta_H(z) = d ln S_H / d ln a - 3 Omega_m(z) - 4 Omega_r(z)
```

Using `S_H proportional to H^-2`, the data-facing form is:

```text
Delta_H(z) = 2(1+z) H'(z)/H(z) - 3 Omega_m(z) - 4 Omega_r(z)
```

and the effective residual geometric equation of state is:

```text
w_geom(z) = -1 + Delta_H(z) / (3 Omega_geom(z))
```

Interpretation:

- `Delta_H = 0`: compatible with Lambda-like horizon equilibrium.
- `Delta_H > 0`: effective quintessence-like reconstruction.
- `Delta_H < 0`: effective phantom-like reconstruction.

## What would count as evidence?

Strong evidence would require all of the following:

1. HEDD passes controlled synthetic tests for LCDM, wCDM, and CPL.
2. The result is robust to reconstruction method.
3. The same pattern appears in more than one independent expansion dataset.
4. The uncertainties exclude `w = -1` with strong significance.
5. Growth data `f sigma_8(z)` confirm the interpretation under GR, or reveal a
   clear expansion-growth inconsistency.
6. Independent researchers can reproduce the result.

At present this repository satisfies only the first methodological layer and a
prototype version of the robustness/growth checks.

## Repository layout

```text
data/
  fsigma8_compilation.csv
  hz_cosmic_chronometers.csv
notebooks/
  HEDD_diagnostic_notebook.ipynb
outputs/
  generated diagnostic outputs
src/
  hedd_analysis.py
  make_hedd_notebook.py
CITATION.cff
README.md
references.bib
requirements.txt
```

## Notebook blocks

1. Sanity checks: LCDM, wCDM with `w=-0.9`, wCDM with `w=-1.1`, and CPL.
2. Real `H(z)` data: reconstruct `H(z)`, `H'(z)`, `S_H(z)`, `Delta_H(z)`, and
   `w_geom(z)`.
3. Robustness: Planck anchor, unanchored, GP-residual, spline, and
   Bayesian-CPL reconstructions.
4. Growth: predict `f sigma_8(z)` from the reconstructed background and compare
   to data.
5. Statistical conclusion: state whether the current output supports Lambda or
   deviation.

## Run

From the repository root, install the requirements and run the analysis:

```bash
pip install -r requirements.txt
python src/hedd_analysis.py

To regenerate the notebook skeleton:

```powershell
python src/make_hedd_notebook.py
```

## Current validation status

The synthetic recovery tests currently pass to numerical precision:

```text
LCDM:          w_geom -> -1
wCDM w=-0.9:  w_geom -> -0.9
wCDM w=-1.1:  w_geom -> -1.1
CPL:           w_geom -> w0 + wa z/(1+z)
```

The real-data analysis remains a prototype. The derivative `H'(z)` is the weak
point and must be treated carefully.

## Scientific caveats

- The current output should not be interpreted as evidence for `w != -1`.
- The GP, spline, and Bayesian-CPL reconstructions are prototype robustness
  checks, not a publication-grade reconstruction.
- The spline can amplify noisy derivatives and is kept as a stress test.
- The Bayesian-CPL block is parametric and should not be mistaken for a full
  cosmological likelihood.
- A publication-grade version needs covariance matrices, BAO/SNe likelihoods,
  CMB priors, propagated `H0` and `Omega_m0` uncertainties, and a formal growth
  comparison.

## Relation to TEC-2 / Cosmic Binding Basins

HEDD and TEC-2 are linked but separate.

TEC-2 asks where gravitational belonging ends: it uses the dark-energy term to
estimate cosmic binding basin boundaries.

HEDD asks what the background term is doing: it diagnoses whether the residual
geometric component behaves like an equilibrium Lambda term or an evolving
horizon-linked component.

If HEDD remains compatible with `w_geom = -1`, TEC-2 with a Lambda baseline is
the natural default. If HEDD robustly found `w_geom != -1`, TEC-2 would need a
redshift-dependent extension using `H(z)`, `Omega_DE(z)`, and `w(z)`.
