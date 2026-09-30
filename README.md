# climate-emulator-ssm

State-space energy balance model (EBM) emulators estimated from CMIP6.

## What this repository is

1. Stochastic multi-layer EBMs are estimated by maximum
   likelihood from the `abrupt-4xCO2` experiment of each CMIP6 model. The
   likelihood is evaluated with the Kalman filter.
2. Instead of fitting the net top-of-atmosphere flux,
   the model treats the anomalies of outgoing
   shortwave (`rsut`) and outgoing longwave (`rlut`) radiation as state
   variables with their own feedback parameters and adjustment rates. The net
   climate feedback is therefore identified from its reflected-shortwave and
   emitted-longwave components.
3. One- to four-layer versions are provided, with negative log-likelihood,
   AIC and BIC for comparison. Multi-layer fits reuse the preceding fitted
   multi-layer model's parameters when available.
4. Estimates for 60 CMIP6 models, with the implied
   net feedback and equilibrium warming for doubled CO2. In an economic model the
   collection of parameter sets can be used to represent structural climate
   uncertainty (running the same economic experiment under each climate model's
   behaviour).

## Use

With Python 3.10+, run from the repository root:

```bash
python -m pip install -r requirements.txt
```

```python
import numpy as np
from climate_emulator_ssm.simulate import load_parameters, simulate

parameters = load_parameters("parameters/three-layer.csv")
forcing = np.full(150, 5.35 * np.log(2))
path = simulate("three-layer", parameters["MIROC6"], forcing, model_id="MIROC6")
print(path["tas"][-1])  # warming after 150 years (K)
```

See [parameters/README.md](parameters/README.md) for units and known issues.

## Relation to existing work

- Cummins, Stephenson and Stott (2020) estimate k-box stochastic EBMs by maximum
  likelihood with the Kalman filter from surface temperature and the net
  top-of-atmosphere flux. This repository follows the same state-space approach
  but separates the shortwave and longwave responses.
- Geoffroy et al. (2013) two-timescale fits provide the starting values.
- Folini et al. (2025) and Eftekhari et al. (2026) develop calibration and
  evaluation strategies for climate emulators used in economics. The present code
  is complementary: it provides statistically estimated, model-by-model
  parameters for the temperature response.

## Model

For the two-layer version, with `u` and `v` the anomalies of `rsut` and `rlut`,
`T_s` the surface temperature anomaly and `T_o` the ocean temperature anomaly,

```
du/dt       = -γ_s u - γ_s λ_s T_s          + σ_u dW_u/dt
dv/dt       = -γ_l v + γ_l λ_l T_s - γ_l F  + σ_v dW_v/dt
χ_s dT_s/dt = -u - v - κ_o (T_s - T_o)      + σ_T dW_T/dt
χ_o dT_o/dt =          κ_o (T_s - T_o)
```

(`γ_s = gamma_su`, `λ_s = lamda_su`, `γ_l = gamma_lu`, `λ_l = lamda_lu`,
`σ_u = sigma_su`, `σ_v = sigma_lu`, `σ_T = sigma_s`, `χ = chi`,
`κ = kappa` in the code and tables). In equilibrium `u = -λ_s T_s` and
`v = λ_l T_s - F`, so the net downward flux is `F - (λ_l - λ_s) T_s` and the net
feedback parameter is `λ_l - λ_s`. The three- and four-layer versions add deeper
ocean layers (`chi_d, kappa_d` and `chi_b, kappa_b`). The continuous-time system
is discretised exactly at an annual step (matrix exponential for the drift;
Van Loan's method over a short sub-step followed by interval doubling for the
noise covariance; see `discretize_noise` in
[base_model.py](climate_emulator_ssm/base_model.py)). `rsut`, `rlut` and `tas` are
the observed annual means, treated as point observations of the states.

**Physical interpretation.** The model linearises the global
energy balance around the pre-industrial equilibrium. The absorbed solar
radiation `R_in = rsdt - rsut` changes through the albedo, and the outgoing
longwave radiation `R_out = rlut` through the Planck response and the greenhouse
effect. Writing the albedo feedback as λ_α, the water-vapour/greenhouse feedback
as λ_β and the Planck response of the pre-industrial state as
λ_out = (1 − β̄) 4σT̄³ ≈ 3.33 W m⁻² K⁻¹ (T̄ = 288 K, β̄ = 0.386),

```
lamda_su = λ_α,    lamda_lu = λ_out - λ_β,    net feedback = λ_out - λ_β - λ_α,
```

and the CO2 forcing `5.35 ln(M/M̄)` equals `σT̄⁴ φ ln(M/M̄)` with φ = 0.0137.
The tables report `lamda_alpha`, `lamda_beta` and `lamda_planck` alongside the
estimated parameters (for MIROC6, λ_α ≈ 0.91 and λ_β ≈ 0.99 W m⁻² K⁻¹). These are
statistical, effective quantities: `lamda_alpha` is the effective shortwave
feedback (it includes shortwave cloud effects, not only surface albedo), and
`lamda_beta` is the longwave response relative to the fixed reference Planck
response λ_out (it absorbs longwave cloud effects and any departure of the model's
Planck response from the reference). The equilibrium warming in the tables is the
value implied by the emulator with the fixed forcing below, not a direct estimate
of each CMIP6 model's ECS.

Conventions and assumptions:

- The CO2 forcing of `abrupt-4xCO2` is fixed at `F = 5.35 ln 4` W m⁻² (Myhre et al.
  1998). It is not estimated; model-specific differences in effective forcing are
  absorbed in the other parameters.
- Parameters are estimated in logs (positive). For GISS-E2-2-G and GISS-E2-2-H the
  shortwave feedback enters with the opposite sign (hard-coded in `models.py`).
- Up to the first 251 years of `abrupt-4xCO2` are used (140–251 years in the
  existing tables, 150 for most). New fits subtract a constant `piControl`
  mean over the same original model-year window for all four variables:
  the earliest common consecutive coverage, up to 251 years. No drift
  correction is applied.

## Re-estimation from CMIP6 data

`data/` is excluded from Git. Raw files go in `data/raw/` and processed
annual CSVs in `data/processed/`. Re-estimation requires
your own monthly CMIP6 Amon files for `tas`, `rsdt`, `rsut` and `rlut` in both
`piControl` and `abrupt-4xCO2`, plus matching `areacella` or valid rectangular
cell bounds. Place the files directly in the input directory using CMIP6
filenames (`{variable}_Amon_{model}_{experiment}_{member}_{grid}_{period}.nc`).
Obtain the source files from ESGF and supply one member/grid per experiment,
using the declared parent member for `piControl`.
Monthly values are treated as whole calendar-month means,
weighted by the declared calendar. The scripts write CSV files and replace
existing output files with the same names.

```bash
# 1. CMIP6 monthly (Amon) tas, rsdt, rsut, rlut for piControl and abrupt-4xCO2
#    (and areacella) downloaded from ESGF into data/raw/
python scripts/process_cmip_data.py --input-dir data/raw --output-dir data/processed --model-id MIROC6

# 2. Estimation (one model, or all available models in parallel)
python scripts/estimate.py --data-dir data/processed --output-dir results --model-id MIROC6
python scripts/estimate.py --data-dir data/processed --output-dir results --all --processes 8
```

These data directories are the script defaults; directory flags can be
omitted when using this layout. Estimation defaults to the two- and three-layer
versions, writing
`results/two-layer.csv` and `results/three-layer.csv`. Add `--model-types all`
to fit all five versions. Each output table contains only the models from
that invocation; repeat `--model-id` to include several specific models.
The processed inputs use `{variable}_{model}_{experiment}.csv` with a
`year,{variable}` header. Simulation returns annual `rsut`, `rlut` and `tas`
anomalies as NumPy arrays.

## Funding

The first open component of the climate module of the project supported by a FY2025
research grant from the Noritz Nukumori Foundation.

## References

- Cummins, D. P., D. B. Stephenson and P. A. Stott (2020) “Optimal estimation of
  stochastic energy balance model parameters,” *Journal of Climate*, 33(18),
  7909–7926.
- Eftekhari, A., D. Folini, A. Friedl, F. Kübler, S. Scheidegger and O. Schenk
  (2026) “Building interpretable climate emulators for economics,” *The Economic
  Journal*, advance access, https://doi.org/10.1093/ej/ueaf131.
- Folini, D., A. Friedl, F. Kübler and S. Scheidegger (2025) “The climate in climate
  economics,” *Review of Economic Studies*, 92(1), 299–338.
- Geoffroy, O., D. Saint-Martin, D. J. L. Olivié, A. Voldoire, G. Bellon and
  S. Tytéca (2013) “Transient climate response in a two-layer energy-balance model.
  Part I,” *Journal of Climate*, 26(6), 1841–1857.
- Myhre, G., E. J. Highwood, K. P. Shine and F. Stordal (1998) “New estimates of
  radiative forcing due to well mixed greenhouse gases,” *Geophysical Research
  Letters*, 25(14), 2715–2718.

## Acknowledgements

We acknowledge the World Climate Research Programme, which, through its Working
Group on Coupled Modelling, coordinated and promoted CMIP6. We thank the climate
modelling groups for producing and making available their model output, the Earth
System Grid Federation (ESGF) for archiving the data and providing access, and the
multiple funding agencies who support CMIP6 and ESGF.

## License

The code is MIT licensed (see `LICENSE`). CMIP6 source data are obtained
separately from their providers.
