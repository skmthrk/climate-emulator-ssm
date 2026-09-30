# Parameter tables

The tables retain the original estimates, including their historical annual
aggregation and control baselines. New processing uses calendar-month weights
and new estimation uses one shared control window, so re-estimates can differ.

CIESM, FGOALS-f3-L, KIOST-ESM and NESM3 have unresolved source or time-axis
issues. MIROC-ES2L's four-layer likelihood, AIC and BIC are unreliable; use its
three-layer version. These rows remain for reference and are excluded by
`load_parameters` by default. Other `caution` rows remain available; the
`note` column records their limitations.
This leaves 56 models available by default, or 55 in the four-layer table.
Use `include_unusable=True` only to inspect the excluded estimates.

One CSV file per model type (`one-layer-nofeedback.csv`, `one-layer.csv`,
`two-layer.csv`, `three-layer.csv`, `four-layer.csv`); one row per CMIP6 model
(`model_id`). Parameters absent from a model type are absent from its file. Columns:

| Column | Meaning | Unit |
|---|---|---|
| `lamda_su`, `lamda_lu` | shortwave and longwave feedback parameters | W m⁻² K⁻¹ |
| `gamma_su`, `gamma_lu` | adjustment rates of the shortwave and longwave responses | yr⁻¹ |
| `sigma_su`, `sigma_lu` | process-noise scales of the two flux states | W m⁻² yr⁻¹ᐟ² |
| `sigma_s` | process-noise scale of the surface energy balance | W m⁻² yr¹ᐟ² |
| `chi_s`, `chi_o`, `chi_d`, `chi_b` | heat capacities of the surface, upper-ocean, deep-ocean and bottom layers | W yr m⁻² K⁻¹ |
| `kappa_o`, `kappa_d`, `kappa_b` | heat-exchange coefficients between layers | W m⁻² K⁻¹ |
| `lamda_alpha` | effective shortwave feedback (sign-adjusted `lamda_su`; includes shortwave cloud effects) | W m⁻² K⁻¹ |
| `lamda_beta` | longwave response relative to the reference Planck response, λ_out − `lamda_lu` | W m⁻² K⁻¹ |
| `lamda_planck` | reference Planck response λ_out = (1 − β̄) 4σT̄³ with T̄ = 288 K, β̄ = 0.386 | W m⁻² K⁻¹ |
| `lamda_net` | net feedback `lamda_lu - lamda_su` (`+` for GISS-E2-2-G/H) | W m⁻² K⁻¹ |
| `forcing_2xCO2` | `5.35 ln 2` | W m⁻² |
| `equilibrium_warming_2xCO2` | `forcing_2xCO2 / lamda_net`: equilibrium warming implied by the emulator (ECS*), not a direct CMIP6 ECS estimate | K |
| `neg_log_likelihood`, `aic`, `bic` | fit statistics on (rsut, rlut, tas) | — |
| `num_parameters`, `num_years` | number of estimated parameters and of annual observations | — |
| `status`, `note` | usability label (`ok`, `caution`, `pending`, `do-not-use`) and known issue | — |

The `one-layer-nofeedback` version fixes shortwave feedback to zero and
longwave feedback to the reference Planck response. Newly estimated CSVs
contain parameters and fit statistics; `status` and `note` belong to the
bundled tables and are not assigned automatically to new fits.

The radiative adjustment rates `gamma_su` and `gamma_lu` are restricted
to 0.5–50 per year. These provisional search bounds can affect the estimates;
annual data cannot resolve very fast adjustment rates.
