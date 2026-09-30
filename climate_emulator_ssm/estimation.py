"""Estimation pipeline for one CMIP6 model.

Steps
-----
1. Load global-mean annual anomalies (abrupt-4xCO2 minus piControl mean) of
   near-surface air temperature (tas), top-of-atmosphere outgoing shortwave
   (rsut) and outgoing longwave (rlut) radiation.
2. Fit the single-flux models for rsut and rlut (OLS on an AR(1) form for
   starting values, then least squares on the deterministic path).
3. Obtain starting values for the heat capacities and exchange coefficients
   from a Geoffroy et al. (2013)-type two-timescale fit of the temperature path.
4. Estimate the stochastic k-layer energy balance models jointly on
   (rsut, rlut, tas) by maximum likelihood, evaluating the likelihood with the
   Kalman filter. Multi-layer estimates reuse the previous fitted multi-layer
   parameters when available. If a fit has a lower likelihood than an
   available smaller model it approximately nests, it is re-estimated from
   an embedding of that fit.
"""
import numpy as np
import scipy.optimize

from .data_io import create_datasets
from .models import (
    RsutFit,
    RlutFit,
    OneLayerNoFeedbackModel,
    OneLayerModel,
    TwoLayerModel,
    ThreeLayerModel,
    FourLayerModel,
    estimate_ar1_parameters_from_ols,
    estimate_equilibrium_climate_params,
    lamda_lu_base,
    phi_co2,
    SIGN_FLIP_MODELS,
)

VARIABLES = ('rsdt', 'rsut', 'rlut', 'tas')
EXPERIMENTS = ('piControl', 'abrupt-4xCO2')
MODEL_CLASSES = {
    'one-layer-nofeedback': OneLayerNoFeedbackModel,
    'one-layer': OneLayerModel,
    'two-layer': TwoLayerModel,
    'three-layer': ThreeLayerModel,
    'four-layer': FourLayerModel,
}
ALL_MODEL_TYPES = tuple(MODEL_CLASSES)
DEEPER_LAYER = {  # heat capacity and exchange coefficient added at each step
    'two-layer': ('chi_o', 'kappa_o', 'chi_s'),
    'three-layer': ('chi_d', 'kappa_d', 'chi_o'),
    'four-layer': ('chi_b', 'kappa_b', 'chi_d'),
}
# The model each model (approximately) nests: the smaller model is the limit of
# a vanishing exchange coefficient (layered models) or of no radiative feedback
# beyond the reference Planck response (one-layer). Parameters are positive and
# estimated in logs, so the limits are approached but never attained exactly.
NESTS = {
    'one-layer': 'one-layer-nofeedback',
    'two-layer': 'one-layer',
    'three-layer': 'two-layer',
    'four-layer': 'three-layer',
}
DEFAULT_METHODS = ('BFGS', 'L-BFGS-B', 'SLSQP', 'Powell', 'COBYQA', 'BOBYQA')
# Provisional search bounds for radiative adjustment rates (per year).
DEFAULT_BOUNDS = {'gamma_su': (0.5, 50.0), 'gamma_lu': (0.5, 50.0)}


def load_observations(model_id, data_dir, baseline_max_years=251, control_years=None,
                      max_abrupt_years=251):
    """Annual anomalies relative to a shared, constant piControl baseline."""
    data = create_datasets(model_id, VARIABLES, EXPERIMENTS, data_dir)[model_id]
    common = data[VARIABLES[0]]['piControl'][0]
    for variable in VARIABLES[1:]:
        common = np.intersect1d(common, data[variable]['piControl'][0])
    if control_years is None:
        baseline = common[:baseline_max_years]
    else:
        start, end = control_years
        baseline = common[(common >= start) & (common <= end)]
        if len(baseline) != end - start + 1:
            raise ValueError('Requested control window is not available for all variables')
    if len(baseline) == 0 or np.any(np.diff(baseline) != 1):
        raise ValueError('A common consecutive piControl baseline is required')
    years = data[VARIABLES[0]]['abrupt-4xCO2'][0][:max_abrupt_years]
    if len(years) == 0 or np.any(np.diff(years) != 1):
        raise ValueError('A consecutive abrupt-4xCO2 series is required')
    observations = {'years': years - years[0] + 1}
    for variable in VARIABLES:
        control_time, control = data[variable]['piControl']
        abrupt_time, abrupt = data[variable]['abrupt-4xCO2']
        if not np.array_equal(years, abrupt_time[:max_abrupt_years]):
            raise ValueError('abrupt-4xCO2 years differ across variables')
        mean = np.mean(control[np.isin(control_time, baseline)])
        observations[variable] = abrupt[:len(years)] - mean
    observations['rndt'] = observations['rsdt'] - observations['rsut'] - observations['rlut']
    return observations


def fit_flux(variable, observations):
    """Single-flux model for rsut or rlut given the observed temperature path."""
    estimated = estimate_ar1_parameters_from_ols(observations, y_variable=variable, x_variable='tas')
    model = RsutFit() if variable == 'rsut' else RlutFit()
    for key, value in estimated.items():
        model.parameters_default[key] = value

    Y = [np.array([[y]]) for y in observations[variable]]
    if variable == 'rsut':
        U = [np.array([[u]]) for u in observations['tas']]
    else:
        U = [np.array([[u], [model.phi_co2 * np.log(4)]]) for u in observations['tas']]
    model.set_observation(Y)
    model.set_control(U)

    keys = [key for key in model.parameters_default if not key.startswith('sigma')]

    def mean_squared_error(x):
        for idx, key in enumerate(keys):
            model.parameters_default[key] = x[idx]
        _, Y_simulated, _, _ = model.generate_sample()
        simulated = np.array([y[0][0] for y in Y_simulated])
        return np.mean((observations[variable] - simulated) ** 2)

    x0 = np.array([estimated[key] for key in keys])
    res = scipy.optimize.minimize(mean_squared_error, x0=x0, method='Nelder-Mead', tol=1e-8,
                                  options={'maxiter': 5000, 'adaptive': True})
    for idx, key in enumerate(keys):
        estimated[key] = float(res.x[idx])
    return estimated


def starting_values(flux_parameters, observations):
    """Two-layer starting values: radiative parameters plus a two-timescale fit."""
    parameters = dict(flux_parameters)
    for key, value in TwoLayerModel().parameters_default.items():
        parameters.setdefault(key, value)
    parameters.update(estimate_equilibrium_climate_params(observations, parameters))
    return parameters


def fit_ebm(model_type, initial, observations, model_id=None, methods=DEFAULT_METHODS,
            num_attempts=2, seek_global_minimum=True, seed=None, bounds=None):
    """Maximum likelihood estimate of a k-layer model on (rsut, rlut, tas).

    `bounds` maps parameter names to (lower, upper); None means unbounded.
    """
    model = MODEL_CLASSES[model_type](model_id=model_id)
    for key in model.parameters_default:
        if key in initial:
            model.parameters_default[key] = initial[key]
    if model_type == 'three-layer':
        model.parameters_default['chi_d'] = initial.get('chi_d', 2 * initial['chi_o'])
        model.parameters_default['kappa_d'] = initial.get('kappa_d', initial['kappa_o'])
    bounds = bounds or {}
    for key in model.parameters_default:
        model.parameters_bounds[key] = tuple(bounds.get(key, (None, None)))
        lower, upper = model.parameters_bounds[key]
        value = model.parameters_default[key]
        if lower is not None and value < lower:
            model.parameters_default[key] = lower * 1.01
        if upper is not None and value > upper:
            model.parameters_default[key] = upper * 0.99
    model.optimization_methods = list(methods)

    Y = [np.array([[a], [b], [c]]) for a, b, c in
         zip(observations['rsut'], observations['rlut'], observations['tas'])]
    U = [np.array([[model.phi_co2 * np.log(4)]]) for _ in observations['tas']]
    model.set_observation(Y)
    model.set_control(U)

    model.estimate(num_attempts=num_attempts, seed=seed,
                   seek_global_minimum=seek_global_minimum, initial_search=False)
    parameters = {key: float(val) for key, val in zip(model.parameters_default, model.parameters)}
    neg_loglik = float(model.objfun(np.log(model.parameters)))
    k = len(parameters)
    n = len(Y)
    return {
        'parameters': parameters,
        'derived': derived_quantities(parameters, model_id=model_id),
        'neg_log_likelihood': neg_loglik,
        'num_parameters': k,
        'num_years': n,
        'aic': 2 * k + 2 * neg_loglik,
        'bic': k * np.log(n) + 2 * neg_loglik,
    }


def derived_quantities(parameters, model_id=None, forcing_coefficient=phi_co2):
    """Net feedback and equilibrium warming implied by the radiative parameters."""
    lamda_su = parameters.get('lamda_su', 0.0)  # no-feedback model: no albedo feedback
    lamda_lu = parameters.get('lamda_lu', lamda_lu_base)  # and only the Planck response
    if model_id in SIGN_FLIP_MODELS:
        lamda_net = lamda_lu + lamda_su
    else:
        lamda_net = lamda_lu - lamda_su
    forcing_2x = forcing_coefficient * np.log(2)
    sign = -1.0 if model_id in SIGN_FLIP_MODELS else 1.0
    return {
        # Notation: albedo (shortwave) feedback lambda_alpha,
        # water-vapour/greenhouse (longwave) feedback lambda_beta, and the
        # Planck response lambda_out = (1 - beta) 4 sigma T^3 at T = 288 K.
        'lamda_alpha': float(sign * lamda_su),
        'lamda_beta': float(lamda_lu_base - lamda_lu),
        'lamda_planck': float(lamda_lu_base),
        'lamda_net': float(lamda_net),
        'forcing_2xCO2': float(forcing_2x),
        'equilibrium_warming_2xCO2': float(forcing_2x / lamda_net),
    }


def nested_start(model_type, previous):
    """Embedding of the smaller model's estimate in `model_type`.

    Layered models: the new layer is nearly decoupled (exchange 0.01, capacity
    five times the layer above). One-layer model: shortwave feedback near zero
    and longwave response equal to the reference Planck response.
    """
    start = dict(previous)
    if model_type == 'one-layer':
        start['lamda_su'] = 1e-3
        start['lamda_lu'] = lamda_lu_base
        return start
    capacity, exchange, upper = DEEPER_LAYER[model_type]
    start[capacity] = 5.0 * previous[upper]
    start[exchange] = 0.01
    return start


def estimate_model(model_id, data_dir, model_types=('two-layer', 'three-layer'),
                   methods=DEFAULT_METHODS, num_attempts=2, seek_global_minimum=True, seed=0,
                   bounds=DEFAULT_BOUNDS, nested_restart=True, baseline_max_years=251,
                   control_years=None, max_abrupt_years=251):
    """Estimate the requested model types, retaining the better nested restart.

    `bounds` defaults to DEFAULT_BOUNDS; pass {} to estimate without bounds.
    """
    model_types = sorted(dict.fromkeys(model_types), key=ALL_MODEL_TYPES.index)
    observations = load_observations(model_id, data_dir, baseline_max_years, control_years,
                                     max_abrupt_years)
    flux = {variable: fit_flux(variable, observations) for variable in ('rsut', 'rlut')}
    radiative = flux['rsut'] | flux['rlut']
    # Starting values for the heat capacities use the signed net feedback
    # (lamda_lu - lamda_su, with lamda_su < 0 for SIGN_FLIP_MODELS) ...
    initial = starting_values(radiative, observations)
    if model_id in SIGN_FLIP_MODELS:
        # ... while the EBM uses the magnitude; the sign is applied inside the model.
        initial['lamda_su'] = abs(initial['lamda_su'])
    record = {'model_id': model_id, 'models': {}}
    options = dict(model_id=model_id, methods=methods, num_attempts=num_attempts,
                   seek_global_minimum=seek_global_minimum, seed=seed, bounds=bounds)
    current = dict(initial)
    for model_type in model_types:
        result = fit_ebm(model_type, current, observations, **options)
        smaller = record['models'].get(NESTS.get(model_type))
        if (nested_restart and smaller is not None
                and result['neg_log_likelihood'] > smaller['neg_log_likelihood'] + 1e-6):
            alternative = fit_ebm(model_type, nested_start(model_type, smaller['parameters']),
                                  observations, **options)
            if alternative['neg_log_likelihood'] < result['neg_log_likelihood']:
                result = alternative
        record['models'][model_type] = result
        if model_type in DEEPER_LAYER:
            current = dict(current) | result['parameters']
    return record
