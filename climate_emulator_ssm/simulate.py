"""Use estimated parameters as a climate emulator.

Example
-------
>>> import numpy as np
>>> from climate_emulator_ssm.simulate import load_parameters, simulate
>>> table = load_parameters('parameters/three-layer.csv')
>>> forcing = [5.35 * np.log(2)] * 150            # abrupt doubling of CO2
>>> path = simulate('three-layer', table['MIROC6'], forcing, model_id='MIROC6')
>>> path['tas'][-1]                                 # warming after 150 years (K)
"""
import csv

import numpy as np

from .estimation import MODEL_CLASSES


def load_parameters(path, include_unusable=False):
    """Read a parameter CSV into {model_id: {name: numeric_value}}.

    Rows marked "pending" or "do-not-use" are excluded by default. Set
    `include_unusable=True` only to inspect historical/unresolved estimates.
    "caution" rows remain included; use `load_status` to inspect their notes.
    Tables without a status column are read without filtering.
    """
    table = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            model_id = row.pop('model_id')
            if row.get('status') in ('pending', 'do-not-use') and not include_unusable:
                continue
            values = {}
            for key, value in row.items():
                try:
                    values[key] = float(value)
                except (TypeError, ValueError):
                    continue  # text columns such as `note`
            table[model_id] = values
    return table


def load_status(path):
    """{model_id: {'status': ..., 'note': ...}} from a parameter table."""
    with open(path) as f:
        return {row['model_id']: {'status': row.get('status', ''), 'note': row.get('note', '')}
                for row in csv.DictReader(f)}


def simulate(model_type, parameters, forcing, model_id):
    """Deterministic annual response to a CO2 forcing path (W m-2).

    Returns anomalies of outgoing shortwave (rsut), outgoing longwave (rlut)
    and near-surface air temperature (tas), starting from equilibrium.
    `model_id` is required because models whose shortwave feedback sign is
    flipped inside the model definition (GISS-E2-2-G and GISS-E2-2-H) are
    simulated incorrectly under any other name; pass None only for parameter
    sets that do not come from those models.
    """
    model = MODEL_CLASSES[model_type](model_id=model_id)
    model.parameters = [float(parameters[key]) for key in model.parameters_default]
    model.set_control([np.array([[float(f)]]) for f in forcing])
    _, Y, _, _ = model.generate_sample()
    Y = np.array([y[:, 0] for y in Y])
    return {'rsut': Y[:, 0], 'rlut': Y[:, 1], 'tas': Y[:, 2]}
