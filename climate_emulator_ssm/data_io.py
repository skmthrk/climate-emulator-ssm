"""Read annual global-mean CSVs, preserving their original model years."""
import csv
from pathlib import Path

import numpy as np


def load_data(variable, model_id, experiment_id, data_dir):
    path = Path(data_dir) / f'{variable}_{model_id}_{experiment_id}.csv'
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    return (np.array([int(row['year']) for row in rows]),
            np.array([float(row[variable]) for row in rows]))


def create_datasets(model_ids, variables, experiment_ids, data_dir):
    if isinstance(model_ids, str):
        model_ids = [model_ids]
    return {model: {variable: {experiment: load_data(variable, model, experiment, data_dir)
                              for experiment in experiment_ids}
                    for variable in variables}
            for model in model_ids}
