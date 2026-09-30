"""Estimate EBMs from annual CSVs; write one parameter CSV per model type."""
import argparse
import csv
import multiprocessing as mp
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from climate_emulator_ssm.estimation import (  # noqa: E402
    ALL_MODEL_TYPES, EXPERIMENTS, VARIABLES, estimate_model)


def available_models(data_dir):
    found = {}
    for path in Path(data_dir).glob('*.csv'):
        parts = path.stem.split('_')
        if len(parts) == 3:
            variable, model_id, experiment = parts
            found.setdefault(model_id, set()).add((variable, experiment))
    required = {(v, e) for v in VARIABLES for e in EXPERIMENTS}
    return sorted(model for model, have in found.items() if required <= have)


def run_one(model_id, data_dir, model_types, seed):
    try:
        record = estimate_model(model_id, data_dir, model_types=model_types, seed=seed)
        print(f'[{model_id}] done', flush=True)
        return record
    except Exception as exc:
        print(f'[{model_id}] failed: {exc}', flush=True)
        return None


def write_results(records, output_dir, model_type):
    rows = []
    for record in records:
        result = record['models'].get(model_type)
        if result is not None:
            rows.append({'model_id': record['model_id'], **result['parameters'], **result['derived'],
                         **{key: result[key] for key in ('neg_log_likelihood', 'num_parameters',
                                                        'num_years', 'aic', 'bic')}})
    if rows:
        path = Path(output_dir) / f'{model_type}.csv'
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'wrote {path} ({len(rows)} models)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data/processed',
                        help='annual CSV directory (default: data/processed)')
    parser.add_argument('--output-dir', default='results', help='CSV destination; existing files are replaced')
    models = parser.add_mutually_exclusive_group(required=True)
    models.add_argument('--model-id', action='append', help='CMIP6 model (repeatable)')
    models.add_argument('--all', action='store_true', help='estimate all models with complete input files')
    parser.add_argument('--model-types', nargs='+', default=['two-layer', 'three-layer'],
                        choices=[*ALL_MODEL_TYPES, 'all'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--processes', type=int, default=1)
    args = parser.parse_args()
    if args.processes < 1:
        parser.error('--processes must be positive')
    model_ids = available_models(args.data_dir) if args.all else sorted(set(args.model_id))
    if not model_ids:
        parser.error('no models with complete input files found')
    model_types = list(ALL_MODEL_TYPES) if 'all' in args.model_types else args.model_types
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    job = partial(run_one, data_dir=args.data_dir, model_types=model_types, seed=args.seed)
    if args.processes > 1:
        with mp.Pool(args.processes) as pool:
            records = pool.map(job, model_ids)
    else:
        records = [job(model_id) for model_id in model_ids]
    completed = [record for record in records if record is not None]
    for model_type in model_types:
        write_results(completed, args.output_dir, model_type)
    print(f'Finished: {len(completed)} succeeded, {len(records) - len(completed)} failed')
    if len(completed) != len(records):
        sys.exit(1)


if __name__ == '__main__':
    main()
