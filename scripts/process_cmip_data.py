"""Convert CMIP6 monthly Amon files to annual global-mean CSVs.

Uses areacella or rectangular latitude/longitude cell bounds for spatial
weights, and the declared calendar's month lengths for annual weights.
Only complete calendar years are written, including years split across files.
Monthly values are assumed to represent whole calendar months.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import xarray as xr


def select_sources(input_dir, variable, model, experiment, variant=None, grid=None):
    paths = sorted(Path(input_dir).glob(f'{variable}_Amon_{model}_{experiment}_*_*.nc'))
    paths = [p for p in paths if len(p.stem.split('_')) == 7
             and (variant is None or p.stem.split('_')[4] == variant)
             and (grid is None or p.stem.split('_')[5] == grid)]
    identities = {tuple(p.stem.split('_')[4:6]) for p in paths}
    if len(identities) != 1:
        raise ValueError(f'{variable} {model} {experiment}: supply one member/grid '
                         f'(found {sorted(identities)}); use --variant-label or --grid-label')
    return paths


def cell_area(dataset, source, input_dir):
    _, _, model, experiment, variant, grid, _ = source.stem.split('_')
    exact = Path(input_dir) / f'areacella_fx_{model}_{experiment}_{variant}_{grid}.nc'
    paths = [exact] if exact.exists() else sorted(
        Path(input_dir).glob(f'areacella_fx_{model}_*_{variant}_{grid}.nc'))
    if paths:
        with xr.open_dataset(paths[0]) as fixed:
            return fixed['areacella'].astype('float64').load()
    lat, lon = dataset['lat'], dataset['lon']
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError(f'{source.name}: this grid requires areacella')
    latitude = dataset[lat.attrs['bounds']].values
    longitude = dataset[lon.attrs['bounds']].values
    band = np.abs(np.diff(np.sin(np.deg2rad(latitude)), axis=1))[:, 0]
    widths = np.abs(longitude[:, 1] - longitude[:, 0])
    widths = np.where(widths > 180, 360 - widths, widths)
    # The common radius and angular conversion cancel in normalized weights.
    return xr.DataArray(band[:, None] * widths[None, :],
                        coords={lat.dims[0]: lat, lon.dims[0]: lon},
                        dims=(lat.dims[0], lon.dims[0]))


def annual_global_mean(paths, input_dir):
    monthly = {}
    for path in paths:
        variable = path.stem.split('_')[0]
        with xr.open_dataset(path) as dataset:
            area = cell_area(dataset, path, input_dir)
            weights = area / area.sum(skipna=False)
            data = dataset[variable]
            years = data.time.dt.year.values
            months = data.time.dt.month.values
            days = data.time.dt.days_in_month.values
            for i, (year, month, duration) in enumerate(zip(years, months, days)):
                key = int(year), int(month)
                if key in monthly:
                    raise ValueError(f'{path.name}: duplicate month {key}')
                field, aligned = xr.align(data.isel(time=i).astype('float64'), weights, join='exact')
                mean = float((field * aligned).sum(skipna=False))
                if not np.isfinite(mean):
                    raise ValueError(f'{path.name}: missing global data in {key}')
                monthly[key] = mean, int(duration)
    years, values = [], []
    for year in sorted({year for year, _ in monthly}):
        if all((year, month) in monthly for month in range(1, 13)):
            means, durations = zip(*(monthly[year, month] for month in range(1, 13)))
            years.append(year)
            values.append(float(np.average(means, weights=durations)))
    if not years:
        raise ValueError('No complete calendar years found')
    return years, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', default='data/raw',
                        help='monthly CMIP6 file directory (default: data/raw)')
    parser.add_argument('--output-dir', default='data/processed',
                        help='annual CSV destination (default: data/processed)')
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--variant-label', help='select a member if multiple members are present')
    parser.add_argument('--grid-label', help='select a grid if multiple grids are present')
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for variable in ('tas', 'rsdt', 'rsut', 'rlut'):
        for experiment in ('piControl', 'abrupt-4xCO2'):
            paths = select_sources(args.input_dir, variable, args.model_id, experiment,
                                   args.variant_label, args.grid_label)
            years, values = annual_global_mean(paths, args.input_dir)
            output = output_dir / f'{variable}_{args.model_id}_{experiment}.csv'
            with output.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['year', variable])
                writer.writerows(zip(years, values))
            print(f'wrote {output} ({len(years)} years)')


if __name__ == '__main__':
    main()
