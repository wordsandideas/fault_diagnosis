"""Controlled sensitivity study for assumed state/input delays d and b."""
import argparse
import csv
import itertools
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from .simulation_data_generator import SimulationDataGenerator
    from .simulation_experiment import SimulationExperiment
except ImportError:
    from simulation_data_generator import SimulationDataGenerator
    from simulation_experiment import SimulationExperiment


def parse_int_list(value):
    try:
        values = [int(part.strip()) for part in value.split(',') if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Expected comma-separated integers') from exc
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError('Provide nonnegative integers')
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--d-values', type=parse_int_list, default=[0, 1, 2, 3, 4])
    parser.add_argument('--b-values', type=parse_int_list, default=[0, 1, 2, 3, 4])
    parser.add_argument('--d-true', type=int, default=2)
    parser.add_argument('--b-true', type=int, default=1)
    parser.add_argument('--seeds', type=parse_int_list, default=[11, 22, 33, 44, 55])
    parser.add_argument('--T', type=int, default=200)
    parser.add_argument('--n-bins', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--k', type=float, default=1.5)
    parser.add_argument('--window-size', type=int, default=20)
    parser.add_argument('--evaluation-start', type=int, default=None)
    parser.add_argument('--fault-start', type=int, default=None)
    parser.add_argument('--fault-types', type=parse_int_list, default=[1, 2])
    parser.add_argument('--profiles', default='step,delayed_sigmoid')
    parser.add_argument('--growth-steps', type=int, default=28)
    parser.add_argument('--fault-delay-steps', type=int, default=2)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--force-data', action='store_true',
                        help='Regenerate data cache; otherwise reuse exact cached sequences')
    args = parser.parse_args(argv)
    if args.d_true < 0 or args.b_true < 0:
        parser.error('True delays must be nonnegative')
    if any(x not in (1, 2) for x in args.fault_types):
        parser.error('fault-types can contain only 1 and/or 2')
    if args.T < 50 or args.n_bins < 3 or args.epochs < 1:
        parser.error('T must be >=50, n-bins >=3, and epochs >=1')
    if args.window_size < 1 or args.k <= 0:
        parser.error('window-size and k must be positive')
    profiles = [item.strip() for item in args.profiles.split(',') if item.strip()]
    if not profiles or any(p not in ('step', 'delayed_sigmoid') for p in profiles):
        parser.error('profiles must be step and/or delayed_sigmoid')
    args.profiles = profiles
    args.evaluation_start = (max(max(args.d_values), max(args.b_values))
                             if args.evaluation_start is None else args.evaluation_start)
    if args.evaluation_start < max(max(args.d_values), max(args.b_values)):
        parser.error('evaluation-start must cover every tested d and b')
    args.fault_start = args.fault_start or max(args.evaluation_start + args.window_size + 10,
                                               int(args.T * 0.55))
    if args.fault_start >= args.T - 5:
        parser.error('fault-start must leave enough post-fault observations')
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data(cache_dir, key, generator, fault_type=None, fault_start=None,
             profile='step', profile_params=None, force=False):
    path = cache_dir / f'{key}.pt'
    if path.exists() and not force:
        return torch.load(path, map_location='cpu', weights_only=False)
    data = generator.generate_data(fault_type=fault_type, fault_start=fault_start,
                                  fault_profile=profile, profile_params=profile_params)
    torch.save(data, path)
    return data


def run(args):
    script_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / 'results' / 'sensitivity_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / 'data_cache' / (
        f'T{args.T}_bins{args.n_bins}_true_d{args.d_true}_b{args.b_true}'
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    run_config = vars(args).copy()
    config_path = output_dir / 'run_config.json'
    result_path = output_dir / 'delay_sensitivity_trials.csv'
    rows = []
    completed = set()
    if result_path.exists() and not args.force_data:
        if config_path.exists():
            with config_path.open('r', encoding='utf-8') as stream:
                previous_config = json.load(stream)
            if previous_config != run_config:
                raise ValueError(
                    f'Existing results use a different configuration: {config_path}. '
                    'Choose a new --output-dir or explicitly remove/archive the old run.'
                )
        with result_path.open('r', newline='', encoding='utf-8-sig') as stream:
            for old_row in csv.DictReader(stream):
                rows.append(old_row)
                completed.add((int(old_row['seed']), int(old_row['d_model']),
                               int(old_row['b_model']), int(old_row['fault_type']),
                               old_row['profile']))
    with config_path.open('w', encoding='utf-8') as stream:
        json.dump(run_config, stream, indent=2, ensure_ascii=False)
    profile_params = {
        'delayed_sigmoid': {
            'delay_steps': args.fault_delay_steps,
            'growth_steps': args.growth_steps,
            'saturation_level': 1.0,
        }
    }
    total_trials = (len(args.seeds) * len(args.d_values) * len(args.b_values)
                    * len(args.fault_types) * len(args.profiles))
    print(f'Planned evaluation rows: {total_trials}; already complete: {len(completed)}',
          flush=True)
    for seed in args.seeds:
        seed_everything(seed)
        generator = SimulationDataGenerator(
            T=args.T, n_bins=args.n_bins, d=args.d_true, b=args.b_true
        )
        train = get_data(cache_dir, f'seed_{seed}_train', generator,
                         force=args.force_data)
        calibration = get_data(cache_dir, f'seed_{seed}_normal_calibration', generator,
                               force=args.force_data)
        tests = {}
        for fault_type, profile in itertools.product(args.fault_types, args.profiles):
            key = f'seed_{seed}_fault_{fault_type}_{profile}'
            tests[(fault_type, profile)] = get_data(
                cache_dir, key, generator, fault_type, args.fault_start, profile,
                profile_params.get(profile), args.force_data
            )

        for d_model, b_model in itertools.product(args.d_values, args.b_values):
            pending = [
                (fault_type, profile)
                for fault_type, profile in itertools.product(args.fault_types, args.profiles)
                if (seed, d_model, b_model, fault_type, profile) not in completed
            ]
            if not pending:
                continue
            # Train once per seed and assumed-delay pair, then reuse for all test conditions.
            seed_everything(seed)
            experiment = SimulationExperiment(
                T=args.T, n_bins=args.n_bins, d=d_model, b=b_model,
                d_true=args.d_true, b_true=args.b_true,
                evaluation_start=args.evaluation_start,
            )
            print(f'Training seed={seed}, d={d_model}, b={b_model} '
                  f'({len(pending)} pending test cases)', flush=True)
            experiment.train_model(*train, epochs=args.epochs, lr=args.lr)
            experiment.calibrate_normal(*calibration, k=args.k,
                                        window_size=args.window_size)
            for fault_type, profile in pending:
                test = tests[(fault_type, profile)]
                result = experiment.evaluate_model(
                    *test, fault_start=args.fault_start, fault_type=fault_type,
                    fault_profile=profile, k=args.k, window_size=args.window_size,
                )
                row = {
                    'seed': seed, 'd_true': args.d_true, 'b_true': args.b_true,
                    'd_model': d_model, 'b_model': b_model,
                    'fault_type': fault_type, 'profile': profile,
                    'fault_start': args.fault_start,
                    'evaluation_start': result['evaluation_start'],
                    'score_start': result['score_start'],
                    'state_rmse': result['state_rmse'],
                    'pdf_rmse_aux': result['pdf_rmse'],
                    'accuracy': result['accuracy'], 'precision': result['precision'],
                    'recall': result['recall'], 'detection_f1': result['f1'],
                    'macro_f1': result['macro_f1'], 'far': result['far'],
                    'fdr': result['fdr'], 'diagnosis_accuracy': result['diagnosis_accuracy'],
                    'detection_delay': result['detection_delay'],
                    'undetected': result['undetected'],
                    'onset_before_evaluation': result['onset_before_evaluation'],
                }
                rows.append(row)
                completed.add((seed, d_model, b_model, fault_type, profile))
                print(f"seed={seed} d={d_model} b={b_model} fault={fault_type}/{profile} "
                      f"RMSE={row['state_rmse']:.5f} Macro-F1={row['macro_f1']:.4f} "
                      f"FAR={row['far']} FDR={row['fdr']} delay={row['detection_delay']} "
                      f"[{len(completed)}/{total_trials}]", flush=True)
                write_header = not result_path.exists() or result_path.stat().st_size == 0
                with result_path.open('a', newline='', encoding='utf-8-sig') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row)
                    stream.flush()

    aggregate_path = output_dir / 'delay_sensitivity_summary.csv'
    metric_keys = ('state_rmse', 'pdf_rmse_aux', 'detection_f1', 'macro_f1',
                   'far', 'fdr', 'diagnosis_accuracy', 'detection_delay')
    groups = {}
    for row in rows:
        key = tuple(int(row[name]) for name in
                    ('d_true', 'b_true', 'd_model', 'b_model', 'fault_type')) + (row['profile'],)
        groups.setdefault(key, []).append(row)
    summary_rows = []
    for key, group in groups.items():
        summary = dict(zip(('d_true', 'b_true', 'd_model', 'b_model', 'fault_type', 'profile'), key))
        summary['n_seeds'] = len(group)
        summary['undetected_rate'] = float(np.mean([
            str(r['undetected']).lower() == 'true' for r in group
        ]))
        for metric in metric_keys:
            values = [float(r[metric]) for r in group
                      if r[metric] not in (None, '', 'None')]
            summary[f'{metric}_mean'] = float(np.mean(values)) if values else None
            summary[f'{metric}_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None
        summary_rows.append(summary)
    if summary_rows:
        with aggregate_path.open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f'\nTrial results: {result_path}')
    print(f'Summary results: {aggregate_path}')
    print(f'Run configuration: {config_path}')
    return rows, summary_rows


def main(argv=None):
    args = parse_args(argv)
    run(args)


if __name__ == '__main__':
    main()
