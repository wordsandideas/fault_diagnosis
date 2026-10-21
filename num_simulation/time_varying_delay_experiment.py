"""Frozen-model evaluation under changing true delays; no training on test scenarios."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch

try:
    from .delay_sensitivity_experiment import parse_int_list, seed_everything
    from .simulation_data_generator import SimulationDataGenerator
    from .simulation_experiment import SimulationExperiment
except ImportError:
    from delay_sensitivity_experiment import parse_int_list, seed_everything
    from simulation_data_generator import SimulationDataGenerator
    from simulation_experiment import SimulationExperiment


SCENARIOS = ('constant', 'step', 'periodic', 'random', 'drift')


def make_delays(args, scenario, seed):
    """Changes begin at source-time change_start, independent of fault onset."""
    d = np.full(args.T, args.d_true, dtype=int)
    b = np.full(args.T, args.b_true, dtype=int)
    n = args.T - args.change_start
    t = np.arange(n)
    rng = np.random.default_rng(seed)
    for seq, base, amplitude in ((d, args.d_true, args.d_amplitude),
                                 (b, args.b_true, args.b_amplitude)):
        if scenario == 'constant':
            offset = np.zeros(n)
        elif scenario == 'step':
            offset = np.full(n, amplitude)
        elif scenario == 'periodic':
            offset = np.rint(amplitude * np.sin(2*np.pi*t/args.period))
        elif scenario == 'random':
            blocks = rng.integers(-amplitude, amplitude+1,
                                  size=(n+args.block_size-1)//args.block_size)
            offset = np.repeat(blocks, args.block_size)[:n]
        elif scenario == 'drift':
            offset = np.rint(amplitude * np.linspace(0, 1, n))
        else:
            raise ValueError(scenario)
        seq[args.change_start:] = np.maximum(0, base + offset).astype(int)
    return d, b


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', type=parse_int_list, default=[11, 22, 33, 44, 55])
    p.add_argument('--d-values', type=parse_int_list, default=[2])
    p.add_argument('--b-values', type=parse_int_list, default=[1])
    for name, default in [('T', 200), ('n-bins', 50), ('epochs', 150),
                          ('d-true', 2), ('b-true', 1), ('d-amplitude', 1),
                          ('b-amplitude', 1), ('change-start', 60),
                          ('fault-start', 110), ('period', 40), ('block-size', 20),
                          ('transition-steps', 20), ('window-size', 20),
                          ('growth-steps', 28), ('fault-delay-steps', 2)]:
        p.add_argument('--'+name, type=int, default=default)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--k', type=float, default=1.5)
    p.add_argument('--scenarios', default=','.join(SCENARIOS))
    p.add_argument('--output-dir', default=str(Path(__file__).resolve().parent /
                                              'results' / 'time_varying_delay_results'))
    a = p.parse_args(argv)
    a.scenarios = list(dict.fromkeys(a.scenarios.split(',')))
    if any(s not in SCENARIOS for s in a.scenarios):
        p.error('Unknown scenario')
    if min(a.d_true, a.b_true, a.d_amplitude, a.b_amplitude,
           a.fault_delay_steps) < 0:
        p.error('Delays and amplitudes must be nonnegative')
    a.evaluation_start = max(a.d_true+a.d_amplitude, a.b_true+a.b_amplitude,
                             max(a.d_values), max(a.b_values))
    if min(a.epochs, a.period, a.block_size, a.transition_steps,
           a.window_size, a.growth_steps, a.lr, a.k) <= 0 or a.n_bins < 3:
        p.error('Training and window parameters must be positive; n-bins >= 3')
    if not (a.evaluation_start+a.window_size < a.change_start <
            a.change_start+a.transition_steps < a.fault_start < a.T-5):
        p.error('Require warmup < change < transition end < fault onset < T-5')
    if len(set(a.seeds)) != len(a.seeds) or len(set(a.d_values)) != len(a.d_values) or len(set(a.b_values)) != len(a.b_values):
        p.error('Seeds and model delay lists must not contain duplicates')
    return a


def write_rows(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / 'run_config.json'
    config = vars(args)
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != config:
        raise ValueError('Configuration differs: choose a new --output-dir')
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    trials_path = output / 'time_varying_delay_trials.csv'
    rows = []
    if trials_path.exists():
        with trials_path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
    identity = lambda r: (int(r['seed']), int(r['d_model']), int(r['b_model']),
                          r['scenario'], int(r['fault_type']), r['profile'])
    completed = {identity(r) for r in rows}
    cases = [(0, 'normal')] + list(itertools.product([1, 2], ['step', 'delayed_sigmoid']))
    total = len(args.seeds)*len(args.d_values)*len(args.b_values)*len(args.scenarios)*len(cases)
    print(f'Evaluations: {total}; completed: {len(completed)}', flush=True)
    for seed in args.seeds:
        seed_everything(seed)
        nominal = SimulationDataGenerator(T=args.T, n_bins=args.n_bins,
                                          d=args.d_true, b=args.b_true)
        train = nominal.generate_data()
        calibration = nominal.generate_data()
        for dm, bm in itertools.product(args.d_values, args.b_values):
            pending = [(s, f, p) for s in args.scenarios for f, p in cases
                       if (seed, dm, bm, s, f, p) not in completed]
            if not pending:
                continue
            seed_everything(seed)
            ex = SimulationExperiment(T=args.T, n_bins=args.n_bins, d=dm, b=bm,
                                      d_true=args.d_true, b_true=args.b_true,
                                      evaluation_start=args.evaluation_start)
            print(f'Training seed={seed}, d={dm}, b={bm}', flush=True)
            ex.train_model(*train, epochs=args.epochs, lr=args.lr)
            ex.calibrate_normal(*calibration, k=args.k, window_size=args.window_size)
            for scenario, fault, profile in pending:
                ds, bs = make_delays(args, scenario, seed)
                # Same random forcing across scenarios and assumed model delays.
                seed_everything(seed*100+fault*10+(profile == 'delayed_sigmoid'))
                generator = SimulationDataGenerator(T=args.T, n_bins=args.n_bins,
                    d=args.d_true, b=args.b_true, d_true_seq=ds, b_true_seq=bs)
                data = generator.generate_data(fault_type=fault or None,
                    fault_start=args.fault_start if fault else None,
                    fault_profile=profile if fault else 'step',
                    profile_params={'delay_steps': args.fault_delay_steps if profile == 'delayed_sigmoid' else 0,
                                    'growth_steps': args.growth_steps})
                # Labels begin at the first nonzero coefficient/fault activation.
                labels = np.zeros(args.T, dtype=int)
                if fault:
                    onset = args.fault_start if profile == 'step' else args.fault_start+args.fault_delay_steps+1
                    labels[onset:] = fault
                result = ex.evaluate_model(*data, true_labels=labels, k=args.k,
                                           window_size=args.window_size)
                times = np.arange(1, args.T)
                flags = result['fault_flags'].cpu().numpy().astype(int)
                raw_error = (result['pred_a_seq']-data[0][1:]).cpu().numpy().reshape(-1)
                valid = result['evaluation_mask'].cpu().numpy()
                row = dict(seed=seed, d_model=dm, b_model=bm, scenario=scenario,
                           fault_type=fault, profile=profile,
                           threshold=ex.normal_calibration['threshold'])
                for key in ('state_rmse', 'f1', 'far', 'fdr', 'detection_delay',
                            'undetected', 'diagnosis_accuracy', 'macro_f1', 'pdf_rmse'):
                    row[key] = result[key]
                for name, lo, hi in [('before_change', 0, args.change_start+1),
                    ('transition', args.change_start+1, args.change_start+args.transition_steps+1),
                    ('after_transition', args.change_start+args.transition_steps+1, args.T)]:
                    mask = valid & (times >= lo) & (times < hi)
                    normal_mask = mask & (labels[1:] == 0)
                    row[name+'_n'] = int(mask.sum())
                    row[name+'_rmse'] = float(np.sqrt(np.mean(raw_error[mask]**2))) if mask.any() else None
                    row[name+'_far'] = float(flags[normal_mask].mean()) if normal_mask.any() else None
                trace_name = f'seed{seed}_d{dm}_b{bm}_{scenario}_{fault}_{profile}.npz'
                np.savez_compressed(output / trace_name, d_true_seq=ds, b_true_seq=bs,
                    target_times=times, true_labels=labels[1:], flags=flags,
                    predicted_labels=result['predicted_labels'].cpu().numpy(),
                    residuals=result['residuals'].cpu().numpy(), errors=raw_error,
                    evaluation_mask=valid)
                rows.append(row)
                completed.add(identity(row))
                write_rows(trials_path, rows)
                print(f'{scenario} fault={fault}/{profile} [{len(completed)}/{total}]', flush=True)
    keys = ('d_model', 'b_model', 'scenario', 'fault_type', 'profile')
    groups = {}
    for row in rows:
        group = tuple(str(row[k]) for k in keys)
        groups.setdefault(group, []).append(row)
    summary = []
    for group, records in groups.items():
        item = dict(zip(keys, group), n_seeds=len(records))
        for metric in rows[0]:
            if metric in keys or metric in ('seed', 'undetected'):
                continue
            values = [float(r[metric]) for r in records if r[metric] not in (None, '', 'None')]
            item[metric+'_mean'] = float(np.mean(values)) if values else None
            item[metric+'_std'] = float(np.std(values, ddof=1)) if len(values)>1 else (0 if values else None)
        summary.append(item)
    if summary:
        write_rows(output / 'time_varying_delay_summary.csv', summary)
    print(f'Results: {output}', flush=True)


if __name__ == '__main__':
    run(parse_args())
