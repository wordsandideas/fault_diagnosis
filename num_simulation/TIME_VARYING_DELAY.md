# Time-varying true delay study

Run from num_simulation:

```powershell
python time_varying_delay_experiment.py
```

Defaults: 5 seeds, model delays (2,1), 150 epochs, T=200, bins=50.
Training and independent normal calibration use constant true delays (2,1).
The trained model and calibration threshold stay frozen for all test cases.
Five scenarios (constant, step, periodic, random blocks, drift) each include
a normal test and both fault types under step and delayed_sigmoid profiles.
This requires 5 training runs and 125 evaluations, not 125 training runs.

True delays change from source time 60; faults are triggered at time 110.
State delay amplitude is 1 and input delay amplitude is 1. Period is 40 and
random block length is 20. Drift rises monotonically toward the upper delay.
Delay sequences are integer steps; no interpolation or adaptive model is used.
The source-time convention is a[t+1] using a[t-d_true_seq[t]] and
u[t-b_true_seq[t]]. Negative history uses zero. Constant interfaces still work.

```python
generator = SimulationDataGenerator(T=200, d=2, b=1,
    d_true_seq=d_sequence, b_true_seq=b_sequence)
```

Sequences must each have length T and contain finite nonnegative integers < T.
They are used only by the generator, never passed to model inference.

Outputs in results/time_varying_delay_results:
- time_varying_delay_trials.csv: per-seed metrics and temporal segment metrics.
- time_varying_delay_summary.csv: mean and sample SD across seeds.
- *.npz: source-time delays, target times, labels, flags, residuals, prediction
  errors and common evaluation masks for plotting and pooled diagnosis metrics.
- run_config.json: strict configuration check for resume.

Repeat the same command to resume. Changed settings require a new --output-dir.
A partially completed model-delay group is retrained deterministically.
Random forcing is paired across delay scenarios. Segment boundaries account
for the one-step shift between source time and predicted target time.
The transition segment is a predefined analysis window, not a measured recovery
time. For periodic/random/drift cases, later segments still contain changing delays.

Primary comparisons: normal-test FAR, faulty-test detection F1/FDR/delay and
state RMSE relative to the paired constant-delay control. Normal-test F1 is zero
by convention because no positives exist; it is not an effectiveness metric.
Gradual fault labels start at first positive activation (trigger + delay + 1).
Per-trajectory Macro-F1 still uses all three labels and PDF error is auxiliary;
check those existing evaluation limitations before manuscript reporting.

Quick check:

```powershell
python time_varying_delay_experiment.py --seeds 11 --epochs 1 --n-bins 12 --scenarios constant,step --output-dir results/time_varying_quick
```

One epoch checks execution only; use the full settings for scientific results.
