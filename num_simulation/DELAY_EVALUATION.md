# Delay evaluation after alignment correction

The generator now updates a[t] using current time t-1 and delayed indices
t-1-d and t-1-b, consistent with the model's update a[t+1]. Prehistory
in the generator is zero. Model predictions begin at max(d, b); earlier
entries are placeholders, not copied target observations.

Set the same evaluation_start for every model (4 for a grid of 0 through 4).
Training state loss and state RMSE exclude residual indices before this
start. Detection scores are trailing window means, scored from
evaluation_start + window_size - 1. Residual index t predicts target t+1.
The common evaluation mask includes both normal and fault targets.

Before evaluate_model, call calibrate_normal on an independent, confirmed
normal sequence with the same k and window_size. Calibration uses the same
warmup and window statistic as detection. Thresholds are fixed after normal
calibration. All models must share the calibration and test sequences.

```python
experiment = SimulationExperiment(d=3, b=2, d_true=2, b_true=1,
                                  evaluation_start=4)
experiment.train_model(*training_data, epochs=100)
experiment.calibrate_normal(*normal_calibration_data, k=3, window_size=20)
result = experiment.evaluate_model(*test_data, k=3, window_size=20,
                                   true_labels=test_labels)
```

true_labels contains one class (0, 1, 2) per observation, including the
initial observation. For gradual faults, define and report the onset-label
convention explicitly. Without true_labels, fault_start/fault_type describe
a sustained fault for scoring only. fault_profile never controls inference.
Changing these labels cannot change predictions or detection thresholds.

Injected F_seq is only a simulator input/offline ground truth. Both training
and evaluation replace it with zero before passing inputs to the model.
Online diagnosis uses observed image PDFs and the existing general feature
scores, without target-type or profile adjustments. The PDF network
reconstructs the current PDF, so PDF errors use matching time indices.

State RMSE and end-to-end Macro-F1 are primary metrics. PDF reconstruction
RMSE is auxiliary because this branch does not directly receive d or b.
FAR/FDR are None when their denominator is absent. Missed trajectories have
detection_delay=None and undetected=True. Report undetected frequency
alongside delay among detected trajectories. Check onset_before_evaluation:
choose fault onset after the common score start to measure full delay.

The public evaluate_model uses the corrected path. The private
_legacy_evaluate_model and its historical diagnosis helpers are retained
for code comparison only and must not be used for reported experiments.
Legacy optimized thresholds are rejected by the public evaluator.

The full paired grid runner is `delay_sensitivity_experiment.py`. Its
default configuration uses true delays (2, 1), model delays d,b in 0..4,
five seeds, two fault types, and step/delayed-sigmoid profiles. Run it from
this directory with `python delay_sensitivity_experiment.py`. Results and
the fixed data cache are written under `results/sensitivity_results/`.
Use `--help` to customize the grid or reduce it for a smoke run. Reuse the
same output directory and cached data across all model delay combinations.
The runner trains once per seed and `(d,b)` pair (125 trainings by default),
then evaluates the four fault/profile cases. Each completed case is appended
to the trial CSV and flushed. Restarting with the exact same configuration
skips completed rows; changing settings requires a new output directory.

run_experiment uses aligned_v2 caches grouped by true delays, length and PDF
bins, and saves an independent normal calibration sequence. It retains
the existing mixed training strategy; this change does not establish that
mixed training is preferable to normal-only training. For a controlled
sensitivity study, explicitly save and reuse all datasets and random seeds;
run_experiment alone is not a complete sensitivity-grid runner.

Regenerate data and retrain after this correction. Old model/results are
not comparable: unknown fault inputs, oracle rules and evaluation intervals
have changed. Do not overwrite old results when collecting revised runs.
