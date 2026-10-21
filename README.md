# Fault Diagnosis Reproduction

This repository contains a selected, reproducible numerical simulation implementation for fault detection and diagnosis in a nonlinear flotation process. The simulation models B-spline probability density functions, delayed nonlinear dynamics, fault detection, fault diagnosis, and detection-delay evaluation.

## Contents

- `num_simulation/simulation_data_generator.py`: synthetic time-series and PDF observation generation.
- `num_simulation/models.py`: B-spline PDF, state dynamics, PINN model, and detection/diagnosis helpers.
- `num_simulation/simulation_experiment.py`: simulation training and evaluation workflow.
- `num_simulation/delay_evaluation.py`: threshold calibration and delay metrics.
- `num_simulation/*delay*experiment.py`: fixed and time-varying delay experiments.
- `num_simulation/test_*.py`: focused checks for delay evaluation.

Generated data, model checkpoints, and experiment outputs are intentionally excluded. Running experiments writes generated files locally.

## Setup

Use Python 3.10 or 3.11. Create an isolated environment, then install the dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run checks

```bash
python -m unittest discover -s num_simulation -p "test_*.py"
```

## Run a simulation

```bash
python num_simulation/simulation_experiment.py
```

The experiment scripts include command-line options for their respective study settings. Run a script with `--help` where available. Plot labels use Chinese fonts when available and fall back to a standard sans-serif font.

## Scope and data

This is a selected code release, not a complete release of the original working directory. It contains no industrial or video data, derived datasets, trained weights, paper drafts, reviewer correspondence, or generated result files. All simulation data can be generated from the included source code.
