# Physics-Informed Neural Network Learning of Time-Delay Froth Bubble Size Distribution Dynamics for Reagent-Dosage Fault Detection and Diagnosis in Mineral Flotation

This repository accompanies the paper **Physics-Informed Neural Network Learning of Time-Delay Froth Bubble Size Distribution Dynamics for Reagent-Dosage Fault Detection and Diagnosis in Mineral Flotation**.

## Background and motivation

Mineral flotation is a core process for efficient separation of mineral resources. It is a complex physicochemical process involving gas--liquid--solid three-phase interactions, bubble--particle collision and attachment, froth drainage, and froth stability evolution. In industrial operation, flotation systems exhibit pronounced time-delay effects, such as delayed reagent action and froth transport lag, as well as strong nonlinearities caused by reagent-dependent bubble-size evolution and multiphase-flow coupling.

Threshold monitoring of pulp level or aeration can miss reagent faults expressed through BSD peak shifts, asymmetry, and multimodality. Fixed thresholds are also vulnerable to time-varying flotation data. Distribution-aware monitoring with temporal and mechanistic information is therefore needed.

## Proposed method

Motivated by these challenges, this paper proposes a physics-informed neural-network-driven fault detection and diagnosis framework for mineral flotation process monitoring, termed PINN-FDDMF. First, B-spline basis functions are used to construct a compact parametric representation of the BSD PDF, transforming distribution evolution into the trajectory modeling of a time-varying weight vector. Then, a nonlinear time-delay dynamic model with delayed states and delayed inputs is developed in the weight-vector space.

Based on this model, a multi-objective PINN is designed by combining data fitting, time-delay physics residuals, and smoothness regularization. Finally, residual statistics are used for fault detection, and PDF morphological features are used to identify typical reagent faults, including reagent overdosage and underdosage.

## Modeling result

B-spline coefficients provide a compact representation of the time-varying BSD PDF, and their evolution is described by a nonlinear state-space model with state and input delays.

![Modeling accuracy comparison](figures/modeling_accuracy_comparison.png)

## Detection result

Prediction residuals are used to detect abnormal deviations, and adaptive residual evaluation accommodates residual distribution drift.

![Detection confusion matrix for the first fault case](figures/detection_confusion_matrix1.png)

![Detection confusion matrix for the second fault case](figures/detection_confusion_matrix2.png)

## Diagnosis result

PDF morphology provides interpretable information for distinguishing reagent overdosage and underdosage faults.

![Comparison of diagnosis methods](figures/diagnosis_comparison.png)

## Experimental conclusion

The evaluation includes numerical simulations and industrial copper flotation experiments. On independent industrial copper-flotation test sequences, PINN-FDDMF obtains the highest F1-scores for overdosage (0.990) and underdosage (0.980); PIDMD obtains the highest underdosage accuracy (0.997). The investigated fault scope is limited to reagent overdosage and underdosage under the studied conditions.

![Industrial diagnostic summary](figures/practical_matrix_combined.png)

## Code and data availability

The code is not included in this public repository. Researchers who need to inspect or request the implementation may contact **424087031@qq.com**.

This repository contains selected paper figures and does not include raw industrial data, video data, or trained model weights.
