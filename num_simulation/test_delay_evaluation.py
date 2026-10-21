import unittest
import numpy as np
import torch
from simulation_data_generator import SimulationDataGenerator
from simulation_experiment import SimulationExperiment


class DelayEvaluationTests(unittest.TestCase):
    def test_generator_delays_refer_to_source_time(self):
        for delay in (0, 1, 2, 4):
            generator = SimulationDataGenerator(T=8, d=delay, b=delay, noise_level=0)
            coefficients = np.zeros((8, 6))
            coefficients[:, 3] = 0.1
            u = np.arange(8).reshape(-1, 1) / 10
            zeros = np.zeros_like(u)
            state = generator.generate_state_sequence(coefficients, u, zeros, zeros)
            expected = np.zeros_like(u)
            for t in range(1, 8):
                source = t - 1 - delay
                expected[t] = expected[t-1] + (0.1*u[source] if source >= 0 else 0)
            np.testing.assert_allclose(state, expected)
            coefficients[:, 3] = 0
            coefficients[:, 1] = 0.1
            state = generator.generate_state_sequence(coefficients, zeros, zeros, zeros,
                                                       initial_state=0.2)
            expected[0] = 0.2
            for t in range(1, 8):
                source = t - 1 - delay
                expected[t] = expected[t-1] + (0.1*expected[source] if source >= 0 else 0)
            np.testing.assert_allclose(state, expected)

    def test_labels_and_unknown_fault_do_not_change_predictions(self):
        experiment = SimulationExperiment(T=40, n_bins=12)
        normal = experiment.generator.generate_data()
        experiment.calibrate_normal(*normal, window_size=3)
        a, u, g, fault, pdf = experiment.generator.generate_data(fault_type=1, fault_start=20)
        first = experiment.evaluate_model(a, u, g, fault, pdf,
                                          fault_type=1, fault_start=20, window_size=3)
        second = experiment.evaluate_model(a, u, g, fault*100, pdf,
                                           fault_type=2, fault_start=15,
                                           fault_profile='delayed_sigmoid', window_size=3)
        for key in ('pred_a_seq', 'pred_pdf_seq', 'fault_flags', 'predicted_labels', 'thresholds'):
            torch.testing.assert_close(first[key], second[key])
        self.assertEqual(first['score_start'], 6)
        self.assertEqual(first['evaluation_mask'].sum().item(), 33)
        expected = (first['pred_pdf_seq'][5:] - pdf[5:]).square().mean().sqrt().item()
        self.assertAlmostEqual(first['pdf_rmse'], expected)

    def test_common_warmup_and_calibration_required(self):
        masks = []
        data = SimulationDataGenerator(T=40, n_bins=12).generate_data()
        for d, b in ((0, 0), (2, 1), (1, 4), (4, 4)):
            experiment = SimulationExperiment(T=40, n_bins=12, d=d, b=b,
                                              d_true=2, b_true=1, evaluation_start=4)
            with self.assertRaises(ValueError):
                experiment.evaluate_model(*data, window_size=3)
            experiment.calibrate_normal(*data, window_size=3)
            result = experiment.evaluate_model(*data, window_size=3)
            masks.append(result['evaluation_mask'])
            self.assertTrue(torch.all(result['pred_a_seq'][:max(d, b)] == 0))
        for mask in masks[1:]:
            torch.testing.assert_close(mask, masks[0])
        with self.assertRaises(ValueError):
            SimulationExperiment(d=5, evaluation_start=4)


if __name__ == '__main__':
    unittest.main()
