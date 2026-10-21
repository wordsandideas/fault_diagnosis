import unittest
import numpy as np

from simulation_data_generator import SimulationDataGenerator
from time_varying_delay_experiment import make_delays, parse_args


class TimeVaryingDelayTests(unittest.TestCase):
    def test_recurrence_uses_source_time(self):
        delays = np.array([0, 1, 0, 2, 1, 3, 0, 1])
        generator = SimulationDataGenerator(T=8, noise_level=0,
                                            d_true_seq=delays, b_true_seq=delays)
        coefficients = np.zeros((8, 6))
        coefficients[:, 1] = 0.1
        coefficients[:, 3] = 0.2
        u = np.arange(8).reshape(-1, 1)/10
        zeros = np.zeros_like(u)
        actual = generator.generate_state_sequence(coefficients, u, zeros, zeros,
                                                     initial_state=0.2)
        expected = np.zeros_like(u)
        expected[0] = 0.2
        for t in range(7):
            source = t-delays[t]
            expected[t+1] = expected[t] + (0.1*expected[source]+0.2*u[source] if source >= 0 else 0)
        np.testing.assert_allclose(actual, expected)

    def test_invalid_sequences(self):
        for values in ([1]*7, [1.5]*8, [-1]*8, [np.nan]*8, [8]*8):
            with self.assertRaises(ValueError):
                SimulationDataGenerator(T=8, d_true_seq=values)

    def test_scenarios_are_reproducible_and_bounded(self):
        args = parse_args([])
        for scenario in args.scenarios:
            first = make_delays(args, scenario, 11)
            second = make_delays(args, scenario, 11)
            for seq, repeat, base, amplitude in zip(first, second,
                    [args.d_true, args.b_true], [args.d_amplitude, args.b_amplitude]):
                np.testing.assert_array_equal(seq, repeat)
                self.assertTrue(np.all(seq[:args.change_start] == base))
                self.assertTrue(np.all((seq >= max(0, base-amplitude)) & (seq <= base+amplitude)))


if __name__ == '__main__':
    unittest.main()
