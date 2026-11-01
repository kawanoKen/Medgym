"""Verify that exporting trajectories preserves evaluation and action coordinates."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/online'))
import _eval_base as evaluation
from scripts.figures.plot_global_vs_perpatient import _load_agg


class PatientEnvironment:
    mean = np.arange(6, dtype=np.float32)
    std = np.ones(6, dtype=np.float32)
    action_scale = np.ones(3, dtype=np.float32)
    action_min = np.zeros(3, dtype=np.float32)
    action_max = np.ones(3, dtype=np.float32)
    use_dt = False
    smdp_state_dim = 8
    option_action_dim = 3
    dt_min = 0.5
    dt_max = 96.0
    total_time_h = 96.0

    def reset(self, patient=None, init_state_norm=None):
        self.phys_state = np.asarray(init_state_norm).copy()
        self.current_t = 0.0
        self.steps = 0
        return np.r_[self.phys_state, 1.0, 1.0].astype(np.float32)

    def step(self, action):
        self.current_t = 2.0 if self.steps == 0 else 96.0
        self.steps += 1
        self.phys_state += 1.0
        obs = np.r_[self.phys_state, 0.0, 0.0].astype(np.float32)
        return obs, 0.0, self.steps == 2, {'sofa': float(self.steps), 'lactate': 1.0}


class Agent:
    def select_action(self, state, deterministic=True):
        return np.full(3, -0.5, dtype=np.float32)


class RolloutExportTest(unittest.TestCase):
    def collect(self, store):
        patient = PatientEnvironment()
        env = evaluation.AdaptedEnv(patient, patient.mean, patient.std,
                                    patient.action_scale, patient.mean, patient.std,
                                    2 * patient.action_scale)
        return evaluation._collect_policy(
            Agent(), env, 2, np.zeros(6, dtype=np.float32), 0.0,
            np.random.default_rng(42), trajectory_store=store,
        )

    def test_export_preserves_metrics_and_records_applied_physical_action(self):
        store = {}
        original = self.collect(None)
        exported = self.collect(store)
        for index in (0, 1, 2, 3, 5, 6):
            np.testing.assert_array_equal(original[index], exported[index])
        self.assertEqual(original[4], exported[4])
        np.testing.assert_array_equal(store['episode_offsets'], [0, 2, 4])
        np.testing.assert_array_equal(store['step_t_start'], [0, 2, 0, 2])
        np.testing.assert_array_equal(store['step_t_end'], [2, 96, 2, 96])
        np.testing.assert_allclose(store['step_policy_action_tanh'], -0.5)
        np.testing.assert_allclose(store['step_applied_action_norm'], 0.5)
        np.testing.assert_allclose(store['step_applied_action_phys'], np.expm1(0.25 * 2))

    def test_plot_loader_holds_end_state_without_stretching_long_interval(self):
        import tempfile
        store = {}
        result = self.collect(store)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'rollout.npz'
            np.savez(path, **store, t_grid=evaluation.T_GRID,
                     sofa_mean=result[0], sofa_std=result[1],
                     lac_mean=result[2], lac_std=result[3])
            data = _load_agg(path)
        # Initial sample at t=0; first endpoint at t=2; second only at t=96.
        np.testing.assert_allclose(data['spo2_mean'][[0, 1, 2, 50, 95, 96]],
                                   [0, 0, 1, 1, 1, 2])
        np.testing.assert_allclose(data['vaso_mean'], 0.5)
        np.testing.assert_allclose(data['sofa_mean'], result[0])


if __name__ == '__main__':
    unittest.main()
