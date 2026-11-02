"""CPU checks for the physical crack reward; no simulator startup needed."""

import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np
import torch


module_path = Path(__file__).resolve().parents[2] / (
    "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/mdp/commands/dual_vision_rewards.py")
spec = importlib.util.spec_from_file_location("dual_vision_rewards", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PhysicalCrackRewardTests(unittest.TestCase):
    def line(self):
        return module.physical_crack_polyline_xy(amplitude_m=.4, wavelength_m=20.,
            curve_half_length_m=20., road_half_length_m=95., mesh_step_m=.05,
            dtype=torch.float64)

    def test_polyline_matches_mesh_station_formula(self):
        curved_x = np.linspace(-20, 20, round(40 / .05) + 1)
        expected = np.vstack(([-95., 0.], np.column_stack((curved_x,
            .4 * np.sin(2 * np.pi * curved_x / 20)**3)), [95., 0.]))
        np.testing.assert_allclose(self.line().numpy(), expected, atol=2e-15)

    def test_zero_on_vertices_and_interpolated_mesh_segments(self):
        line = self.line()
        samples = torch.stack((line[271], .37 * line[401] + .63 * line[402], line[801]))
        points = torch.cat((samples, torch.tensor([[.0], [.6], [2.]], dtype=torch.float64)), dim=-1)[None]
        distance = module.physical_crack_lateral_distance_w(points, torch.zeros(1, 3), line)
        self.assertLess(distance.max().item(), 1e-12)

    def test_distance_is_nearest_segment_not_vertical_y_error(self):
        line = torch.tensor([[0., 0.], [1., 1.]], dtype=torch.float64)
        points = torch.tensor([[[.5, .6, .0]]], dtype=torch.float64)
        distance = module.physical_crack_lateral_distance_w(points, torch.zeros(1, 3), line)
        self.assertAlmostEqual(distance.item(), .1 / math.sqrt(2), places=12)

    def test_environment_translation_and_endpoint_clamping(self):
        line = torch.tensor([[0., 0.], [1., 0.]], dtype=torch.float64)
        origins = torch.tensor([[10., -4., 0], [-2., 7., 0]], dtype=torch.float64)
        points = torch.tensor([[[10.5, -3.95, .2], [11.2, -4., .1]],
                               [[-1.5, 7.05, .1], [-.8, 7., .7]]], dtype=torch.float64)
        distance = module.physical_crack_lateral_distance_w(points, origins, line)
        torch.testing.assert_close(distance, torch.tensor([[.05, .2], [.05, .2]], dtype=torch.float64))

    def test_flat_outer_sections_match_rendered_scene(self):
        points = torch.tensor([[[40., .07, .1], [-40., .09, .2]]], dtype=torch.float64)
        distance = module.physical_crack_lateral_distance_w(points, torch.zeros(1, 3), self.line())
        torch.testing.assert_close(distance, torch.tensor([[.07, .09]], dtype=torch.float64))

    def test_malformed_curve_is_rejected(self):
        with self.assertRaises(ValueError):
            module.physical_crack_polyline_xy(amplitude_m=.4, wavelength_m=0.,
                curve_half_length_m=20., road_half_length_m=95., mesh_step_m=.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
