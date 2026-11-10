import importlib.util
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "vd5_geometry", ROOT / "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/mdp/commands/vd5_geometry.py"
)
vd5 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vd5)


class VD5GeometryTests(unittest.TestCase):
    def test_samples_requested_distance_behind_on_straight_path(self):
        path = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
        point = vd5.sample_polyline_behind_position(path, torch.tensor([[1.4, 0.2, 0.0]]), 0.5)
        torch.testing.assert_close(point, torch.tensor([[0.9, 0.0, 0.0]]))

    def test_startup_extrapolates_backwards_from_first_tangent(self):
        path = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])
        point = vd5.sample_polyline_behind_position(path, torch.tensor([[0.1, 0.0, 0.0]]), 0.5)
        torch.testing.assert_close(point, torch.tensor([[-0.4, 0.0, 0.0]]))

    def test_samples_across_corner_by_arc_length(self):
        path = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])
        point = vd5.sample_polyline_behind_position(path, torch.tensor([[1.0, 0.7, 0.0]]), 0.5)
        torch.testing.assert_close(point, torch.tensor([[1.0, 0.2, 0.0]]))

    def test_world_position_conversion_tracks_translated_and_yawed_base(self):
        angle = torch.tensor(torch.pi / 2)
        quat = torch.tensor([[torch.cos(angle / 2), 0.0, 0.0, torch.sin(angle / 2)]])
        base = torch.tensor([[2.0, 3.0, 0.5]])
        world = torch.tensor([[2.0, 4.0, 1.5]])
        torch.testing.assert_close(
            vd5.world_position_to_base(world, base, quat), torch.tensor([[1.0, 0.0, 1.0]])
        )

    def test_rejects_invalid_distance_and_path(self):
        path = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            vd5.sample_polyline_behind_position(path, torch.zeros(1, 3), -1.0)
        with self.assertRaisesRegex(ValueError, "at least two"):
            vd5.sample_polyline_behind_position(path[:, :1], torch.zeros(1, 3), 0.1)


if __name__ == "__main__":
    unittest.main()
