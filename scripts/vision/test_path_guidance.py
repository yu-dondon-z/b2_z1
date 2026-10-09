"""Run local guidance geometry checks without starting Isaac Sim."""
import importlib.util
import math
from pathlib import Path
import unittest

import torch

source = Path(__file__).resolve().parents[2] / "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/mdp/commands/path_guidance.py"
spec = importlib.util.spec_from_file_location("path_guidance", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PathGuidanceTests(unittest.TestCase):
    def test_offsets_and_headings(self):
        for slope in (0.0, -0.3, 0.3):
            for offset in (0.0, -0.2, 0.2):
                x = torch.linspace(0.5, 2.0, 20)
                points = torch.stack((x, slope * x + offset, torch.zeros_like(x)), dim=1)
                for ordered in (points, points.flip(0)):
                    tangent, lateral = module.local_path_geometry(ordered)
                    self.assertAlmostEqual(lateral.item(), offset / math.sqrt(1 + slope ** 2), places=5)
                    self.assertAlmostEqual(torch.atan2(tangent[1], tangent[0]).item(), math.atan(slope), places=5)

    def test_duplicates(self):
        points = torch.tensor([[0.5, 0.2], [0.5, 0.2], [1.0, 0.2]])
        _, lateral = module.local_path_geometry(points)
        self.assertAlmostEqual(lateral.item(), 0.2, places=5)
        with self.assertRaises(ValueError):
            module.local_path_geometry(torch.zeros(3, 2))

    def test_local_not_far_tangent(self):
        points = torch.tensor([[0.5, 0.0], [1.0, 0.0], [2.0, 1.0]])
        tangent, lateral = module.local_path_geometry(points)
        self.assertEqual(lateral.item(), 0.0)
        self.assertEqual(tangent[1].item(), 0.0)

    def test_cached_unordered_fit(self):
        x = torch.tensor([0.4, -0.2, 0.1, -0.4, 0.3, 2.0])
        points = torch.stack((x, 0.3 * x + 0.2), dim=1)
        tangent, lateral = module.cached_local_geometry(points)
        self.assertAlmostEqual(lateral.item(), 0.2 / math.sqrt(1.09), places=5)
        self.assertAlmostEqual(torch.atan2(tangent[1], tangent[0]).item(), math.atan(0.3), places=5)
        self.assertIsNone(module.cached_local_geometry(points[-1:]))


if __name__ == "__main__":
    unittest.main()
