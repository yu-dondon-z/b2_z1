"""Pure CPU checks of probe status; no Isaac app is launched."""

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/diagnostics/probe_vd12_guard.py"
SPEC = importlib.util.spec_from_file_location("vd12_probe_status", SCRIPT)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class ProbeResultTests(unittest.TestCase):
    def test_timely_termination_for_every_affected_environment(self):
        bad = {"physics_step": 234, "affected_env_ids": [0, 1]}
        term = {"physics_step": 236, "env_ids": [0, 1, 2]}
        self.assertEqual(PROBE.classify_enforcement(bad, term, 4, 1),
                         ("guard_enforcement_verified", True, 0))

    def test_observed_violation_without_termination_fails_at_either_blend(self):
        bad = {"physics_step": 234, "affected_env_ids": [0]}
        for blend in (0, 1):
            self.assertEqual(PROBE.classify_enforcement(bad, None, 4, blend),
                             ("guard_enforcement_failed", False, 2))

    def test_late_early_or_partial_termination_cannot_pass(self):
        bad = {"physics_step": 234, "affected_env_ids": [0, 1]}
        for term in ({"physics_step": 238, "env_ids": [0, 1]},
                     {"physics_step": 233, "env_ids": [0, 1]},
                     {"physics_step": 236, "env_ids": [0]}):
            with self.subTest(termination=term):
                self.assertEqual(PROBE.classify_enforcement(bad, term, 4, 0),
                                 ("guard_enforcement_failed", False, 2))

    def test_unexplained_termination_fails(self):
        self.assertEqual(PROBE.classify_enforcement(None, {"physics_step": 4, "env_ids": [0]}, 4, 0),
                         ("guard_enforcement_failed", False, 2))

    def test_no_observation_is_not_enforcement_evidence(self):
        self.assertEqual(PROBE.classify_enforcement(None, None, 4, 0),
                         ("no_exclusion_observed_in_short_probe", False, 0))
        self.assertEqual(PROBE.classify_enforcement(None, None, 4, 1),
                         ("no_exclusion_observed_in_short_probe", False, 2))


if __name__ == "__main__":
    unittest.main()
