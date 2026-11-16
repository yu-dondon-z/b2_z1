"""CPU shell forwarding tests; never starts Isaac or loads an actor."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUN = "2026-11-14_15-07-38_vd12_geometry_guard_long_20261114"


class FinalEntryTests(unittest.TestCase):
    def test_fixed_final_checkpoint_and_user_arguments_forwarded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            shutil.copyfile(ROOT / "scripts/play_vd12_final.sh", root / "scripts/play_vd12_final.sh")
            checkpoint = root / "logs/rsl_rl/b2_z1_vd12_geometry_guard_amp" / RUN / "model_5999.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"fixture")
            (root / "scripts/play_vd12_checked.sh").write_text('printf "%s\\n" "$@"\nexit 7\n')
            result = subprocess.run(["bash", str(root / "scripts/play_vd12_final.sh"),
                                     "--no-render", "--duration", "30"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 7)
            self.assertEqual(result.stdout.splitlines(), ["--checkpoint", str(checkpoint),
                                                           "--no-render", "--duration", "30"])

    def test_missing_final_checkpoint_fails_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            shutil.copyfile(ROOT / "scripts/play_vd12_final.sh", root / "scripts/play_vd12_final.sh")
            result = subprocess.run(["bash", str(root / "scripts/play_vd12_final.sh")],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Missing completed VD12 checkpoint", result.stderr)


if __name__ == "__main__":
    unittest.main()
