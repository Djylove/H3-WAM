import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/h3wam/build_libero_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_libero_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class H3ManifestTest(unittest.TestCase):
    def test_evenly_spaced_starts_include_endpoints(self):
        self.assertEqual(MODULE.evenly_spaced_starts(100, 5), [0, 25, 50, 75, 100])

    def test_short_episode_has_no_window(self):
        self.assertEqual(MODULE.evenly_spaced_starts(-1, 5), [])

    def test_zero_range_is_not_duplicated(self):
        self.assertEqual(MODULE.evenly_spaced_starts(0, 5), [0])


if __name__ == "__main__":
    unittest.main()
