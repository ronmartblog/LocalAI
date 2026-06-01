import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src import resource_manager


class ResourceManagerStartupDetectionTests(unittest.TestCase):
    def test_get_free_disk_uses_existing_parent_for_missing_models_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_models_dir = Path(tmp) / "models"
            free_bytes = 64 * 1_073_741_824
            with patch(
                "src.resource_manager.shutil.disk_usage",
                return_value=SimpleNamespace(free=free_bytes),
            ) as disk_usage:
                free_gb = resource_manager.get_free_disk_gb(missing_models_dir)

        self.assertEqual(free_gb, 64)
        disk_usage.assert_called_once_with(str(Path(tmp)))

    def test_should_suggest_low_resources_ignores_unavailable_ram_probes(self):
        with patch("src.resource_manager.get_free_disk_gb", return_value=100.0), \
             patch("src.resource_manager.get_total_ram_gb", return_value=0.0), \
             patch("src.resource_manager.get_available_ram_gb", return_value=0.0):
            suggest, reason = resource_manager.should_suggest_low_resources("models")

        self.assertFalse(suggest)
        self.assertEqual(reason, "")

    def test_should_suggest_low_resources_still_reports_real_low_disk(self):
        with patch("src.resource_manager.get_free_disk_gb", return_value=12.0), \
             patch("src.resource_manager.get_total_ram_gb", return_value=32.0), \
             patch("src.resource_manager.get_available_ram_gb", return_value=16.0):
            suggest, reason = resource_manager.should_suggest_low_resources("models")

        self.assertTrue(suggest)
        self.assertIn("Low disk space: 12 GB free", reason)


if __name__ == "__main__":
    unittest.main()
