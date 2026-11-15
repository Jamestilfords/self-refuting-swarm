"""Regression checks for resource separation and provisioning semantics."""

from dataclasses import asdict
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from modal_jobs.config import OPEN_MOUNT, SEALED_MOUNT, Settings
from modal_jobs.setup import configure, validate_report


def report(role, volume_id):
    return {
        "role": role,
        "mount": OPEN_MOUNT if role == "open" else SEALED_MOUNT,
        "volume_id": volume_id,
        "write_read_commit": "ok",
        "other_mount_absent": True,
    }


class SettingsTests(unittest.TestCase):
    def test_open_and_sealed_cannot_alias(self):
        with self.assertRaisesRegex(ValueError, "different names"):
            Settings(open_volume_name="same", sealed_volume_name="same")

    def test_only_named_configuration_is_loaded(self):
        settings = Settings.from_env({
            "SKEPTIC_SECRET_NAMES": "hf, openai",
            "MODAL_TOKEN_SECRET": "must-not-be-forwarded",
            "OPENAI_API_KEY": "must-not-be-forwarded",
        })
        self.assertEqual(settings.secret_names, ("hf", "openai"))
        self.assertNotIn("must-not-be-forwarded", str(asdict(settings)))
        self.assertNotIn("must-not-be-forwarded", str(settings.container_env()))

    def test_custom_configuration_survives_container_import(self):
        settings = Settings(
            app_name="experiment-2",
            environment="dev",
            open_volume_name="experiment-2-open",
            sealed_volume_name="experiment-2-sealed",
            secret_names=("hf", "openai"),
        )
        self.assertEqual(Settings.from_env(settings.container_env()), settings)

    def test_bad_names_are_rejected(self):
        for value in ("", "contains spaces", "../other", "x" * 65):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Settings(app_name=value)
        for value in ("hf,", "hf,hf"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Settings.from_env({"SKEPTIC_SECRET_NAMES": value})


class ReportTests(unittest.TestCase):
    def test_wrong_volume_or_exposed_mount_fails(self):
        for key, value in (
            ("volume_id", "vo-other"),
            ("mount", SEALED_MOUNT),
            ("other_mount_absent", False),
            ("write_read_commit", "failed"),
        ):
            response = report("open", "vo-open")
            response[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate_report(response, "open", "vo-open")


class ProvisioningTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        self.open = Mock(object_id="vo-open")
        self.sealed = Mock(object_id="vo-sealed")
        self.volumes = patch("modal_jobs.setup.modal.Volume.from_name").start()
        self.volumes.side_effect = [self.open, self.sealed]
        self.secret = patch("modal_jobs.setup.modal.Secret.from_name").start()
        self.lookup = patch("modal_jobs.setup.modal.App.lookup").start()
        self.lookup.return_value.app_id = "ap-test"
        self.lookup.return_value.get_dashboard_url.return_value = "https://modal.com/apps/test"
        self.functions = patch("modal_jobs.setup.modal.Function.from_name").start()
        self.functions.side_effect = [
            Mock(remote=Mock(return_value=report("open", "vo-open"))),
            Mock(remote=Mock(return_value=report("sealed", "vo-sealed"))),
        ]
        self.addCleanup(patch.stopall)

    def test_check_only_never_creates_or_deploys(self):
        with patch("modal_jobs.setup.importlib.import_module") as register:
            result = configure(self.settings, check_only=True)
        self.assertEqual(result["status"], "ok")
        register.assert_not_called()
        for call in self.volumes.call_args_list:
            self.assertFalse(call.kwargs["create_if_missing"])
            self.assertEqual(call.kwargs["environment_name"], "main")
        self.lookup.assert_called_once_with("skeptic", environment_name="main")

    def test_missing_named_secret_prevents_resource_creation(self):
        from modal.exception import NotFoundError

        self.secret.return_value.hydrate.side_effect = NotFoundError("missing")
        with self.assertRaises(NotFoundError):
            configure(Settings(secret_names=("missing",)))
        self.volumes.assert_not_called()
        self.lookup.assert_not_called()

    def test_same_remote_volume_is_rejected(self):
        self.sealed.object_id = "vo-open"
        with self.assertRaisesRegex(RuntimeError, "same Modal Volume"):
            configure(self.settings, check_only=True)
        self.lookup.assert_not_called()
        self.functions.assert_not_called()

    def test_provision_reuses_names_and_registers_job_modules(self):
        fake_app_module = Mock()
        fake_app_module.settings = self.settings
        with patch.dict("sys.modules", {"modal_jobs.app": fake_app_module}), \
             patch("modal_jobs.setup.importlib.import_module") as register:
            configure(self.settings)
        self.assertEqual([call.args[0] for call in self.volumes.call_args_list],
                         ["skeptic-open", "skeptic-sealed"])
        for call in self.volumes.call_args_list:
            self.assertTrue(call.kwargs["create_if_missing"])
            self.assertEqual(call.kwargs["version"], 2)
        fake_app_module.app.deploy.assert_called_once_with(environment_name="main")
        self.assertEqual({call.args[0] for call in register.call_args_list}, {
            "modal_jobs.train40", "modal_jobs.seal", "modal_jobs.sandbox",
            "modal_jobs.cache_llama", "modal_jobs.score_guard",
        })


class MountProbeTests(unittest.TestCase):
    def test_probe_cleans_only_its_marker(self):
        from modal_jobs.app import _check_volume

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "open"
            root.mkdir()
            existing = root / "unrelated.txt"
            existing.write_text("keep")
            volume = Mock(object_id="vo-open")
            response = _check_volume("open", str(root), str(Path(directory) / "sealed"), volume)
            self.assertEqual(response["write_read_commit"], "ok")
            self.assertEqual(existing.read_text(), "keep")
            self.assertEqual(list((root / ".skeptic-setup").iterdir()), [])
            self.assertEqual(volume.commit.call_count, 2)
            volume.reload.assert_called_once()

    def test_probe_rejects_other_mount_before_writing(self):
        from modal_jobs.app import _check_volume

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "open"
            forbidden = Path(directory) / "sealed"
            root.mkdir()
            forbidden.mkdir()
            volume = Mock()
            with self.assertRaisesRegex(RuntimeError, "other volume"):
                _check_volume("open", str(root), str(forbidden), volume)
            volume.commit.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
