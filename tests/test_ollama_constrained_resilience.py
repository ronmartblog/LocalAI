# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""
Contract tests for the constrained-cloud-VM Ollama resilience work.
Covers:

* :mod:`src.constrained_env` detection, disk-pressure probe, daemon-error
  mapping, and pre-pull skip behaviour.
* :class:`src.batch_runner.BatchRunner._run_ollama` reclassifies
  disk-full failures as ``environment_skip`` and does NOT bump the
  ``_consecutive_failures`` counter so a benchmark can proceed past the
  first 5 failing rows and still try every smaller model.
* :func:`src.ollama_client.OllamaClient.pull_model` retry/cancellation
  invariants are preserved (no regression of the existing happy-path /
  transient-network-error retry budget).
* The all-skipped banner fires when every Ollama row in the run was
  environment-skipped, and stays silent on a normal run.
* The shipped helper batch file ``set_ollama_models_dir.bat`` is wired
  into the release payload.
"""

import io
import os
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src import constrained_env, ollama_client
from src.batch_runner import BatchRunner
from src.batch_report import RunResult
from src.ollama_client import OllamaClient, OllamaError


ROOT = Path(__file__).resolve().parents[1]


def _force_constrained(value: bool):
    """Context-manager wrapper that pins is_constrained_vm() result."""
    if value:
        return patch.dict(os.environ, {"LOCALAI_FORCE_CONSTRAINED": "1"})
    return patch.dict(os.environ, {"LOCALAI_FORCE_CONSTRAINED": "0"})


class ConstrainedVmDetectionTests(unittest.TestCase):
    def setUp(self):
        constrained_env._reset_cache_for_tests()

    def tearDown(self):
        constrained_env._reset_cache_for_tests()

    def test_forced_on_via_env_var_returns_true(self):
        with _force_constrained(True):
            self.assertTrue(constrained_env.is_constrained_vm(force_refresh=True))

    def test_forced_off_via_env_var_returns_false(self):
        with _force_constrained(False):
            self.assertFalse(constrained_env.is_constrained_vm(force_refresh=True))

    def test_vanilla_machine_without_signals_returns_false(self):
        """A regular workstation must NOT be flagged as constrained.  The
        detection requires BOTH the profile-container marker AND the
        managed-VM marker; without both, the answer is False."""
        with patch.dict(os.environ, {"LOCALAI_FORCE_CONSTRAINED": ""}, clear=False), \
             patch("src.constrained_env._looks_like_managed_vm", return_value=False), \
             patch("src.constrained_env._looks_like_user_profile_container_present", return_value=False):
            self.assertFalse(constrained_env.is_constrained_vm(force_refresh=True))

    def test_profile_container_only_without_managed_vm_returns_false(self):
        """Some company laptops have a roaming profile container without
        being managed VMs.  We must not light up the OLLAMA_MODELS hint on
        those — they don't have the container size-cap problem."""
        with patch.dict(os.environ, {"LOCALAI_FORCE_CONSTRAINED": ""}, clear=False), \
             patch("src.constrained_env._looks_like_managed_vm", return_value=False), \
             patch("src.constrained_env._looks_like_user_profile_container_present", return_value=True):
            self.assertFalse(constrained_env.is_constrained_vm(force_refresh=True))

    def test_managed_vm_only_without_profile_container_returns_false(self):
        with patch.dict(os.environ, {"LOCALAI_FORCE_CONSTRAINED": ""}, clear=False), \
             patch("src.constrained_env._looks_like_managed_vm", return_value=True), \
             patch("src.constrained_env._looks_like_user_profile_container_present", return_value=False):
            self.assertFalse(constrained_env.is_constrained_vm(force_refresh=True))

    def test_synthetic_constrained_signals_return_true(self):
        with patch.dict(os.environ, {"LOCALAI_FORCE_CONSTRAINED": ""}, clear=False), \
             patch("src.constrained_env._looks_like_managed_vm", return_value=True), \
             patch("src.constrained_env._looks_like_user_profile_container_present", return_value=True):
            self.assertTrue(constrained_env.is_constrained_vm(force_refresh=True))

    def test_cpc_computer_name_pattern_is_managed_vm_signal(self):
        """CPC-<user>-<id> is a common managed-VM host naming convention."""
        with patch.dict(os.environ, {"COMPUTERNAME": "CPC-host01-4SOV0"}, clear=False):
            self.assertTrue(constrained_env._looks_like_managed_vm())

    def test_detection_is_cached_across_calls(self):
        """Performance reviewer: the warm path must not pay the detection
        cost twice."""
        with patch.dict(os.environ, {"LOCALAI_FORCE_CONSTRAINED": "1"}):
            constrained_env.is_constrained_vm(force_refresh=True)
            with patch("src.constrained_env._looks_like_managed_vm") as mocked:
                # Without force_refresh, cached result is used — mock must NOT fire.
                constrained_env.is_constrained_vm()
                self.assertEqual(mocked.call_count, 0)


class DiskFullErrorClassificationTests(unittest.TestCase):
    def setUp(self):
        constrained_env._reset_cache_for_tests()

    def tearDown(self):
        constrained_env._reset_cache_for_tests()

    def test_no_space_left_string_is_disk_full(self):
        self.assertTrue(constrained_env.is_disk_full_error_text(
            "write /home/user/.ollama/blobs/sha256-xxx: no space left on device"
        ))

    def test_errno_28_is_disk_full(self):
        self.assertTrue(constrained_env.is_disk_full_error_text(
            "OSError: [Errno 28] No space left on device"
        ))

    def test_permission_denied_is_disk_full_signal(self):
        """Roaming profile containers sometimes surface as 'permission denied'
        when the container is full and the filter driver short-circuits writes."""
        self.assertTrue(constrained_env.is_disk_full_error_text("permission denied"))

    def test_model_not_found_is_not_disk_full(self):
        self.assertFalse(constrained_env.is_disk_full_error_text(
            "pull manifest: model 'fakemodel' not found in registry"
        ))

    def test_empty_string_is_not_disk_full(self):
        self.assertFalse(constrained_env.is_disk_full_error_text(""))


class QuotaAwareOllamaErrorTests(unittest.TestCase):
    def setUp(self):
        constrained_env._reset_cache_for_tests()

    def tearDown(self):
        constrained_env._reset_cache_for_tests()

    def test_disk_full_on_constrained_vm_prepends_hint(self):
        with _force_constrained(True):
            constrained_env._reset_cache_for_tests()
            msg = constrained_env.quota_aware_ollama_error(
                "write blobs: no space left on device"
            )
        # Post-v5.3.7: the helper-batch reference replaces the old
        # hardcoded D:\OllamaModels example — the bat file is now the
        # canonical "how do I move this off my profile drive" entry
        # point and its DEFAULT_TARGET is app-path-relative.
        self.assertIn("set_ollama_models_dir.bat", msg)
        self.assertNotIn("D:\\OllamaModels", msg)
        self.assertIn("no space left", msg)

    def test_disk_full_off_constrained_vm_passes_through(self):
        """A regular workstation must NOT see the constrained-VM hint."""
        with _force_constrained(False):
            constrained_env._reset_cache_for_tests()
            msg = constrained_env.quota_aware_ollama_error(
                "write blobs: no space left on device"
            )
        self.assertNotIn("set_ollama_models_dir.bat", msg)
        self.assertIn("no space left", msg)

    def test_non_disk_error_on_constrained_vm_passes_through(self):
        """A 404 on a constrained VM must NOT get the disk hint."""
        with _force_constrained(True):
            constrained_env._reset_cache_for_tests()
            msg = constrained_env.quota_aware_ollama_error(
                "Model 'totally-fake-tag' not found in the Ollama registry."
            )
        self.assertNotIn("set_ollama_models_dir.bat", msg)
        self.assertIn("not found", msg)

    def test_wrapped_message_is_single_line(self):
        with _force_constrained(True):
            constrained_env._reset_cache_for_tests()
            msg = constrained_env.quota_aware_ollama_error(
                "errno 28\nno space left on device\nblob write failed"
            )
        self.assertNotIn("\n", msg)

    def test_wrapped_message_is_capped(self):
        with _force_constrained(True):
            constrained_env._reset_cache_for_tests()
            long = "errno 28 " + ("x" * 1000)
            msg = constrained_env.quota_aware_ollama_error(long)
        self.assertLessEqual(len(msg), 400)


class PrePullDiskCheckTests(unittest.TestCase):
    def setUp(self):
        constrained_env._reset_cache_for_tests()

    def tearDown(self):
        constrained_env._reset_cache_for_tests()

    def test_room_to_spare_returns_none(self):
        """Healthy machine: a 5 GB model with 100 GB free must NOT be
        blocked (returning None is the happy-path noop)."""
        with patch("src.constrained_env.get_ollama_models_dir_free_gb", return_value=100.0):
            self.assertIsNone(constrained_env.precheck_ollama_pull(5.0))

    def test_too_tight_returns_skip_reason(self):
        with _force_constrained(True), \
             patch("src.constrained_env.get_ollama_models_dir_free_gb", return_value=1.0):
            constrained_env._reset_cache_for_tests()
            reason = constrained_env.precheck_ollama_pull(5.0)
        self.assertIsNotNone(reason)
        self.assertIn("5.0 GB", reason)
        self.assertIn("1.0 GB", reason)
        # Post-v5.3.7: helper-bat reference replaces the old
        # hardcoded OLLAMA_MODELS=D:\OllamaModels example.
        self.assertIn("set_ollama_models_dir.bat", reason)
        self.assertNotIn("D:\\OllamaModels", reason)

    def test_too_tight_off_constrained_vm_returns_generic_hint(self):
        with _force_constrained(False), \
             patch("src.constrained_env.get_ollama_models_dir_free_gb", return_value=1.0):
            constrained_env._reset_cache_for_tests()
            reason = constrained_env.precheck_ollama_pull(5.0)
        self.assertIsNotNone(reason)
        self.assertIn("Free up disk", reason)
        self.assertNotIn("set_ollama_models_dir.bat", reason)

    def test_unknown_free_space_does_not_block(self):
        """If we can't measure disk (probe failure, weird path), we must
        NOT block — that would regress the healthy-machine happy path
        when shutil.disk_usage fails for some unrelated reason."""
        with patch("src.constrained_env.get_ollama_models_dir_free_gb", return_value=0.0):
            self.assertIsNone(constrained_env.precheck_ollama_pull(5.0))

    def test_zero_size_does_not_block(self):
        """Models without a size estimate must not be blocked."""
        with patch("src.constrained_env.get_ollama_models_dir_free_gb", return_value=1.0):
            self.assertIsNone(constrained_env.precheck_ollama_pull(0))


class OllamaModelsDirTests(unittest.TestCase):
    def test_env_var_override_takes_priority(self):
        with patch.dict(os.environ, {"OLLAMA_MODELS": str(ROOT)}, clear=False):
            path = constrained_env.get_ollama_models_dir()
        self.assertEqual(Path(path), ROOT)

    def test_default_is_user_profile_dot_ollama_models(self):
        with patch.dict(os.environ, {"OLLAMA_MODELS": ""}, clear=False):
            path = constrained_env.get_ollama_models_dir()
        self.assertEqual(path, Path.home() / ".ollama" / "models")


class BatchRunnerOllamaPhaseTests(unittest.TestCase):
    """Verify the BatchRunner reclassifies disk failures as
    ``environment_skip`` and does not bump consecutive_failures."""

    def setUp(self):
        constrained_env._reset_cache_for_tests()

    def tearDown(self):
        constrained_env._reset_cache_for_tests()

    @staticmethod
    def _model() -> dict:
        return {
            "id": "small",
            "name": "Small",
            "ollama_tag": "small:latest",
            "size_gb": 1.0,
            "min_ram_gb": 0,
            "min_vram_gb": 0,
        }

    def test_ollama_disk_full_failure_uses_environment_skip_phase(self):
        class FakeOllama:
            def is_running(self):
                return True

            def pull_model(self, *_args, **_kwargs):
                raise OllamaError(
                    "write blobs/sha256-foo: no space left on device"
                )

        runner = BatchRunner()
        runner.ollama = FakeOllama()
        runner._ollama_attempt_count = 0

        with _force_constrained(True), \
             patch.object(runner, "_unload_running_ollama_models"), \
             patch.object(runner, "_is_ollama_tag_local", return_value=True), \
             patch("src.constrained_env.precheck_ollama_pull", return_value=None):
            constrained_env._reset_cache_for_tests()
            with redirect_stdout(io.StringIO()):
                result = runner._run_ollama(self._model(), "ollama_cpu", threading.Event())

        self.assertFalse(result.success)
        self.assertEqual(result.failure_phase, "environment_skip")
        self.assertIn("set_ollama_models_dir.bat", result.error)
        self.assertIn("no space left", result.error)

    def test_ollama_404_failure_keeps_download_failed_phase(self):
        """A model-not-found 404 must still be classified as
        download_failed — only true disk-pressure errors get the
        environment_skip downgrade."""
        class FakeOllama:
            def is_running(self):
                return True

            def pull_model(self, *_args, **_kwargs):
                raise OllamaError("Model 'no-such:tag' not found in the Ollama registry.")

        runner = BatchRunner()
        runner.ollama = FakeOllama()

        with patch.object(runner, "_unload_running_ollama_models"), \
             patch.object(runner, "_is_ollama_tag_local", return_value=True):
            with redirect_stdout(io.StringIO()):
                result = runner._run_ollama(self._model(), "ollama_cpu", threading.Event())

        self.assertFalse(result.success)
        self.assertEqual(result.failure_phase, "download_failed")

    def test_pre_pull_skip_uses_environment_skip_phase(self):
        """When the pre-pull disk check tells us the destination can't
        fit the model, we must skip with environment_skip BEFORE calling
        ollama.pull_model (otherwise we'd waste a multi-minute retry)."""
        pull_calls: list[tuple] = []

        class FakeOllama:
            def is_running(self):
                return True

            def pull_model(self, *args, **kwargs):
                pull_calls.append((args, kwargs))

        runner = BatchRunner()
        runner.ollama = FakeOllama()

        with patch.object(runner, "_unload_running_ollama_models"), \
             patch.object(runner, "_is_ollama_tag_local", return_value=False), \
             patch("src.constrained_env.precheck_ollama_pull",
                   return_value="needs ~5 GB, only 0.5 GB free"):
            with redirect_stdout(io.StringIO()):
                result = runner._run_ollama(self._model(), "ollama_cpu", threading.Event())

        self.assertFalse(result.success)
        self.assertEqual(result.failure_phase, "environment_skip")
        self.assertIn("Skipped", result.error)
        self.assertIn("0.5 GB", result.error)
        self.assertEqual(pull_calls, [],
                         "pre-pull skip must NOT call ollama.pull_model")

    def test_environment_skip_does_not_bump_consecutive_failures(self):
        """The bug: 5 environment_skip rows in a row must NOT trip the
        max_failures bail-out, so the run continues with smaller models."""
        models = [
            {"id": f"m{i}", "name": f"M{i}", "ollama_tag": f"m{i}:tag",
             "size_gb": 100.0, "min_ram_gb": 0, "min_vram_gb": 0}
            for i in range(6)
        ]
        # Final model is small and should still get a shot at running.
        models.append({
            "id": "small", "name": "Small", "ollama_tag": "small:tag",
            "size_gb": 0.5, "min_ram_gb": 0, "min_vram_gb": 0,
        })

        runner = BatchRunner(model_ids=[m["id"] for m in models], max_failures=5)
        # Inject our synthetic models directly so we don't depend on the
        # real catalog.  _select_models() reads from the catalog, so we
        # patch it to return our list verbatim.
        with patch.object(runner, "_select_models", return_value=models), \
             patch.object(runner, "_methods_for", side_effect=lambda m: ["ollama_cpu"]), \
             patch.object(runner, "_iter_samples_for",
                          side_effect=lambda m, method: [{"id": "", "title": "",
                                                          "source": "", "prompt": ""}]):
            # Stub out the actual run so each call returns environment_skip
            # for the big models and success for the small one.
            attempts = {"count": 0}

            def fake_run_one(model, method):
                attempts["count"] += 1
                runner._ollama_attempt_count += 1
                if model["size_gb"] >= 10:
                    return RunResult(
                        model_id=model["id"], model_name=model["name"],
                        method=method, success=False,
                        error="Skipped: needs ~100 GB",
                        failure_phase="environment_skip",
                    )
                return RunResult(
                    model_id=model["id"], model_name=model["name"],
                    method=method, success=True,
                    tokens_per_sec=10.0, ttft=0.1, total_time=1.0,
                )

            with patch.object(runner, "_run_one", side_effect=fake_run_one), \
                 patch.object(runner, "_cleanup_model"), \
                 patch.object(runner, "_release_after_run"), \
                 patch.object(runner, "_save_report"), \
                 patch.object(runner.report, "save_json", return_value=Path(".")), \
                 patch.object(runner.report, "save_html", return_value=Path(".")):
                with redirect_stdout(io.StringIO()):
                    runner.run()

        # All 7 rows must have been attempted: 6 environment_skip + 1 success.
        # If environment_skip bumped _consecutive_failures we'd bail at 5
        # and the small model would never get a shot.
        self.assertEqual(attempts["count"], 7)
        self.assertEqual(runner._environment_skip_count, 6)
        # consecutive_failures got reset on the final success.
        self.assertEqual(runner._consecutive_failures, 0)

    def test_all_skipped_emits_banner(self):
        models = [
            {"id": f"m{i}", "name": f"M{i}", "ollama_tag": f"m{i}:tag",
             "size_gb": 100.0, "min_ram_gb": 0, "min_vram_gb": 0}
            for i in range(3)
        ]
        runner = BatchRunner(model_ids=[m["id"] for m in models])

        def fake_run_one(model, method):
            runner._ollama_attempt_count += 1
            return RunResult(
                model_id=model["id"], model_name=model["name"],
                method=method, success=False,
                error="Skipped: needs ~100 GB",
                failure_phase="environment_skip",
            )

        with patch.object(runner, "_select_models", return_value=models), \
             patch.object(runner, "_methods_for", side_effect=lambda m: ["ollama_cpu"]), \
             patch.object(runner, "_iter_samples_for",
                          side_effect=lambda m, method: [{"id": "", "title": "",
                                                          "source": "", "prompt": ""}]), \
             patch.object(runner, "_run_one", side_effect=fake_run_one), \
             patch.object(runner, "_cleanup_model"), \
             patch.object(runner, "_release_after_run"), \
             patch.object(runner, "_save_report"), \
             patch.object(runner.report, "save_json", return_value=Path(".")), \
             patch.object(runner.report, "save_html", return_value=Path(".")):
            buf = io.StringIO()
            with redirect_stdout(buf):
                runner.run()

        output = buf.getvalue()
        self.assertIn("All Ollama models were skipped", output)

    def test_normal_run_does_not_emit_banner(self):
        """When even ONE Ollama row succeeded, the all-skipped banner
        must NOT fire."""
        models = [
            {"id": "ok", "name": "Ok", "ollama_tag": "ok:tag",
             "size_gb": 1.0, "min_ram_gb": 0, "min_vram_gb": 0},
        ]
        runner = BatchRunner(model_ids=["ok"])

        def fake_run_one(model, method):
            runner._ollama_attempt_count += 1
            return RunResult(
                model_id=model["id"], model_name=model["name"],
                method=method, success=True,
                tokens_per_sec=10.0, ttft=0.1, total_time=1.0,
            )

        with patch.object(runner, "_select_models", return_value=models), \
             patch.object(runner, "_methods_for", side_effect=lambda m: ["ollama_cpu"]), \
             patch.object(runner, "_iter_samples_for",
                          side_effect=lambda m, method: [{"id": "", "title": "",
                                                          "source": "", "prompt": ""}]), \
             patch.object(runner, "_run_one", side_effect=fake_run_one), \
             patch.object(runner, "_cleanup_model"), \
             patch.object(runner, "_release_after_run"), \
             patch.object(runner, "_save_report"), \
             patch.object(runner.report, "save_json", return_value=Path(".")), \
             patch.object(runner.report, "save_html", return_value=Path(".")):
            buf = io.StringIO()
            with redirect_stdout(buf):
                runner.run()

        self.assertNotIn("All Ollama models were skipped", buf.getvalue())


class OllamaPullCancellationTests(unittest.TestCase):
    """Pin the existing happy-path retry behaviour so the constrained-VM work
    doesn't regress it."""

    def test_pull_model_honors_stop_event_mid_stream(self):
        """A stop_event set after the first chunk is read must abort the
        pull before the second chunk fires."""
        client = OllamaClient("http://test")
        stop_event = threading.Event()
        progress_calls: list[str] = []

        class FakeResponse:
            status_code = 200
            text = ""

            def raise_for_status(self):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def iter_lines(self):
                yield b'{"status": "pulling 1/2", "completed": 0, "total": 100}'
                # After yielding the first line we set the stop event.
                stop_event.set()
                yield b'{"status": "pulling 2/2", "completed": 50, "total": 100}'

        def fake_post(*_args, **_kwargs):
            return FakeResponse()

        def progress_cb(status, completed, total):
            progress_calls.append(status)

        with patch.object(ollama_client.requests, "post", side_effect=fake_post):
            client.pull_model("demo:tag", progress_cb=progress_cb, stop_event=stop_event)

        # The first chunk's progress callback fired; the second should NOT
        # have fired because stop_event was set between them.
        self.assertEqual(len(progress_calls), 1)
        self.assertIn("pulling 1/2", progress_calls[0])

    def test_pull_model_retries_on_transient_network_error(self):
        """The pre-existing exponential-backoff retry on
        ``requests.ConnectionError`` / ``requests.Timeout`` must still
        work — the constrained-VM fix only adds error CLASSIFICATION after
        retries exhaust, it must not short-circuit the retry."""
        client = OllamaClient("http://test")
        attempts = {"count": 0}

        class FakeOkResponse:
            status_code = 200
            text = ""

            def raise_for_status(self):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def iter_lines(self):
                yield b'{"status": "success", "completed": 100, "total": 100}'

        def fake_post(*_args, **_kwargs):
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise ollama_client.requests.ConnectionError("socket reset")
            return FakeOkResponse()

        # Sleep is mocked so the test doesn't actually wait 3+ seconds.
        with patch.object(ollama_client.requests, "post", side_effect=fake_post), \
             patch.object(ollama_client.time, "sleep"):
            client.pull_model("demo:tag", max_retries=3)

        self.assertEqual(attempts["count"], 3,
                         "ConnectionError must be retried up to max_retries")

    def test_pull_model_propagates_daemon_disk_full_error(self):
        """The daemon's JSON error stream is the canonical disk-full
        signal — it must be raised as OllamaError immediately (no retry)
        with the error text intact so BatchRunner can map it."""
        client = OllamaClient("http://test")

        class FakeResponse:
            status_code = 200
            text = ""

            def raise_for_status(self):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def iter_lines(self):
                yield b'{"status": "pulling", "completed": 0, "total": 100}'
                yield (b'{"error": "write '
                       b'/home/user/.ollama/blobs/sha256-xxx: '
                       b'no space left on device"}')

        with patch.object(ollama_client.requests, "post", return_value=FakeResponse()):
            with self.assertRaises(OllamaError) as ctx:
                client.pull_model("demo:tag")
        self.assertIn("no space left", str(ctx.exception))


class ShippedHelperBatchTests(unittest.TestCase):
    def test_set_ollama_models_dir_bat_exists(self):
        """The relocation helper must ship in the release zip — needed for
        constrained cloud VMs."""
        self.assertTrue((ROOT / "set_ollama_models_dir.bat").is_file())

    def test_set_ollama_models_dir_bat_uses_setx_user_scope(self):
        """The fix must persist across sessions (setx writes the
        registry); per-process ``set`` alone would evaporate when the
        cmd window closes."""
        text = (ROOT / "set_ollama_models_dir.bat").read_text(
            encoding="utf-8", errors="ignore"
        )
        self.assertIn("setx OLLAMA_MODELS", text)

    def test_set_ollama_models_dir_bat_defaults_to_app_path(self):
        """The shipped default must be ``%~dp0Ollama`` so the helper
        targets a folder next to LocalAI Studio (the app path) rather
        than a hardcoded drive letter that may not exist on the user's
        machine."""
        text = (ROOT / "set_ollama_models_dir.bat").read_text(
            encoding="utf-8", errors="ignore"
        )
        self.assertIn("%~dp0Ollama", text)

    def test_set_ollama_models_dir_bat_has_no_hardcoded_drives(self):
        """v5.3.8 DO NOT REGRESS: the shipped helper must not reference
        any specific drive letter or environment variable that bakes in
        an installation drive. Any of these tokens slipping back in
        means the helper has regressed to assuming a constrained-VM layout."""
        text = (ROOT / "set_ollama_models_dir.bat").read_text(
            encoding="utf-8", errors="ignore"
        )
        for forbidden in ("D:\\", "E:\\", "%HOMEDRIVE%", "%LOCALAPPDATA%"):
            self.assertNotIn(forbidden, text, f"helper still references {forbidden!r}")

    def test_set_ollama_models_dir_bat_explains_profile_container(self):
        """The helper's framing must explain *why* the relocation
        matters (profile container overflow on constrained cloud VMs)
        rather than just dumping a ``setx`` command on the user."""
        text = (ROOT / "set_ollama_models_dir.bat").read_text(
            encoding="utf-8", errors="ignore"
        )
        self.assertIn("profile container", text)
        self.assertIn("roaming profile", text)


if __name__ == "__main__":
    unittest.main()
