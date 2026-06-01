# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""Unit tests for ``src.migration.MigrationEngine`` and helpers.

These tests run headless (no Tk) so they're safe to execute on CI / from
plain ``py -3.12 -m pytest tests/``.  We always inject a fake copy_runner
to keep the engine off robocopy / rsync — the real subprocess wiring is
covered by the integration tests in ``test_setup_release_contracts``.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Optional

from src import migration as m


# ── Helpers ──────────────────────────────────────────────────────────────────


def _write_file(path: Path, content: bytes = b"hello") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _mirror_tree(source: Path, target: Path) -> None:
    """Copy source tree to target — used by the fake copy_runner."""
    if target.exists():
        shutil.rmtree(str(target))
    shutil.copytree(str(source), str(target))


def _fake_runner(engine: m.MigrationEngine) -> None:
    _mirror_tree(Path(engine.plan.source), Path(engine.plan.target))
    engine.state.bytes_done = engine.state.bytes_total
    engine.state.files_done = engine.state.files_total
    engine.emit_progress(m.ProgressEvent(
        files_done=engine.state.files_done,
        files_total=engine.state.files_total,
        bytes_done=engine.state.bytes_done,
        bytes_total=engine.state.bytes_total,
        percent=100.0,
    ), force=True)


def _cancel_runner_factory(engine_ref: list) -> "callable[[m.MigrationEngine], None]":
    """Return a copy_runner that triggers cancel after copying half the bytes."""
    def _runner(engine: m.MigrationEngine) -> None:
        engine_ref.append(engine)
        # Pretend we copied half the tree, then user cancels.
        target = Path(engine.plan.target)
        target.mkdir(parents=True, exist_ok=True)
        (target / "_partial").write_bytes(b"partial")
        engine.cancel()
        raise m.MigrationCancelled("user cancelled mid-copy")
    return _runner


# ── State file / lock / resume tests ─────────────────────────────────────────


class StateFileTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app_root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_write_then_read_roundtrips(self):
        s = m.MigrationState(kind="comfyui", source="src", target="dst",
                             phase=m.MigrationPhase.COPYING.value,
                             bytes_done=42, bytes_total=100, files_done=1, files_total=4)
        self.assertTrue(m.write_state(self.app_root, s))
        loaded = m.read_state(self.app_root)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.kind, "comfyui")
        self.assertEqual(loaded.phase, "copying")
        self.assertEqual(loaded.bytes_done, 42)

    def test_read_returns_none_for_missing_file(self):
        self.assertIsNone(m.read_state(self.app_root))

    def test_read_returns_none_for_corrupt_json(self):
        m.state_file_path(self.app_root).write_text("{not json", encoding="utf-8")
        self.assertIsNone(m.read_state(self.app_root))

    def test_find_resumable_state_skips_done_and_clears(self):
        s = m.MigrationState(kind="models", source="a", target="b",
                             phase=m.MigrationPhase.DONE.value)
        m.write_state(self.app_root, s)
        self.assertIsNone(m.find_resumable_state(self.app_root))
        self.assertFalse(m.state_file_path(self.app_root).exists(),
                         "DONE state should be cleared by find_resumable_state")

    def test_find_resumable_state_returns_in_flight_phase(self):
        s = m.MigrationState(kind="ollama", source="a", target="b",
                             phase=m.MigrationPhase.COPYING.value)
        m.write_state(self.app_root, s)
        got = m.find_resumable_state(self.app_root)
        self.assertIsNotNone(got)
        self.assertEqual(got.phase, "copying")

    def test_user_prompt_phases_covers_copying_and_verifying_only(self):
        # PENDING / PRE_FLIGHT must NOT trigger the user resume prompt.
        self.assertNotIn(m.MigrationPhase.PENDING, m.USER_PROMPT_RESUME_PHASES)
        self.assertNotIn(m.MigrationPhase.PRE_FLIGHT, m.USER_PROMPT_RESUME_PHASES)
        self.assertIn(m.MigrationPhase.COPYING, m.USER_PROMPT_RESUME_PHASES)
        self.assertIn(m.MigrationPhase.VERIFYING, m.USER_PROMPT_RESUME_PHASES)
        # COMMITTING / CLEANUP are idempotent — must NOT prompt.
        self.assertNotIn(m.MigrationPhase.COMMITTING, m.USER_PROMPT_RESUME_PHASES)
        self.assertNotIn(m.MigrationPhase.CLEANUP, m.USER_PROMPT_RESUME_PHASES)


# ── Robocopy parser tests ────────────────────────────────────────────────────


class RobocopyParserTests(unittest.TestCase):

    def test_parses_percent_and_size(self):
        result = m.parse_robocopy_line("    100%        12345    foo/bar/baz.bin")
        self.assertIsNotNone(result)
        pct, size, name = result
        self.assertEqual(int(pct), 100)
        self.assertEqual(size, 12345)
        self.assertEqual(name, "foo/bar/baz.bin")

    def test_returns_none_for_non_file_lines(self):
        # Banners, blank lines, robocopy header.
        for line in ("", "\n", "  Total    Copied   Skipped", " 0 Bytes", "----"):
            self.assertIsNone(m.parse_robocopy_line(line),
                              f"expected None for {line!r}")

    def test_normalise_digest_handles_colon_and_dash(self):
        a = m._normalise_digest("sha256:" + "a" * 64)
        b = m._normalise_digest("sha256-" + "a" * 64)
        self.assertEqual(a, b)
        self.assertEqual(a, "sha256-" + "a" * 64)

    def test_normalise_digest_returns_none_for_garbage(self):
        self.assertIsNone(m._normalise_digest("not a digest"))
        self.assertIsNone(m._normalise_digest(""))


# ── Sentinel tests ───────────────────────────────────────────────────────────


class SentinelTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_comfyui_sentinel_requires_main_py(self):
        self.assertFalse(m.has_comfyui_sentinel(self.root))
        _write_file(self.root / "main.py", b"# comfy")
        self.assertTrue(m.has_comfyui_sentinel(self.root))

    def test_ollama_sentinel_requires_blobs_subdir(self):
        self.assertFalse(m.has_ollama_sentinel(self.root))
        (self.root / "blobs").mkdir()
        self.assertTrue(m.has_ollama_sentinel(self.root))

    def test_models_sentinel_accepts_gguf_or_onnx_dir(self):
        self.assertFalse(m.has_models_sentinel(self.root))
        _write_file(self.root / "phi-4.gguf")
        self.assertTrue(m.has_models_sentinel(self.root))


# ── Ollama cross-validation tests ────────────────────────────────────────────


class OllamaCrossValidationTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def _make_manifest(self, name: str, digests: list[str]) -> None:
        mpath = self.root / "manifests" / "registry.ollama.ai" / "library" / name / "latest"
        mpath.parent.mkdir(parents=True, exist_ok=True)
        layers = [{"digest": f"sha256:{d}"} for d in digests]
        mpath.write_text(json.dumps({
            "config": {"digest": f"sha256:{digests[0]}"},
            "layers": layers,
        }), encoding="utf-8")

    def _make_blob(self, hexdigest: str) -> None:
        blobs = self.root / "blobs"
        blobs.mkdir(parents=True, exist_ok=True)
        (blobs / f"sha256-{hexdigest}").write_bytes(b"x")

    def test_all_blobs_present_yields_ok(self):
        h1 = "a" * 64
        h2 = "b" * 64
        self._make_blob(h1)
        self._make_blob(h2)
        self._make_manifest("phi4", [h1, h2])
        ok, missing = m.cross_validate_ollama_manifests(self.root)
        self.assertTrue(ok)
        self.assertEqual(missing, [])

    def test_one_blob_missing_yields_failure_and_does_not_clobber_source(self):
        h1 = "a" * 64
        h2 = "b" * 64
        self._make_blob(h1)  # only h1; h2 missing
        self._make_manifest("phi4", [h1, h2])
        ok, missing = m.cross_validate_ollama_manifests(self.root)
        self.assertFalse(ok)
        self.assertEqual(missing, [f"sha256-{h2}"])

    def test_no_manifests_yields_failure(self):
        """No manifests = we cannot verify, so we MUST fail closed."""
        self._make_blob("c" * 64)
        ok, missing = m.cross_validate_ollama_manifests(self.root)
        self.assertFalse(ok)


# ── Size / volume helpers ────────────────────────────────────────────────────


class SizeVolumeTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_measure_tree_size_walks_subdirs(self):
        _write_file(self.root / "a.bin", b"x" * 100)
        _write_file(self.root / "sub" / "b.bin", b"y" * 200)
        size, count = m.measure_tree_size(self.root)
        self.assertEqual(size, 300)
        self.assertEqual(count, 2)

    def test_same_volume_self_check_is_true(self):
        self.assertTrue(m.same_volume(self.root, self.root))

    def test_free_space_bytes_returns_positive_for_real_dir(self):
        self.assertGreater(m.free_space_bytes(self.root), 0)


# ── Pre-flight tests ─────────────────────────────────────────────────────────


class PreflightTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.source = self.root / "comfy"
        _write_file(self.source / "main.py", b"# comfy")
        _write_file(self.source / "models" / "checkpoints" / "sd.safetensors", b"x" * 1000)
        self.target = self.root / "moved"

    def _plan(self) -> m.MigrationPlan:
        return m.MigrationPlan(
            kind="comfyui", source=self.source, target=self.target,
            sentinel_check=m.has_comfyui_sentinel,
        )

    def test_preflight_passes_for_healthy_setup(self):
        engine = m.MigrationEngine(self._plan(), app_root=self.root,
                                    copy_runner=_fake_runner)
        result = engine.preflight()
        self.assertTrue(result.ok, result.reason)
        self.assertGreater(result.source_size, 0)

    def test_preflight_refuses_when_target_not_empty(self):
        self.target.mkdir(parents=True, exist_ok=True)
        (self.target / "leftover").write_bytes(b"old")
        engine = m.MigrationEngine(self._plan(), app_root=self.root,
                                    copy_runner=_fake_runner)
        result = engine.preflight()
        self.assertFalse(result.ok)
        self.assertIn("not empty", result.reason)

    def test_preflight_refuses_when_source_missing_sentinel(self):
        (self.source / "main.py").unlink()
        engine = m.MigrationEngine(self._plan(), app_root=self.root,
                                    copy_runner=_fake_runner)
        result = engine.preflight()
        self.assertFalse(result.ok)
        self.assertIn("sentinel", result.reason)

    def test_preflight_refuses_when_target_free_space_too_low(self):
        engine = m.MigrationEngine(
            self._plan(), app_root=self.root,
            copy_runner=_fake_runner,
            free_space_probe=lambda _p: 10,  # 10 bytes — impossibly small
            same_volume_probe=lambda _a, _b: False,  # force the 1.1x rule
        )
        result = engine.preflight()
        self.assertFalse(result.ok)
        self.assertIn("free", result.reason.lower())

    def test_preflight_same_volume_skips_huge_free_requirement(self):
        engine = m.MigrationEngine(
            self._plan(), app_root=self.root,
            copy_runner=_fake_runner,
            free_space_probe=lambda _p: 512 * 1024 * 1024,  # 512 MB free
            same_volume_probe=lambda _a, _b: True,  # SAME volume
            size_probe=lambda _p: (10 * 1024 ** 3, 1),  # 10 GB source
        )
        result = engine.preflight()
        # Same-volume = metadata-only, so 512 MB free is plenty.
        self.assertTrue(result.ok, result.reason)
        self.assertTrue(result.same_volume)

    def test_preflight_records_drive_mismatch_without_refusing(self):
        # source / target on the same drive as app_root; flag should be False.
        engine = m.MigrationEngine(self._plan(), app_root=self.root,
                                    copy_runner=_fake_runner)
        result = engine.preflight()
        self.assertTrue(result.ok)
        # drive_mismatch may be True or False depending on the test host's
        # drive layout; we just assert the field is a bool.
        self.assertIsInstance(result.drive_mismatch, bool)


# ── End-to-end engine.run() tests ────────────────────────────────────────────


class EngineRunTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.source = self.root / "comfy"
        _write_file(self.source / "main.py", b"# comfy")
        _write_file(self.source / "models" / "checkpoints" / "a.bin", b"x" * 100)
        self.target = self.root / "comfy_new"

    def test_happy_path_run_transitions_to_done_and_cleans_up_state(self):
        committed: list[Path] = []

        def _commit(target: Path) -> bool:
            committed.append(target)
            return True

        engine = m.MigrationEngine(
            m.MigrationPlan(
                kind="comfyui",
                source=self.source,
                target=self.target,
                sentinel_check=m.has_comfyui_sentinel,
            ),
            app_root=self.root,
            copy_runner=_fake_runner,
            config_commit=_commit,
        )
        engine.run()
        self.assertEqual(engine.phase, m.MigrationPhase.DONE)
        self.assertEqual(committed, [self.target])
        # State file deleted after DONE.
        self.assertFalse(m.state_file_path(self.root).exists())
        # Source renamed to <source>.deleteme.
        self.assertFalse(self.source.exists())
        self.assertTrue((self.root / "comfy.deleteme").exists())

    def test_cancel_during_copy_marks_state_done_and_clears(self):
        engine_ref: list = []
        engine = m.MigrationEngine(
            m.MigrationPlan(
                kind="comfyui",
                source=self.source,
                target=self.target,
                sentinel_check=m.has_comfyui_sentinel,
            ),
            app_root=self.root,
            copy_runner=_cancel_runner_factory(engine_ref),
        )
        with self.assertRaises(m.MigrationCancelled):
            engine.run()
        # Partial target removed; source still intact.
        self.assertFalse(self.target.exists())
        self.assertTrue(self.source.exists())
        self.assertTrue((self.source / "main.py").exists())
        # State file cleared.
        self.assertFalse(m.state_file_path(self.root).exists())

    def test_verify_failure_keeps_source_intact(self):
        """If verify says target is short, we MUST NOT delete source."""
        def _short_copy(engine: m.MigrationEngine) -> None:
            # Copy only main.py; deliberately drop the checkpoint file.
            Path(engine.plan.target).mkdir(parents=True, exist_ok=True)
            shutil.copy(str(Path(engine.plan.source) / "main.py"),
                        str(Path(engine.plan.target) / "main.py"))

        engine = m.MigrationEngine(
            m.MigrationPlan(
                kind="comfyui",
                source=self.source,
                target=self.target,
                sentinel_check=m.has_comfyui_sentinel,
            ),
            app_root=self.root,
            copy_runner=_short_copy,
        )
        with self.assertRaises(m.MigrationVerifyFailed):
            engine.run()
        self.assertEqual(engine.phase, m.MigrationPhase.FAILED)
        # Source MUST be intact — this is the today's-disaster prevention.
        self.assertTrue(self.source.exists())
        self.assertTrue((self.source / "models" / "checkpoints" / "a.bin").exists())

    def test_identity_keys_preserved_into_target(self):
        # Ollama-shaped layout
        ollama_src = self.root / "ollama_src"
        (ollama_src / "blobs").mkdir(parents=True)
        _write_file(ollama_src / "blobs" / "sha256-deadbeef", b"x")
        _write_file(ollama_src / "id_ed25519", b"PRIVATE")
        _write_file(ollama_src / "id_ed25519.pub", b"PUBLIC")
        # Manifest pointing at the blob
        h = "d" * 64
        manifest_dir = ollama_src / "manifests" / "registry.ollama.ai" / "library" / "x"
        manifest_dir.mkdir(parents=True)
        (manifest_dir / "latest").write_text(json.dumps({
            "config": {"digest": f"sha256:{h}"},
            "layers": [{"digest": f"sha256:{h}"}],
        }), encoding="utf-8")
        # Add the blob matching the manifest so verify passes.
        _write_file(ollama_src / "blobs" / f"sha256-{h}", b"y")
        ollama_dst = self.root / "ollama_dst"

        engine = m.MigrationEngine(
            m.MigrationPlan(
                kind="ollama",
                source=ollama_src,
                target=ollama_dst,
                sentinel_check=m.has_ollama_sentinel,
                is_ollama=True,
            ),
            app_root=self.root,
            copy_runner=_fake_runner,
        )
        engine.run()
        self.assertEqual(engine.phase, m.MigrationPhase.DONE)
        # Identity keys must exist at target.
        self.assertTrue((ollama_dst / "id_ed25519").exists(),
                        "identity private key must be preserved")
        self.assertTrue((ollama_dst / "id_ed25519.pub").exists(),
                        "identity public key must be preserved")

    def test_ollama_verify_rejects_missing_blob_and_keeps_source(self):
        ollama_src = self.root / "ollama_src"
        (ollama_src / "blobs").mkdir(parents=True)
        h = "e" * 64
        _write_file(ollama_src / "blobs" / f"sha256-{h}", b"x")
        manifest_dir = ollama_src / "manifests" / "registry.ollama.ai" / "library" / "x"
        manifest_dir.mkdir(parents=True)
        # Manifest references TWO blobs; only one exists.
        h2 = "f" * 64
        (manifest_dir / "latest").write_text(json.dumps({
            "config": {"digest": f"sha256:{h}"},
            "layers": [{"digest": f"sha256:{h}"}, {"digest": f"sha256:{h2}"}],
        }), encoding="utf-8")
        ollama_dst = self.root / "ollama_dst"

        engine = m.MigrationEngine(
            m.MigrationPlan(
                kind="ollama",
                source=ollama_src,
                target=ollama_dst,
                sentinel_check=m.has_ollama_sentinel,
                is_ollama=True,
            ),
            app_root=self.root,
            copy_runner=_fake_runner,
        )
        with self.assertRaises(m.MigrationVerifyFailed) as ctx:
            engine.run()
        self.assertIn(f"sha256-{h2}", " ".join(ctx.exception.missing_blobs))
        # Source intact.
        self.assertTrue(ollama_src.exists())
        self.assertTrue((ollama_src / "blobs" / f"sha256-{h}").exists())


# ── Scheduled-delete cleanup ─────────────────────────────────────────────────


class ScheduledDeleteTests(unittest.TestCase):

    def test_process_scheduled_deletes_removes_dot_deleteme_dirs(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "comfy.deleteme").mkdir()
        _write_file(root / "comfy.deleteme" / "x.bin", b"x")
        (root / "models.deleteme.1").mkdir()
        (root / "keepme").mkdir()
        deleted = m.process_scheduled_deletes(root)
        names = {Path(p).name for p in deleted}
        self.assertIn("comfy.deleteme", names)
        self.assertIn("models.deleteme.1", names)
        self.assertFalse((root / "comfy.deleteme").exists())
        self.assertTrue((root / "keepme").exists())


# ── Drive-volatility guard ───────────────────────────────────────────────────


class DriveTypeTests(unittest.TestCase):

    def test_get_drive_type_returns_int(self):
        v = m.get_drive_type(Path.cwd())
        self.assertIsInstance(v, int)

    def test_reboot_volatile_drive_types_set_contains_removable_and_ram(self):
        self.assertIn(m._GDT_REMOVABLE, m.REBOOT_VOLATILE_DRIVE_TYPES)
        self.assertIn(m._GDT_RAMDISK, m.REBOOT_VOLATILE_DRIVE_TYPES)
        self.assertNotIn(m._GDT_FIXED, m.REBOOT_VOLATILE_DRIVE_TYPES)
        self.assertNotIn(m._GDT_REMOTE, m.REBOOT_VOLATILE_DRIVE_TYPES)


if __name__ == "__main__":
    unittest.main()
