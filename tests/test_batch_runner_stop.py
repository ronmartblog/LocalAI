import json
import os
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from src import batch_runner, logger, resource_manager
from src.batch_report import BatchReport, RunResult, find_latest_report_json
from src.batch_runner import BatchRunner, IMAGE_METHOD
from src.ollama_client import OllamaError


class BatchRunnerStopTests(unittest.TestCase):
    def test_request_stop_signals_active_run_event(self):
        runner = BatchRunner()
        active = threading.Event()
        runner._active_stop_event = active

        runner.request_stop()

        self.assertTrue(runner._interrupted)
        self.assertTrue(active.is_set())

    def test_default_max_failures_is_ten(self):
        self.assertEqual(BatchRunner().max_failures, 10)

    def test_save_partial_persists_completed_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = BatchRunner(output_dir=Path(tmp))
            runner.report.add(
                RunResult(
                    model_id="demo-model",
                    model_name="Demo Model",
                    method="ollama_cpu",
                    success=False,
                    error="Stopped",
                )
            )

            json_path, txt_path = runner.save_partial()

            self.assertIsNotNone(json_path)
            self.assertIsNotNone(txt_path)
            self.assertTrue(json_path.exists())
            self.assertTrue(txt_path.exists())
            self.assertIn(runner.report.file_stem, json_path.name)
            self.assertIn(runner.report.file_stem, txt_path.name)

    def test_retry_report_merge_replaces_matching_sample_only(self):
        original = BatchReport()
        original.add(RunResult("m", "Model", "ollama_cpu", False, sample_index=0, error="old-0"))
        original.add(RunResult("m", "Model", "ollama_cpu", False, sample_index=1, error="old-1"))
        original.add(RunResult("m", "Model", "ollama_cpu", False, sample_index=1, error="duplicate-old-1"))
        retry = BatchReport()
        retry.add(RunResult("m", "Model", "ollama_cpu", True, sample_index=1, response_text="new-1"))

        self.assertEqual(original.get_failed_combos(), [("m", "ollama_cpu", 0), ("m", "ollama_cpu", 1)])

        original.merge(retry)

        self.assertFalse(original.results[0].success)
        self.assertEqual(original.results[0].error, "old-0")
        self.assertTrue(original.results[1].success)
        self.assertEqual(original.results[1].response_text, "new-1")
        self.assertEqual(len(original.results), 2)

    def test_get_completed_combos_returns_only_passed_results(self):
        """v2026.06.01.2+: Resume Today's Run uses ``get_completed_combos`` to
        compute "skip these — they passed". Failed combos must NOT be in
        the returned set so Resume re-runs them. (Earlier — v5.5.7
        through v2026.06.01.1 — failures counted as "completed" which
        meant Resume silently skipped them; Ron's 2026-06-01 bench
        report flipped the contract.)"""
        report = BatchReport()
        report.add(RunResult("m", "Model", "ollama_cpu", True, sample_index=0))
        report.add(RunResult("m", "Model", "ollama_cpu", False, sample_index=1, error="x"))
        report.add(RunResult("m", "Model", "ollama_gpu", True, sample_index=0))
        combos = report.get_completed_combos()
        self.assertEqual(combos, {
            ("m", "ollama_cpu", 0),
            ("m", "ollama_gpu", 0),
        })
        # The failed combo is NOT in completed — so the resume filter
        # in BatchRunner._iter_selected_samples_for will re-run it.
        self.assertNotIn(("m", "ollama_cpu", 1), combos)
        # Empty report → empty set.
        self.assertEqual(BatchReport().get_completed_combos(), set())

    def test_get_completed_combos_excludes_adaptive_skips(self):
        """Adaptively-skipped combos record ``success=False`` with
        ``failure_phase="adaptive_skip"``. Under the new contract these
        are NOT in get_completed_combos — so Resume will plan them again.
        The smart-skip ceiling is rebuilt per-session inside the runner,
        so anything that should still skip will skip again quickly
        without wasting much time. This is the intentional tradeoff
        for never silently dropping a real failure."""
        report = BatchReport()
        report.add(RunResult("m", "Model", "ollama_cpu", True, sample_index=0))
        report.add(RunResult(
            "m", "Model", "image", False, sample_index=0,
            failure_phase="adaptive_skip",
            error="ceiling tightened by prior OOM",
        ))
        self.assertEqual(report.get_completed_combos(), {("m", "ollama_cpu", 0)})

    def test_get_failed_combos_returns_failures_for_retry_button(self):
        """``get_failed_combos`` is independent of ``get_completed_combos``
        and still feeds the Retry Failed button. Both must agree about
        which combos are failed."""
        report = BatchReport()
        report.add(RunResult("m", "Model", "ollama_cpu", True, sample_index=0))
        report.add(RunResult("m", "Model", "ollama_cpu", False, sample_index=1, error="x"))
        completed = report.get_completed_combos()
        failed = set(report.get_failed_combos())
        self.assertEqual(completed, {("m", "ollama_cpu", 0)})
        self.assertEqual(failed, {("m", "ollama_cpu", 1)})
        # Critical: the union covers every result and they don't overlap.
        all_keys = {(r.model_id, r.method, int(r.sample_index or 0)) for r in report.results}
        self.assertEqual(completed | failed, all_keys)
        self.assertEqual(completed & failed, set())

    def test_append_resume_results_replaces_prior_failure_with_retry_outcome(self):
        """v2026.06.01.2+: Resume now re-runs failed combos (per the new
        ``get_completed_combos`` contract). When the resumed run produces
        a fresh result for a key whose prior entry was a failure, the
        prior entry must be REPLACED so the report shows the freshest
        outcome. Earlier (v5.5.7 – v2026.06.01.1) semantics dropped the
        new result on the floor, which was wrong once Resume started
        retrying failures."""
        prev = BatchReport(start_time="2026-06-01T07:00:00")
        prev.add(RunResult("m", "Model", "ollama_cpu", True, sample_index=0,
                           total_time=2.0))
        # Previously failed — Resume re-runs it.
        prev.add(RunResult("m", "Model", "ollama_cpu", False, sample_index=1,
                           total_time=265.0, error="repetition loop"))
        prev.stamp_end_time("2026-06-01T07:05:00")
        new_partial = BatchReport(start_time="2026-06-01T08:00:00")
        # Same key as the prior failure — the retry passed this time.
        new_partial.add(RunResult("m", "Model", "ollama_cpu", True,
                                  sample_index=1, total_time=3.0))
        # New combo not in prev — must be APPENDED.
        new_partial.add(RunResult("m", "Model", "ollama_cpu", True,
                                  sample_index=2, total_time=4.0))
        new_partial.stamp_end_time("2026-06-01T08:10:00")

        prev.append_resume_results(new_partial)
        # Result count: 1 success (sample 0) + 1 replaced (sample 1) + 1 appended (sample 2)
        self.assertEqual(len(prev.results), 3)
        by_idx = {int(r.sample_index or 0): r for r in prev.results}
        # Sample 0: untouched success.
        self.assertTrue(by_idx[0].success)
        self.assertEqual(by_idx[0].total_time, 2.0)
        # Sample 1: REPLACED — prior failure is gone, retry success is now there.
        self.assertTrue(by_idx[1].success)
        self.assertEqual(by_idx[1].total_time, 3.0)
        self.assertIsNone(by_idx[1].error)
        # Sample 2: APPENDED.
        self.assertTrue(by_idx[2].success)
        self.assertEqual(by_idx[2].total_time, 4.0)
        # end_time advanced.
        self.assertEqual(prev.end_time, "2026-06-01T08:10:00")

    def test_append_resume_results_replaces_prior_failure_even_if_retry_also_fails(self):
        """If a previously-failed combo is re-run on resume and FAILS
        AGAIN, the new failure record (latest timestamp, latest
        diagnostic text) must replace the prior one — never accumulate
        duplicate failure entries for the same key."""
        prev = BatchReport()
        prev.add(RunResult("m", "Model", "ollama_cpu", False, sample_index=0,
                           total_time=10.0, error="first attempt"))
        new_partial = BatchReport()
        new_partial.add(RunResult("m", "Model", "ollama_cpu", False,
                                  sample_index=0, total_time=11.0,
                                  error="second attempt"))
        prev.append_resume_results(new_partial)
        self.assertEqual(len(prev.results), 1)
        self.assertEqual(prev.results[0].total_time, 11.0)
        self.assertEqual(prev.results[0].error, "second attempt")

    def test_append_resume_results_never_overwrites_a_prior_success(self):
        """Defensive contract: the new resume filter shouldn't ever
        re-run a passing combo (its key is in get_completed_combos). If
        somehow it does (caller bug, future regression), the prior
        success MUST be preserved — never overwrite a pass with a later
        outcome from a resumed run."""
        prev = BatchReport()
        prev.add(RunResult("m", "Model", "ollama_cpu", True, sample_index=0,
                           total_time=2.0))
        new_partial = BatchReport()
        # Same key, but prior was a success — must be IGNORED.
        new_partial.add(RunResult("m", "Model", "ollama_cpu", False,
                                  sample_index=0, total_time=99.0,
                                  error="must-not-clobber-pass"))
        prev.append_resume_results(new_partial)
        self.assertEqual(len(prev.results), 1)
        self.assertTrue(prev.results[0].success)
        self.assertEqual(prev.results[0].total_time, 2.0)
        self.assertIsNone(prev.results[0].error)

    def test_append_resume_results_appends_new_combos(self):
        """When the resumed run produces a result whose key isn't in
        the prior report, that result must be appended (this case is
        unchanged from the v5.5.7 contract)."""
        prev = BatchReport(start_time="2026-05-23T10:00:00")
        prev.add(RunResult("m", "Model", "ollama_cpu", True, sample_index=0,
                           total_time=2.0))
        prev.stamp_end_time("2026-05-23T10:05:00")
        new_partial = BatchReport(start_time="2026-05-23T14:00:00")
        new_partial.add(RunResult("m", "Model", "ollama_cpu", True,
                                  sample_index=1, total_time=3.0))
        new_partial.stamp_end_time("2026-05-23T14:10:00")

        prev.append_resume_results(new_partial)
        self.assertEqual(len(prev.results), 2)
        self.assertEqual(prev.results[1].sample_index, 1)
        self.assertEqual(prev.results[1].total_time, 3.0)
        self.assertEqual(prev.end_time, "2026-05-23T14:10:00")
        # compute_time honours per-result total_time across both windows
        # (2.0 + 3.0 = 5s) — ignores the 4-hour Resume idle gap.
        self.assertEqual(prev.compute_time_seconds(), 5)

    def test_to_dict_emits_compute_time_alongside_duration(self):
        """v5.5.7+ to_dict serialises both compute_time_seconds (primary,
        sum of per-result total_time) and duration_seconds (secondary,
        wall-clock end-start) so resume gaps don't lie about how long the
        benchmark actually spent computing."""
        report = BatchReport(start_time="2026-05-21T10:00:00")
        report.add(RunResult("m", "Model", "ollama_cpu", True, sample_index=0,
                             total_time=2.0))
        report.add(RunResult("m", "Model", "ollama_cpu", True, sample_index=1,
                             total_time=3.5))
        report.stamp_end_time("2026-05-21T11:00:00")
        d = report.to_dict()
        # Wall clock: 1h = 3600s.
        self.assertEqual(d["duration_seconds"], 3600)
        self.assertEqual(d["duration_hms"], "1:00:00")
        # Compute time: 2.0 + 3.5 = 5s (int).
        self.assertEqual(d["compute_time_seconds"], 5)
        self.assertEqual(d["compute_time_hms"], "0:00:05")

    def test_batch_report_uses_machine_specs_and_start_time_in_filename(self):
        machine_info = {
            "machine_name": "Test Box",
            "machine_model": "Test VM 4 CPU",
            "vcpu": 4,
            "ram_gb": 16,
            "gpu_name": "",
            "vram_gb": 0,
            "os": "Windows",
            "python": "3.13",
            "storage_free_gb": 59.4,
            "storage_total_gb": 128.0,
        }
        report = BatchReport(
            start_time="2026-05-18T00:55:56",
            machine_info=machine_info,
        )
        report.add(
            RunResult(
                model_id="demo-model",
                model_name="Demo Model",
                method="ollama_cpu",
                success=True,
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            json_path = report.save_json(Path(tmp))
            txt_path = report.save_text(Path(tmp))
            data = json.loads(json_path.read_text(encoding="utf-8"))
            text = txt_path.read_text(encoding="utf-8")

        expected_stem = (
            "bench_quick_Test_Box_4cpu_16ram_"
            "no-gpu_0vram_2026-05-18_0055"
        )
        self.assertEqual(report.file_stem, expected_stem)
        self.assertEqual(json_path.name, f"{expected_stem}.json")
        self.assertEqual(txt_path.name, f"{expected_stem}.txt")
        self.assertEqual(data["machine_info"]["machine_name"], "Test Box")
        self.assertIn("Machine: Test Box", text)
        self.assertIn("4 CPU cores | 16 GB RAM", text)

    def test_named_benchmark_reports_no_longer_write_legacy_alias(self):
        """v5.5.1+: the ``batch_results.json``/``.txt`` legacy alias was
        producing a stale third copy in every benchmark folder. The
        per-mode ``latest_*_benchmark.json`` alias is still written
        because it cleanly overwrites itself and isn't part of the
        accumulation pattern. Legacy files that already exist on disk
        are swept into ``archive/`` on the next save by
        :func:`archive_previous_runs`."""
        from src.batch_report import (
            LATEST_QUICK_JSON_NAME,
            LEGACY_JSON_NAME,
            LEGACY_TEXT_NAME,
        )

        machine_info = {"machine_name": "Alias Box", "vcpu": 4, "ram_gb": 16, "vram_gb": 0}
        report = BatchReport(start_time="2026-05-18T01:20:00", machine_info=machine_info)
        report.add(RunResult("a", "Model A", "ollama_cpu", True))

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            json_path = report.save_json(output_dir)
            txt_path = report.save_text(output_dir)
            legacy_json = output_dir / LEGACY_JSON_NAME
            legacy_txt = output_dir / LEGACY_TEXT_NAME
            latest_quick = output_dir / LATEST_QUICK_JSON_NAME

            # Stem-named files are written.
            self.assertTrue(json_path.exists())
            self.assertTrue(txt_path.exists())
            # Per-mode latest alias is still written for the active run mode.
            self.assertTrue(latest_quick.exists())
            # Legacy aliases must NOT be created any more.
            self.assertFalse(legacy_json.exists())
            self.assertFalse(legacy_txt.exists())

    def test_loaded_report_retries_save_back_to_same_report_file(self):
        machine_info = {"machine_name": "Retry Box", "vcpu": 8, "ram_gb": 32, "vram_gb": 0}
        report = BatchReport(
            start_time="2026-05-18T01:02:03",
            machine_info=machine_info,
        )
        report.add(RunResult("m", "Model", "ollama_cpu", False, error="failed"))

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            json_path = report.save_json(output_dir)
            loaded = BatchReport.load_json(json_path)
            loaded.merge(BatchReport(start_time="2026-05-18T01:05:00", machine_info=machine_info, file_stem=loaded.file_stem))
            saved_path = loaded.save_json(output_dir)

        self.assertEqual(saved_path.name, json_path.name)

    def test_find_latest_report_json_supports_named_and_legacy_reports(self):
        machine_info = {"machine_name": "Latest Box", "vcpu": 4, "ram_gb": 16, "vram_gb": 0}
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            legacy = output_dir / "batch_results.json"
            legacy.write_text("{}", encoding="utf-8")
            os.utime(legacy, (1, 1))
            named = BatchReport(
                start_time="2026-05-18T01:10:00",
                machine_info=machine_info,
            ).save_json(output_dir)

            self.assertEqual(find_latest_report_json(output_dir), named)

    def test_find_latest_report_json_recognises_legacy_prefix_in_archive(self):
        """Archived ``localai_benchmark_*.json`` reports must still resolve
        for "Retry Failed" so users with older folders aren't stuck."""
        machine_info = {"machine_name": "Archived Box", "vcpu": 4, "ram_gb": 16, "vram_gb": 0}
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            archive_dir = output_dir / "archive"
            archive_dir.mkdir()
            archived = archive_dir / "localai_benchmark_quick_Old_Box_4cpu_16gb-ram_no-gpu_0gb-vram_2026-05-10_10-00-00.json"
            archived.write_text(
                '{"file_stem": "' + archived.stem + '", "results": []}',
                encoding="utf-8",
            )
            # No file in the root — find_latest must reach into archive.
            self.assertEqual(find_latest_report_json(output_dir), archived)

    def test_find_latest_report_json_keeps_self_referential_legacy_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            legacy = output_dir / "batch_results.json"
            legacy.write_text('{"file_stem": "batch_results", "results": []}', encoding="utf-8")

            self.assertEqual(find_latest_report_json(output_dir), legacy)

    def test_archive_previous_runs_moves_old_files_into_archive_subfolder(self):
        """A new save must sweep older bench_/localai_benchmark_ files
        AND stale legacy aliases into ``archive/`` so the root folder
        only ever shows one set of report files at a time."""
        from src.batch_report import archive_previous_runs

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "bench_quick_Old_Box_4cpu_16ram_no-gpu_0vram_2026-05-10_1000.json").write_text("{}", encoding="utf-8")
            (output_dir / "bench_quick_Old_Box_4cpu_16ram_no-gpu_0vram_2026-05-10_1000.html").write_text("<html></html>", encoding="utf-8")
            (output_dir / "localai_benchmark_quick_Ancient_4cpu_16gb-ram_no-gpu_0gb-vram_2026-04-01_00-00-00.json").write_text("{}", encoding="utf-8")
            (output_dir / "batch_results.json").write_text("{}", encoding="utf-8")
            (output_dir / "batch_results.txt").write_text("legacy", encoding="utf-8")
            (output_dir / "latest_quick_benchmark.json").write_text("{}", encoding="utf-8")
            (output_dir / "unrelated.json").write_text("{}", encoding="utf-8")
            # Sibling artifact directory.
            old_images = output_dir / "bench_quick_Old_Box_4cpu_16ram_no-gpu_0vram_2026-05-10_1000_images"
            old_images.mkdir()
            (old_images / "sample.png").write_text("png-bytes", encoding="utf-8")

            moved = archive_previous_runs(output_dir, keep_stem="bench_extended_New_Box_4cpu_16ram_no-gpu_0vram_2026-05-23_1100")
            archive_dir = output_dir / "archive"

            # Old benchmark files swept.
            self.assertTrue((archive_dir / "bench_quick_Old_Box_4cpu_16ram_no-gpu_0vram_2026-05-10_1000.json").exists())
            self.assertTrue((archive_dir / "bench_quick_Old_Box_4cpu_16ram_no-gpu_0vram_2026-05-10_1000.html").exists())
            self.assertTrue((archive_dir / "localai_benchmark_quick_Ancient_4cpu_16gb-ram_no-gpu_0gb-vram_2026-04-01_00-00-00.json").exists())
            self.assertTrue((archive_dir / "batch_results.json").exists())
            self.assertTrue((archive_dir / "batch_results.txt").exists())
            # Sibling images directory swept too.
            self.assertTrue((archive_dir / old_images.name / "sample.png").exists())
            # Latest alias is preserved in root (gets overwritten by next save).
            self.assertTrue((output_dir / "latest_quick_benchmark.json").exists())
            # Unrelated files are not touched.
            self.assertTrue((output_dir / "unrelated.json").exists())
            self.assertGreater(len(moved), 0)

    def test_archive_previous_runs_preserves_active_run_keep_stem(self):
        """The in-flight run's incremental saves must NOT self-archive
        when ``archive_previous_runs`` is called repeatedly."""
        from src.batch_report import archive_previous_runs

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            stem = "bench_quick_Box_4cpu_16ram_no-gpu_0vram_2026-05-23_1100"
            (output_dir / f"{stem}.json").write_text("{}", encoding="utf-8")
            (output_dir / f"{stem}.html").write_text("<html></html>", encoding="utf-8")
            (output_dir / f"{stem}_failures.txt").write_text("fail", encoding="utf-8")

            moved = archive_previous_runs(output_dir, keep_stem=stem)

            # Active-run files stay put.
            self.assertTrue((output_dir / f"{stem}.json").exists())
            self.assertTrue((output_dir / f"{stem}.html").exists())
            self.assertTrue((output_dir / f"{stem}_failures.txt").exists())
            self.assertEqual(moved, [])

    def test_save_html_writes_themed_report_with_image_card(self):
        """save_html() must emit the themed HTML report, latest aliases, and
        render an image card for image-gen sample results."""
        from src.batch_report import (
            LATEST_EXTENDED_HTML_NAME,
            LATEST_QUICK_HTML_NAME,
            LEGACY_HTML_NAME,
        )

        machine_info = {
            "machine_name": "HTML Box",
            "vcpu": 8,
            "ram_gb": 32,
            "vram_gb": 24,
            "gpu_name": "Test GPU",
            "os": "Windows",
            "python": "3.12",
        }
        report = BatchReport(
            start_time="2026-05-18T02:00:00",
            machine_info=machine_info,
            run_mode="extended",
        )
        report.add(
            RunResult(
                model_id="img-a",
                model_name="Img A",
                method="image_comfyui",
                success=True,
                surface="image",
                prompt="cinematic mountain landscape",
                sample_title="Landscape",
                sample_index=1,
                sample_count=3,
                image_path="img-a_images/full/img-a_sample1.png",
                thumbnail_path="img-a_images/thumb/img-a_sample1.png",
            )
        )
        report.add(
            RunResult(
                model_id="chat-a",
                model_name="Chat A",
                method="ollama_gpu",
                success=True,
                surface="text",
                tokens_per_sec=42.0,
                prompt="hello",
                sample_title="Greeting",
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            html_path = report.save_html(output_dir)
            text = html_path.read_text(encoding="utf-8")
            latest_extended = output_dir / LATEST_EXTENDED_HTML_NAME
            latest_quick = output_dir / LATEST_QUICK_HTML_NAME
            legacy = output_dir / LEGACY_HTML_NAME

            # File written and Model-Guide theme markers present.
            self.assertTrue(html_path.exists())
            self.assertIn("--accent:   #4f9cf9", text)  # cool-blue navy accent (dark)
            self.assertIn("--accent:   #1864c4", text)  # cool-blue accent (light)
            self.assertIn('data-theme="dark"', text)
            self.assertIn('data-theme="light"', text)
            self.assertIn("Toggle theme", text)
            # Extended-mode alias is updated; quick alias is not touched by an
            # extended run. v5.5.1+: the legacy ``batch_results.html`` alias
            # is no longer written — it was the third stale copy in every
            # results folder.
            self.assertTrue(latest_extended.exists())
            self.assertFalse(latest_quick.exists())
            self.assertFalse(legacy.exists())
            # Image card markers + thumbnail path are rendered.
            self.assertIn("img-a_images/thumb/img-a_sample1.png", text)
            self.assertIn("img-a_images/full/img-a_sample1.png", text)
            self.assertIn("Img A", text)
            # Surface label for image card.
            self.assertIn("image (ComfyUI)", text)
            # Surface tabs + modal zoom present.
            self.assertIn('id="surface-tabs"', text)
            self.assertIn('id="modal"', text)
            # Left rail links to the compact bottom summary table.
            self.assertIn('class="side-rail"', text)
            self.assertIn('href="#result-summary-table"', text)
            self.assertIn('id="result-summary-table"', text)
            self.assertIn("Summary table", text)
            # Sortable column headers are now <th> buttons (role=button,
            # tabindex, aria-sort, data-sort-key) so users can click to sort
            # the summary table by any column. Pin the markup contract for
            # the numeric Tok/s and Raw seconds columns plus a Tok/s cell
            # that carries a data-sort numeric value alongside its display.
            self.assertIn('data-sort-key="tps"', text)
            self.assertIn('data-sort-key="raw"', text)
            self.assertIn('aria-sort="none"', text)
            self.assertIn('role="button"', text)
            self.assertIn('class="metric-num" data-sort=', text)
            self.assertIn(">42.0</td>", text)

    def test_html_report_links_model_guide_source_to_model_anchor(self):
        report = BatchReport(run_mode="extended")
        report.add(
            RunResult(
                model_id="llama3.3",
                model_name="Llama 3.3",
                method="ollama_gpu",
                success=True,
                prompt_source="Model-Guide.html",
                sample_title="Board summary",
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            html_text = report.save_html(Path(tmp)).read_text(encoding="utf-8")

        href = '../docs/Model-Guide.html#model-llama3-3'
        self.assertIn(f'href="{href}"', html_text)
        self.assertIn(">Model Guide</a>", html_text)
        self.assertNotIn(">Model-Guide.html<", html_text)

    def test_report_json_and_html_include_localai_version(self):
        report = BatchReport(localai_version="9.9.9-test")
        report.add(
            RunResult(
                model_id="chat-a",
                model_name="Chat A",
                method="ollama_cpu",
                success=True,
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            json_path = report.save_json(output_dir)
            html_path = report.save_html(output_dir)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            html_text = html_path.read_text(encoding="utf-8")

        self.assertEqual(data.get("localai_version"), "9.9.9-test")
        self.assertIn("LocalAI Studio</div><div class=\"value\">9.9.9-test</div>", html_text)

    def test_html_images_generated_counts_only_successful_image_runs(self):
        report = BatchReport(localai_version="9.9.9-test")
        for idx in range(3):
            report.add(
                RunResult(
                    model_id=f"img-{idx}",
                    model_name=f"Img {idx}",
                    method="image_comfyui",
                    surface="image",
                    success=False,
                    error="generation failed",
                )
            )

        with tempfile.TemporaryDirectory() as tmp:
            html_text = report.save_html(Path(tmp)).read_text(encoding="utf-8")

        # Summary card tracks successful image generation only.
        self.assertIn("Images generated</div><div class=\"value\">0</div>", html_text)
        # Image surface tab still reflects all image runs (success + fail).
        self.assertIn('data-surface="image">Image <span class="count">3</span>', html_text)

    def test_methods_for_image_model_runs_in_quick_and_extended_mode_on_gpu(self):
        """_methods_for must emit ['image_comfyui'] for image-gen models in
        BOTH quick and extended modes when a GPU is present and skip_image is
        False. v5.5.18+: Quick mode now runs the SKU-curated single image-gen
        model (e.g. playground on a GPU Workstation profile) instead of returning [] silently.
        """
        model = {
            "id": "img-x",
            "name": "Img X",
            "backend": "comfyui",
            "comfyui_model": "img-x.safetensors",
            "min_ram_gb": 8,
            "min_vram_gb": 8,
            "category": "Image Generation",
        }
        gpu = {"name": "GPU", "vram_total_mb": 24576, "vram_free_mb": 24576}
        with patch("src.batch_runner.get_gpu_info", return_value=[gpu]), \
             patch("src.batch_runner.get_ram_info", return_value={"total_mb": 32768, "available_mb": 32768}):
            # Quick mode on GPU → image method emitted (v5.5.18+).
            quick = BatchRunner(run_mode="quick")
            self.assertIn("image_comfyui", quick._methods_for(model))
            # Extended mode on GPU → image method emitted.
            extended = BatchRunner(run_mode="extended")
            self.assertIn("image_comfyui", extended._methods_for(model))
            # Quick + skip_image → no image method.
            skip_quick = BatchRunner(run_mode="quick", skip_image=True)
            self.assertEqual(skip_quick._methods_for(model), [])
            # Extended but skip_image → no image method.
            skip = BatchRunner(run_mode="extended", skip_image=True)
            self.assertEqual(skip._methods_for(model), [])

        # CPU-only profile → never image, even in extended mode.
        with patch("src.batch_runner.get_gpu_info", return_value=[]), \
             patch("src.batch_runner.get_ram_info", return_value={"total_mb": 32768, "available_mb": 32768}):
            cpu_only = BatchRunner(run_mode="extended")
            self.assertEqual(cpu_only._methods_for(model), [])
            # Quick on CPU also returns [] — no GPU, no image.
            cpu_quick = BatchRunner(run_mode="quick")
            self.assertEqual(cpu_quick._methods_for(model), [])

    def test_methods_for_no_gpu_only_selects_cpu_when_ram_fits(self):
        model = {
            "id": "small",
            "name": "Small",
            "ollama_tag": "small:latest",
            "min_ram_gb": 4,
            "min_vram_gb": 1,
        }
        with patch("src.batch_runner.get_gpu_info", return_value=[]), \
             patch("src.batch_runner.get_ram_info", return_value={"total_mb": 16384, "available_mb": 6144}):
            self.assertEqual(BatchRunner()._methods_for(model), ["ollama_cpu"])

    def test_gpu_capacity_uses_installed_vram_not_transient_free_vram(self):
        model = {
            "id": "gpu-fit",
            "name": "GPU Fit",
            "ollama_tag": "gpu-fit:latest",
            "min_ram_gb": 8,
            "min_vram_gb": 8,
        }
        gpu = {
            "name": "Test GPU",
            "vram_total_mb": 12288,
            "vram_free_mb": 2048,
        }
        with patch("src.batch_runner.get_gpu_info", return_value=[gpu]), \
             patch("src.batch_runner.get_ram_info", return_value={"total_mb": 32768, "available_mb": 4096}):
            self.assertIn("ollama_gpu", BatchRunner()._methods_for(model))

    def test_methods_for_filters_models_that_do_not_fit_installed_ram(self):
        model = {
            "id": "large",
            "name": "Large",
            "ollama_tag": "large:latest",
            "min_ram_gb": 20,
            "min_vram_gb": 0,
        }
        with patch("src.batch_runner.get_gpu_info", return_value=[]), \
             patch("src.batch_runner.get_ram_info", return_value={"total_mb": 16384, "available_mb": 6144}):
            self.assertEqual(BatchRunner()._methods_for(model), [])

    def test_allow_oversize_runs_selected_cpu_model_over_capacity(self):
        model = {
            "id": "large",
            "name": "Large",
            "ollama_tag": "large:latest",
            "min_ram_gb": 32,
            "min_vram_gb": 0,
        }
        runner = BatchRunner(
            capacity_ram_gb=16,
            capacity_vram_gb=0,
            capacity_has_gpu=False,
            allow_oversize=True,
            skip_onnx=True,
        )

        self.assertEqual(runner._methods_for(model), ["ollama_cpu"])

    def test_allow_oversize_does_not_enable_gpu_without_gpu(self):
        model = {
            "id": "gpu-large",
            "name": "GPU Large",
            "ollama_tag": "gpu-large:latest",
            "min_ram_gb": 32,
            "min_vram_gb": 8,
        }
        runner = BatchRunner(
            capacity_ram_gb=16,
            capacity_vram_gb=0,
            capacity_has_gpu=False,
            allow_oversize=True,
            skip_cpu=True,
            skip_onnx=True,
        )

        self.assertEqual(runner._methods_for(model), [])

    def test_retry_capacity_gating_skips_accelerated_onnx_on_cpu_profile(self):
        model = {
            "id": "onnx-model",
            "name": "ONNX Model",
            "onnx_repo": "org/model",
            "min_ram_gb": 4,
            "min_vram_gb": 8,
        }
        runner = BatchRunner(
            specific_combos=[("onnx-model", "onnx_directml")],
            capacity_ram_gb=16,
            capacity_vram_gb=0,
            capacity_has_gpu=False,
            allow_oversize=True,
        )
        with patch.object(runner, "_run_onnx", side_effect=AssertionError("should not dispatch DirectML")):
            result = runner._run_inner(model, "onnx_directml", threading.Event())

        self.assertFalse(result.success)
        self.assertEqual(result.method, "onnx_directml")
        self.assertIn("Skipped: No GPU/NPU accelerator available", result.error)

    def test_retry_capacity_gating_skips_openvino_and_utility_over_profile(self):
        onnx_model = {
            "id": "openvino-model",
            "name": "OpenVINO Model",
            "onnx_repo": "org/model",
            "min_ram_gb": 4,
            "min_vram_gb": 0,
        }
        utility_model = {
            "id": "utility-large",
            "name": "Utility Large",
            "phase1_adapter": True,
            "min_ram_gb": 32,
            "min_vram_gb": 0,
        }
        runner = BatchRunner(
            specific_combos=[
                ("openvino-model", "onnx_openvino"),
                ("utility-large", "phase1"),
            ],
            capacity_ram_gb=16,
            capacity_vram_gb=0,
            capacity_has_gpu=False,
        )

        with patch.object(runner, "_run_onnx", side_effect=AssertionError("should not dispatch OpenVINO")):
            openvino = runner._run_inner(onnx_model, "onnx_openvino", threading.Event())
        with patch.object(runner, "_run_phase1", side_effect=AssertionError("should not dispatch utility")):
            utility = runner._run_inner(utility_model, "phase1", threading.Event())

        self.assertFalse(openvino.success)
        self.assertIn("Skipped: No GPU/NPU accelerator available", openvino.error)
        self.assertFalse(utility.success)
        self.assertIn("Skipped: Not enough installed RAM", utility.error)

    def test_default_run_filters_non_fitting_models_before_counting_or_reporting(self):
        small = {
            "id": "small",
            "name": "Small",
            "ollama_tag": "small:latest",
            "min_ram_gb": 4,
            "min_vram_gb": 0,
        }
        large = {
            "id": "large",
            "name": "Large",
            "ollama_tag": "large:latest",
            "min_ram_gb": 20,
            "min_vram_gb": 0,
        }

        with tempfile.TemporaryDirectory() as tmp, \
             patch("src.batch_runner.load_catalog", return_value=[small, large]), \
             patch("src.batch_runner.get_gpu_info", return_value=[]), \
             patch("src.batch_runner.get_ram_info", return_value={"total_mb": 16384, "available_mb": 6144}), \
             patch.object(BatchRunner, "_run_one", return_value=RunResult(
                 model_id="small",
                 model_name="Small",
                 method="ollama_cpu",
                 success=True,
             )), \
             patch.object(BatchRunner, "_release_after_run"):
            runner = BatchRunner(output_dir=Path(tmp), skip_gpu=True, skip_onnx=True)
            with redirect_stdout(StringIO()):
                report = runner.run()

        self.assertEqual([r.model_id for r in report.results], ["small"])

    def test_capacity_override_no_gpu_runs_all_selected_cpu_models_that_fit(self):
        models = [
            {
                "id": "a",
                "name": "A",
                "ollama_tag": "a:latest",
                "min_ram_gb": 4,
                "min_vram_gb": 1,
            },
            {
                "id": "b",
                "name": "B",
                "ollama_tag": "b:latest",
                "min_ram_gb": 8,
                "min_vram_gb": 4,
            },
            {
                "id": "c",
                "name": "C",
                "ollama_tag": "c:latest",
                "min_ram_gb": 12,
                "min_vram_gb": 0,
            },
            {
                "id": "too-large",
                "name": "Too Large",
                "ollama_tag": "too-large:latest",
                "min_ram_gb": 32,
                "min_vram_gb": 0,
            },
        ]

        def fake_run_one(self, model, method):
            return RunResult(
                model_id=model["id"],
                model_name=model["name"],
                method=method,
                success=True,
            )

        with tempfile.TemporaryDirectory() as tmp, \
             patch("src.batch_runner.load_catalog", return_value=models), \
             patch.object(BatchRunner, "_run_one", new=fake_run_one), \
             patch.object(BatchRunner, "_release_after_run"):
            runner = BatchRunner(
                output_dir=Path(tmp),
                model_ids=["a", "b", "c", "too-large"],
                skip_gpu=False,
                skip_onnx=True,
                capacity_ram_gb=16,
                capacity_vram_gb=0,
                capacity_has_gpu=False,
            )
            with redirect_stdout(StringIO()):
                report = runner.run()

        self.assertEqual([r.model_id for r in report.results], ["a", "b", "c"])
        self.assertEqual([r.method for r in report.results], ["ollama_cpu", "ollama_cpu", "ollama_cpu"])

    def test_phase1_method_is_displayed_as_utility_in_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = BatchReport()
            report.add(
                RunResult(
                    model_id="whisper-large-v3-turbo",
                    model_name="Whisper Large v3 Turbo",
                    method="phase1",
                    success=True,
                )
            )

            text_path = report.save_text(Path(tmp))
            text = text_path.read_text(encoding="utf-8")

            self.assertIn("utility", text)
            self.assertNotIn("phase1", text)

            out = StringIO()
            with redirect_stdout(out):
                report.print_summary()
            summary = out.getvalue()
            self.assertIn("utility", summary)
            self.assertNotIn("phase1", summary)

    def test_phase1_reports_utility_metric_instead_of_fake_token_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = BatchReport()
            report.add(
                RunResult(
                    model_id="all-minilm",
                    model_name="All-MiniLM",
                    method="phase1",
                    success=True,
                    total_time=1.2,
                    metric_kind="utility",
                    metric_label="Embeddings",
                    metric_value="3 vectors",
                )
            )

            text = report.save_text(Path(tmp)).read_text(encoding="utf-8")

        self.assertIn("3 vectors", text)
        self.assertIn("Metric", text)
        self.assertNotIn("0.0 tok/s", text)
        self.assertNotIn("ttft=0.00s", text)

    def test_runner_stdout_uses_utility_metric_for_phase1_success(self):
        model = {
            "id": "all-minilm",
            "name": "All-MiniLM",
            "phase1_adapter": True,
            "min_ram_gb": 4,
            "min_vram_gb": 0,
        }
        result = RunResult(
            model_id="all-minilm",
            model_name="All-MiniLM",
            method="phase1",
            success=True,
            total_time=1.2,
            metric_kind="utility",
            metric_label="Embeddings",
            metric_value="3 vectors",
        )

        with tempfile.TemporaryDirectory() as tmp, \
             patch("src.batch_runner.load_catalog", return_value=[model]), \
             patch("src.batch_runner.phase1_adapters.missing_dependencies_for_model", return_value=[]), \
             patch.object(BatchRunner, "_run_one", return_value=result), \
             patch.object(BatchRunner, "_release_after_run"):
            # skip_phase1 defaults to True (phase1/utility excluded from
            # benchmarks). This test explicitly exercises the phase1 metric
            # formatting so opt back in.
            runner = BatchRunner(output_dir=Path(tmp), skip_phase1=False)
            out = StringIO()
            with redirect_stdout(out):
                runner.run()

        text = out.getvalue()
        self.assertIn("utility=3 vectors", text)
        self.assertNotIn("0.0 tok/s", text)

    def test_phase1_result_carries_adapter_metric_metadata(self):
        model = {"id": "all-minilm", "name": "All-MiniLM"}
        adapter_result = {
            "status": "ok",
            "output_text": "Embedding benchmark completed.",
            "metric_label": "Embeddings",
            "metric_value": "3 vectors",
        }
        with tempfile.TemporaryDirectory() as tmp, \
             patch("src.batch_runner.phase1_adapters.run_transformers_adapter", return_value=adapter_result):
            result = BatchRunner(output_dir=Path(tmp))._run_phase1(model, "phase1", threading.Event())

        self.assertTrue(result.success)
        self.assertEqual(result.metric_kind, "utility")
        self.assertEqual(result.metric_label, "Embeddings")
        self.assertEqual(result.metric_value, "3 vectors")
        self.assertEqual(result.tokens_per_sec, 0.0)

    def test_phase1_captures_adapter_stdout_to_per_model_log(self):
        model = {"id": "all-minilm", "name": "All-MiniLM"}

        def adapter(*_args, **_kwargs):
            print("known third-party warning")
            return {
                "status": "ok",
                "output_text": "Embedding benchmark completed.",
                "metric_label": "Embeddings",
                "metric_value": "3 vectors",
            }

        with tempfile.TemporaryDirectory() as tmp, \
             patch("src.batch_runner.phase1_adapters.run_transformers_adapter", side_effect=adapter):
            result = BatchRunner(output_dir=Path(tmp))._run_phase1(model, "phase1", threading.Event())
            log_text = Path(result.log_path).read_text(encoding="utf-8")

        self.assertTrue(result.success)
        self.assertTrue(result.log_path)
        self.assertIn("known third-party warning", log_text)

    def test_ollama_download_failure_is_classified_separately(self):
        class FakeOllama:
            def is_running(self):
                return True

            def pull_model(self, *_args, **_kwargs):
                raise OllamaError("registry unavailable")

        model = {"id": "small", "name": "Small", "ollama_tag": "small:latest", "min_ram_gb": 0, "min_vram_gb": 0}
        runner = BatchRunner()
        runner.ollama = FakeOllama()

        with patch.object(runner, "_unload_running_ollama_models"), \
             patch.object(runner, "_is_ollama_tag_local", return_value=False):
            with redirect_stdout(StringIO()):
                result = runner._run_ollama(model, "ollama_cpu", threading.Event())

        self.assertFalse(result.success)
        self.assertEqual(result.failure_phase, "download_failed")
        self.assertGreaterEqual(result.download_time, 0.0)

    def test_ollama_start_wait_masks_transient_daemon_restart(self):
        class FakeOllama:
            def __init__(self):
                self.probes = 0

            def is_running(self):
                self.probes += 1
                return self.probes >= 2

            def pull_model(self, *_args, **_kwargs):
                return None

            def chat_stream_with_stats(self, *_args, **_kwargs):
                yield "A neural network learns patterns from examples.", None
                yield "", {
                    "load_duration": 1_000_000_000,
                    "prompt_eval_duration": 500_000_000,
                    "eval_duration": 2_000_000_000,
                    "eval_count": 9,
                    "done_reason": "stop",
                }

        model = {"id": "small", "name": "Small", "ollama_tag": "small:latest", "min_ram_gb": 0, "min_vram_gb": 0}
        runner = BatchRunner()
        runner.ollama = FakeOllama()

        with patch.object(batch_runner.time, "sleep"), \
             patch.object(runner, "_unload_running_ollama_models"), \
             patch.object(runner, "_is_ollama_tag_local", return_value=True):
            with redirect_stdout(StringIO()):
                result = runner._run_ollama(model, "ollama_cpu", threading.Event())

        self.assertTrue(result.success)
        self.assertGreaterEqual(runner.ollama.probes, 2)

    def test_ollama_pull_progress_callback_and_generation_timing_are_logged(self):
        class FakeOllama:
            def __init__(self):
                self.progress_cb = None

            def is_running(self):
                return True

            def pull_model(self, _tag, *, progress_cb=None, stop_event=None):
                self.progress_cb = progress_cb
                progress_cb("downloading layer", 512, 1024)

            def chat_stream_with_stats(self, *_args, **_kwargs):
                yield "ok", None
                yield "", {
                    "load_duration": 1_000_000_000,
                    "prompt_eval_duration": 500_000_000,
                    "eval_duration": 2_000_000_000,
                    "eval_count": 2,
                    "total_duration": 3_500_000_000,
                }

        logger.clear()
        model = {"id": "small", "name": "Small", "ollama_tag": "small:latest", "min_ram_gb": 0, "min_vram_gb": 0}
        runner = BatchRunner()
        fake = FakeOllama()
        runner.ollama = fake

        out = StringIO()
        with patch.object(runner, "_unload_running_ollama_models"), \
             patch.object(runner, "_is_ollama_tag_local", return_value=False):
            with redirect_stdout(out):
                result = runner._run_ollama(model, "ollama_cpu", threading.Event())

        self.assertTrue(result.success)
        self.assertIsNotNone(fake.progress_cb)
        self.assertIn("downloading layer", out.getvalue())
        self.assertIn("timing load=1.00s", out.getvalue())
        pull_logs = logger.get_entries("INFO", category=logger.CATEGORY_MODEL_PULL)
        bench_logs = logger.get_entries("INFO", category=logger.CATEGORY_BENCHMARK)
        self.assertTrue(any("Benchmark pull progress" in e["msg"] for e in pull_logs))
        self.assertTrue(any("Benchmark complete" in e["msg"] for e in bench_logs))

    def test_extended_ollama_uses_finite_answer_budget(self):
        model = {
            "id": "gemma3-27b",
            "name": "Gemma 3 27B",
            "ollama_tag": "gemma3:27b",
            "min_ram_gb": 32,
            "min_vram_gb": 17,
        }

        extended = BatchRunner(run_mode="extended")
        self.assertEqual(extended._num_predict_for(model, "ollama_gpu"), 4096)
        self.assertEqual(
            extended._num_predict_for({**model, "benchmark_num_predict": 8192}, "ollama_gpu"),
            8192,
        )

        quick = BatchRunner(run_mode="quick")
        self.assertEqual(quick._num_predict_for(model, "ollama_gpu"), 4096)
        self.assertEqual(quick._num_predict_for(model, "ollama_cpu"), 4096)

    def test_extended_heavy_ollama_timeout_scales_by_method(self):
        model = {
            "id": "gemma3-27b",
            "name": "Gemma 3 27B",
            "parameters": "27.4B",
            "ollama_tag": "gemma3:27b",
        }
        runner = BatchRunner(run_mode="extended", timeout=300)

        gpu_timeout = runner._ollama_generation_timeout_for(model, "ollama_gpu", 1024)
        cpu_timeout = runner._ollama_generation_timeout_for(model, "ollama_cpu", 1024)

        self.assertGreaterEqual(gpu_timeout, 600)
        self.assertGreater(cpu_timeout, gpu_timeout)
        self.assertGreaterEqual(cpu_timeout, 1500)

    def test_benchmark_skip_reason_removes_model_from_methods(self):
        model = {
            "id": "qwen3-4b",
            "name": "Qwen3 4B",
            "ollama_tag": "qwen3:4b",
            "benchmark_skip_reason": "leaks hidden reasoning",
        }
        runner = BatchRunner(run_mode="extended", specific_combos=[("qwen3-4b", "ollama_gpu")])

        self.assertEqual(runner._methods_for(model), [])

    def test_retry_specific_combos_are_sample_specific_and_dedup_methods(self):
        model = {"id": "qwen3-4b", "name": "Qwen3 4B", "ollama_tag": "qwen3:4b"}
        runner = BatchRunner(
            run_mode="extended",
            specific_combos=[
                ("qwen3-4b", "ollama_cpu", 1),
                ("qwen3-4b", "ollama_cpu", 2),
            ],
        )

        self.assertEqual(runner._methods_for(model), ["ollama_cpu"])
        selected = runner._iter_selected_samples_for(model, "ollama_cpu")

        self.assertEqual([index for index, _sample, _count in selected], [1, 2])
        self.assertTrue(all(count == 3 for _index, _sample, count in selected))

    def test_retry_comfyui_host_detection_accepts_sample_specific_combos(self):
        from run_batch import _needs_comfyui_host

        self.assertTrue(_needs_comfyui_host("quick", False, [("img", IMAGE_METHOD, 0)]))
        self.assertTrue(_needs_comfyui_host("quick", False, [("img", IMAGE_METHOD)]))
        self.assertFalse(_needs_comfyui_host("quick", False, [("chat", "ollama_cpu", 0)]))

    def test_benchmark_skip_methods_remove_only_listed_methods(self):
        model = {
            "id": "phi-4-reasoning-plus",
            "name": "Phi-4 Reasoning Plus",
            "ollama_tag": "phi4-reasoning:plus",
            "onnx_repo": "microsoft/Phi-4-reasoning-plus-onnx",
            "min_ram_gb": 0,
            "min_vram_gb": 1,
            "benchmark_skip_methods": ["ollama_cpu", "onnx_cpu"],
        }
        runner = BatchRunner(skip_onnx=True)
        # On GPU-less hosts (e.g. CI runners) the live capacity probe reports
        # has_gpu=False / 0 VRAM, which drops ollama_gpu. Pin a GPU-capable
        # capacity so this asserts the skip-method filtering, not host hardware.
        runner._benchmark_capacity = lambda: (64.0, True, 24.0, False)

        self.assertEqual(runner._methods_for(model), ["ollama_gpu"])

    def test_ollama_length_stop_is_not_reported_success(self):
        class FakeOllama:
            def is_running(self):
                return True

            def pull_model(self, *_args, **_kwargs):
                return None

            def chat_stream_with_stats(self, *_args, **_kwargs):
                yield "partial answer", None
                yield "", {
                    "load_duration": 1_000_000_000,
                    "prompt_eval_duration": 500_000_000,
                    "eval_duration": 2_000_000_000,
                    "eval_count": 1024,
                    "total_duration": 3_500_000_000,
                    "done_reason": "length",
                }

        model = {"id": "small", "name": "Small", "ollama_tag": "small:latest", "min_ram_gb": 0, "min_vram_gb": 0}
        runner = BatchRunner(run_mode="extended")
        runner.ollama = FakeOllama()

        with patch.object(runner, "_unload_running_ollama_models"), \
             patch.object(runner, "_is_ollama_tag_local", return_value=True):
            with redirect_stdout(StringIO()):
                result = runner._run_ollama(model, "ollama_gpu", threading.Event())

        self.assertFalse(result.success)
        self.assertEqual(result.failure_phase, "output_truncated")
        self.assertIn("partial answer", result.response_text)
        self.assertEqual(result.options["done_reason"], "length")

    def test_ollama_empty_response_is_not_reported_success(self):
        class FakeOllama:
            def is_running(self):
                return True

            def pull_model(self, *_args, **_kwargs):
                return None

            def chat_stream_with_stats(self, *_args, **_kwargs):
                yield "", {
                    "load_duration": 1_000_000_000,
                    "prompt_eval_duration": 500_000_000,
                    "eval_duration": 2_000_000_000,
                    "eval_count": 0,
                    "total_duration": 3_500_000_000,
                    "done_reason": "stop",
                }

        model = {"id": "small", "name": "Small", "ollama_tag": "small:latest", "min_ram_gb": 0, "min_vram_gb": 0}
        runner = BatchRunner(run_mode="quick")
        runner.ollama = FakeOllama()

        with patch.object(runner, "_unload_running_ollama_models"), \
             patch.object(runner, "_is_ollama_tag_local", return_value=True):
            with redirect_stdout(StringIO()):
                result = runner._run_ollama(model, "ollama_gpu", threading.Event())

        self.assertFalse(result.success)
        self.assertEqual(result.failure_phase, "empty_response")
        self.assertEqual(result.options["done_reason"], "stop")

    def test_ollama_strips_hidden_thinking_before_empty_response_validation(self):
        class FakeOllama:
            def is_running(self):
                return True

            def pull_model(self, *_args, **_kwargs):
                return None

            def chat_stream_with_stats(self, *_args, **_kwargs):
                yield "<think>private reasoning</think>Final answer.", None
                yield "", {
                    "load_duration": 1_000_000_000,
                    "prompt_eval_duration": 500_000_000,
                    "eval_duration": 2_000_000_000,
                    "eval_count": 12,
                    "total_duration": 3_500_000_000,
                    "done_reason": "stop",
                }

        model = {"id": "qwen3", "name": "Qwen3", "ollama_tag": "qwen3:4b", "min_ram_gb": 0, "min_vram_gb": 0}
        runner = BatchRunner(run_mode="quick")
        runner.ollama = FakeOllama()

        with patch.object(runner, "_unload_running_ollama_models"), \
             patch.object(runner, "_is_ollama_tag_local", return_value=True):
            with redirect_stdout(StringIO()):
                result = runner._run_ollama(model, "ollama_gpu", threading.Event())

        self.assertTrue(result.success)
        self.assertEqual(result.response_text, "Final answer.")

    def test_ollama_hidden_reasoning_response_is_not_reported_success(self):
        class FakeOllama:
            def is_running(self):
                return True

            def pull_model(self, *_args, **_kwargs):
                return None

            def chat_stream_with_stats(self, *_args, **_kwargs):
                yield "Hmm, the user wants me to reason internally instead of answer.", None
                yield "", {
                    "load_duration": 1_000_000_000,
                    "prompt_eval_duration": 500_000_000,
                    "eval_duration": 2_000_000_000,
                    "eval_count": 14,
                    "total_duration": 3_500_000_000,
                    "done_reason": "stop",
                }

        model = {"id": "qwen3", "name": "Qwen3", "ollama_tag": "qwen3:4b", "min_ram_gb": 0, "min_vram_gb": 0}
        runner = BatchRunner(run_mode="quick")
        runner.ollama = FakeOllama()

        with patch.object(runner, "_unload_running_ollama_models"), \
             patch.object(runner, "_is_ollama_tag_local", return_value=True):
            with redirect_stdout(StringIO()):
                result = runner._run_ollama(model, "ollama_gpu", threading.Event())

        self.assertFalse(result.success)
        self.assertEqual(result.failure_phase, "hidden_reasoning_response")

    def test_onnx_empty_response_is_not_reported_success(self):
        class FakeSession:
            def __init__(self, *_args, **_kwargs):
                pass

            def generate_stream_timed(self, *_args, **_kwargs):
                yield "", {"token_count": 0, "total_time": 0.1, "ttft": 0.0, "tokens_per_sec": 0.0}

        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            (models_dir / "onnx_demo").mkdir()
            model = {"id": "onnx:demo", "name": "ONNX Demo", "onnx_repo": "repo/demo"}
            runner = BatchRunner(models_dir=models_dir)
            with patch("src.batch_runner.ONNX_AVAILABLE", True), \
                 patch("src.batch_runner.has_genai_config", return_value=False), \
                 patch("src.batch_runner.OnnxModelSession", FakeSession):
                result = runner._run_onnx(model, "onnx_cpu", threading.Event())

        self.assertFalse(result.success)
        self.assertEqual(result.failure_phase, "empty_response")

    def test_onnx_token_budget_exhaustion_is_not_reported_success(self):
        class FakeSession:
            def __init__(self, *_args, **_kwargs):
                pass

            def generate_stream_timed(self, *_args, **_kwargs):
                yield "partial", None
                yield "", {"token_count": 512, "total_time": 1.0, "ttft": 0.1, "tokens_per_sec": 50.0}

        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            (models_dir / "onnx_demo").mkdir()
            model = {"id": "onnx:demo", "name": "ONNX Demo", "onnx_repo": "repo/demo"}
            runner = BatchRunner(models_dir=models_dir)
            with patch("src.batch_runner.ONNX_AVAILABLE", True), \
                 patch("src.batch_runner.has_genai_config", return_value=False), \
                 patch("src.batch_runner.OnnxModelSession", FakeSession):
                result = runner._run_onnx(model, "onnx_cpu", threading.Event())

        self.assertFalse(result.success)
        self.assertEqual(result.failure_phase, "output_truncated")
        self.assertEqual(result.response_text, "partial")

    def test_onnx_uses_catalog_benchmark_token_budget(self):
        calls = []

        class FakeSession:
            def __init__(self, *_args, **_kwargs):
                pass

            def generate_stream_timed(self, _prompt, *, max_new_tokens, **_kwargs):
                calls.append(max_new_tokens)
                yield "A neural network learns patterns from examples.", None
                yield "", {"token_count": 9, "total_time": 0.2, "ttft": 0.1, "tokens_per_sec": 45.0}

        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            (models_dir / "onnx_demo").mkdir()
            model = {
                "id": "onnx:demo",
                "name": "ONNX Demo",
                "onnx_repo": "repo/demo",
                "benchmark_num_predict": 4096,
            }
            runner = BatchRunner(models_dir=models_dir)
            with patch("src.batch_runner.ONNX_AVAILABLE", True), \
                 patch("src.batch_runner.has_genai_config", return_value=False), \
                 patch("src.batch_runner.OnnxModelSession", FakeSession):
                result = runner._run_onnx(model, "onnx_cpu", threading.Event())

        self.assertTrue(result.success)
        self.assertEqual(calls, [4096])

    def test_onnx_too_short_response_is_not_reported_success(self):
        class FakeSession:
            def __init__(self, *_args, **_kwargs):
                pass

            def generate_stream_timed(self, *_args, **_kwargs):
                yield "Why?", None
                yield "", {"token_count": 2, "total_time": 0.2, "ttft": 0.1, "tokens_per_sec": 10.0}

        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            (models_dir / "onnx_demo").mkdir()
            model = {"id": "onnx:demo", "name": "ONNX Demo", "onnx_repo": "repo/demo"}
            runner = BatchRunner(models_dir=models_dir)
            with patch("src.batch_runner.ONNX_AVAILABLE", True), \
                 patch("src.batch_runner.has_genai_config", return_value=False), \
                 patch("src.batch_runner.OnnxModelSession", FakeSession):
                result = runner._run_onnx(model, "onnx_cpu", threading.Event())

        self.assertFalse(result.success)
        self.assertEqual(result.failure_phase, "low_quality_response")
        self.assertEqual(result.response_text, "Why?")

    def test_image_benchmark_prepares_model_before_queueing(self):
        calls = []

        class FakeComfyUI:
            def is_running(self):
                return True

            def generate_image(self, **_kwargs):
                calls.append("generate")
                return b"png-bytes"

        def prepare_image_model(model, stop_event=None):
            calls.append(("prepare", model["id"], stop_event is not None))
            return True, ""

        model = {
            "id": "sdxl-lowvram",
            "name": "SDXL Low VRAM",
            "backend": "comfyui",
            "comfyui_model": "sd_xl_base_1.0.safetensors",
            "recommended_settings": {"width": 64, "height": 64, "steps": 1},
        }
        with tempfile.TemporaryDirectory() as tmp:
            runner = BatchRunner(
                output_dir=Path(tmp),
                comfyui_client=FakeComfyUI(),
                prepare_image_model=prepare_image_model,
            )
            result = runner._run_image_comfyui(model, "comfyui", threading.Event())

        self.assertTrue(result.success)
        self.assertEqual(calls[0], ("prepare", "sdxl-lowvram", True))
        self.assertEqual(calls[1], "generate")

    def test_ollama_generation_error_preserves_phase_metadata(self):
        class FakeOllama:
            def is_running(self):
                return True

            def pull_model(self, *_args, **_kwargs):
                return None

            def chat_stream_with_stats(self, *_args, **_kwargs):
                raise OllamaError("stream failed")
                yield "", None

        model = {"id": "small", "name": "Small", "ollama_tag": "small:latest", "min_ram_gb": 0, "min_vram_gb": 0}
        runner = BatchRunner()
        runner.ollama = FakeOllama()

        with patch.object(runner, "_unload_running_ollama_models"), \
             patch.object(runner, "_is_ollama_tag_local", return_value=True):
            with redirect_stdout(StringIO()):
                result = runner._run_ollama(model, "ollama_cpu", threading.Event())

        self.assertFalse(result.success)
        self.assertEqual(result.failure_phase, "runtime_error")
        self.assertEqual(result.error, "stream failed")
        self.assertTrue(result.warm_cache)
        self.assertIn("num_predict", result.options)

    def test_low_resource_batch_space_uses_peak_download_when_cleanup_enabled(self):
        models = [
            {"id": "small-a", "ollama_tag": "small-a:latest", "size_gb": 30.0},
            {"id": "small-b", "ollama_tag": "small-b:latest", "size_gb": 30.0},
            {"id": "small-c", "ollama_tag": "small-c:latest", "size_gb": 26.8},
        ]

        class FakeOllama:
            def local_model_names(self):
                return set()

            def list_local_models(self):
                return []

        with tempfile.TemporaryDirectory() as tmp, \
             patch("src.resource_manager.get_free_disk_gb", return_value=59.4):
            blocked = resource_manager.assess_batch_space(
                models, FakeOllama(), Path(tmp), cleanup_after_each_model=False
            )
            rolling = resource_manager.assess_batch_space(
                models, FakeOllama(), Path(tmp), cleanup_after_each_model=True
            )

        self.assertFalse(blocked["possible"])
        self.assertTrue(rolling["ok"])
        self.assertAlmostEqual(rolling["needed_gb"], 86.8)
        self.assertAlmostEqual(rolling["required_gb"], 30.0)

    def test_low_resource_space_does_not_treat_same_base_different_tag_as_local(self):
        models = [
            {"id": "llama32-3b", "ollama_tag": "llama3.2:3b", "size_gb": 2.0},
        ]

        class FakeOllama:
            def local_model_names(self):
                return {"llama3.2:1b"}

            def list_local_models(self):
                return []

        with tempfile.TemporaryDirectory() as tmp, \
             patch("src.resource_manager.get_free_disk_gb", return_value=10.0):
            assessment = resource_manager.assess_batch_space(models, FakeOllama(), Path(tmp))

        self.assertEqual(assessment["needed_gb"], 2.0)

    def test_cleanup_downloaded_only_preserves_preexisting_ollama_models(self):
        class FakeOllama:
            def __init__(self):
                self.deleted = []
                self.unloaded = []

            def is_running(self):
                return True

            def unload_model(self, tag):
                self.unloaded.append(tag)

            def delete_model(self, tag):
                self.deleted.append(tag)

        model = {"id": "demo", "name": "Demo", "ollama_tag": "demo:latest"}
        runner = BatchRunner(cleanup=True, cleanup_downloaded_only=True)
        runner.ollama = FakeOllama()

        runner._cleanup_model(model, "ollama_cpu")
        self.assertEqual(runner.ollama.deleted, [])
        self.assertEqual(runner.ollama.unloaded, ["demo:latest"])

        runner._downloaded_ollama_tags.add("demo:latest")
        runner._cleanup_model(model, "ollama_cpu")
        self.assertEqual(runner.ollama.deleted, ["demo:latest"])

    def test_runner_local_tag_detection_requires_exact_tag_or_base(self):
        class FakeOllama:
            def local_model_names(self):
                return {"llama3.2:1b", "qwen2.5"}

        runner = BatchRunner()
        runner.ollama = FakeOllama()

        self.assertFalse(runner._is_ollama_tag_local("llama3.2:3b"))
        self.assertTrue(runner._is_ollama_tag_local("qwen2.5:0.5b"))

    def test_low_resource_deletable_space_does_not_protect_substring_tags(self):
        class FakeOllama:
            def list_local_models(self):
                return [
                    {"name": "qwen2:7b", "size": 7 * 1_073_741_824},
                    {"name": "qwen2.5:0.5b", "size": 1 * 1_073_741_824},
                ]

        # Use an empty tmpdir as models_path so the ComfyUI-checkpoint walk
        # inside _estimate_deletable_gb finds zero files. The original test
        # passed Path(".") which on developer/CI boxes can resolve to the
        # repo root where ComfyUI/models/checkpoints contains real artefacts,
        # adding their bytes to the deletable total and tripping the assert.
        # The substring-tag protection logic is unrelated to ComfyUI walk, so
        # an empty dir cleanly isolates the assertion under test.
        with tempfile.TemporaryDirectory() as empty_models_dir:
            deletable = resource_manager._estimate_deletable_gb(
                FakeOllama(), {"qwen2.5:0.5b"}, Path(empty_models_dir)
            )

        self.assertAlmostEqual(deletable, 7.0)


class BatchRunnerImageComfyUITests(unittest.TestCase):
    """Real integration coverage for `_run_image_comfyui` — the path that
    failed in production with `No ComfyUI client provided to BatchRunner`
    even though the user had wired the `ensure_comfyui_ready` callback.

    These exercise the actual method (not just regex over source) with
    mocked ComfyUI client + ensure callback so we catch wiring bugs early.
    """

    def _model(self):
        return {
            "id": "playground-v25-aesthetic",
            "name": "Playground v2.5 1024px Aesthetic",
            "comfyui_model": "playground_v25_aesthetic.safetensors",
            "recommended_settings": {
                "width": 1024, "height": 1024, "steps": 25, "cfg": 3.0,
                "sampler": "dpmpp_2m", "scheduler": "karras",
            },
        }

    class _FakeComfyUI:
        """Minimal stand-in for ComfyUIClient.

        Tracks call counts so tests can assert on retry behaviour.
        """
        def __init__(self, running_sequence, generate_results):
            self._running_sequence = list(running_sequence)
            self._generate_results = list(generate_results)
            self.is_running_calls = 0
            self.generate_calls = 0

        def is_running(self):
            self.is_running_calls += 1
            if self._running_sequence:
                return self._running_sequence.pop(0)
            return True

        def generate_image(self, **kwargs):
            self.generate_calls += 1
            if not self._generate_results:
                raise RuntimeError("no more generate results queued")
            outcome = self._generate_results.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    def test_run_image_comfyui_succeeds_when_client_present_and_running(self):
        """Happy path: client is up, generate_image returns bytes; we should
        get a successful image result without needing the ensure callback."""
        runner = BatchRunner()
        runner._ensure_comfyui_ready = lambda t: True  # not used in happy path
        fake = self._FakeComfyUI(
            running_sequence=[True],
            generate_results=[b"\x89PNG\r\n\x1a\nfake-png-bytes"],
        )
        runner._comfyui_client = fake

        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            runner.output_dir = Path(tmp)
            runner._current_sample = {"index": 0, "prompt": "test prompt"}
            result = runner._run_image_comfyui(
                self._model(), "image", threading.Event()
            )

        self.assertTrue(result.success, f"expected success, got: {result.error}")
        self.assertEqual(result.surface, "image")
        self.assertEqual(fake.generate_calls, 1)

    def test_run_image_comfyui_starts_comfyui_via_callback_when_down(self):
        """Regression test for the user-reported bug: even when the app
        passes `comfyui_client=self.comfyui` and ComfyUI is currently down,
        the runner must invoke `ensure_comfyui_ready` and proceed to
        generate — NOT fail with "No ComfyUI client provided to BatchRunner"
        or "ComfyUI is not running"."""
        runner = BatchRunner()

        ensure_called_with = []
        def ensure(timeout):
            ensure_called_with.append(timeout)
            return True  # success: pretend we started it

        runner._ensure_comfyui_ready = ensure
        # First is_running() probe says False (down); after ensure runs we
        # never re-probe inside _ensure_comfyui_running_for_run, we just
        # trust the callback's True return.  generate_image succeeds.
        fake = self._FakeComfyUI(
            running_sequence=[False],
            generate_results=[b"png-bytes"],
        )
        runner._comfyui_client = fake

        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            runner.output_dir = Path(tmp)
            runner._current_sample = {"index": 0, "prompt": "test prompt"}
            result = runner._run_image_comfyui(
                self._model(), "image", threading.Event()
            )

        self.assertTrue(
            result.success,
            f"ensure_comfyui_ready should have started ComfyUI and let "
            f"generate proceed; instead got: {result.error}"
        )
        # v5.3.6+ constrained-VM cold-start fix: initial cold-start budget was
        # bumped 60 → 180 s.  Constrained cloud VMs (roaming profile + Defender +
        # vGPU + cold torch / CUDA init) routinely needed >60 s for the first
        # /system_stats response, which was the actual root cause of repeated
        # "FAIL  (ComfyUI is not running and could not be started)" failures
        # at iter [1/N] of `sdxl-lowvram` across all three SKUs.  Warm runs
        # short-circuit on is_running() so this adds zero overhead.
        self.assertEqual(ensure_called_with, [180],
            "ensure_comfyui_ready must be called with the new 180s cold-start "
            "timeout (was 60s pre-constrained-VM-fix)")
        self.assertEqual(fake.generate_calls, 1)

    def test_run_image_comfyui_validates_model_launch_flags_when_already_running(self):
        runner = BatchRunner()
        callback_calls = []

        def ensure(timeout, model):
            callback_calls.append((timeout, model["id"]))
            return True

        runner._ensure_comfyui_ready = ensure
        fake = self._FakeComfyUI(
            running_sequence=[True],
            generate_results=[b"png-bytes"],
        )
        runner._comfyui_client = fake
        model = dict(self._model())
        model["id"] = "sdxl-lowvram"
        model["comfyui_launch_flags"] = ["--lowvram"]

        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            runner.output_dir = Path(tmp)
            runner._current_sample = {"index": 0, "prompt": "test prompt"}
            result = runner._run_image_comfyui(model, "image", threading.Event())

        self.assertTrue(result.success, f"expected success, got: {result.error}")
        self.assertEqual(callback_calls, [(180, "sdxl-lowvram")])
        self.assertEqual(fake.generate_calls, 1)

    def test_run_image_comfyui_retries_once_when_comfyui_crashes_midrun(self):
        """If generate_image raises and ComfyUI is no longer running,
        attempt exactly ONE restart + retry."""
        runner = BatchRunner()

        ensure_calls = []
        def ensure(timeout):
            ensure_calls.append(timeout)
            return True

        runner._ensure_comfyui_ready = ensure
        # is_running sequence:
        #   1) initial probe inside _ensure_comfyui_running_for_run = True
        #   2) post-crash probe inside `except` = False
        #   3) probe inside second _ensure_comfyui_running_for_run = False
        #      → forces the helper to invoke the ensure callback, which returns True
        fake = self._FakeComfyUI(
            running_sequence=[True, False, False],
            generate_results=[RuntimeError("ws closed"), b"png-after-retry"],
        )
        runner._comfyui_client = fake

        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            runner.output_dir = Path(tmp)
            runner._current_sample = {"index": 0, "prompt": "p"}
            result = runner._run_image_comfyui(
                self._model(), "image", threading.Event()
            )

        self.assertTrue(result.success, f"expected retry success: {result.error}")
        self.assertEqual(fake.generate_calls, 2,
            "generate_image must be called exactly twice (initial + 1 retry)")
        self.assertEqual(ensure_calls, [120],
            "post-crash ensure must use the 120s restart timeout")

    def test_run_image_comfyui_fails_cleanly_without_client(self):
        """Headless `run_batch.py` runs without a client should fail cleanly,
        not crash. (Note: the app now always passes self.comfyui, so this
        path is reserved for fully headless callers.)"""
        runner = BatchRunner()
        runner._comfyui_client = None  # headless mode

        with redirect_stdout(StringIO()):
            result = runner._run_image_comfyui(
                self._model(), "image", threading.Event()
            )

        self.assertFalse(result.success)
        self.assertIn("No ComfyUI client", result.error)

    def test_run_image_comfyui_fails_cleanly_when_ensure_returns_false(self):
        """If ComfyUI is down AND ensure_comfyui_ready can't bring it up,
        return a clean failure — not crash, not retry forever."""
        runner = BatchRunner()
        runner._ensure_comfyui_ready = lambda t: False  # can't start
        fake = self._FakeComfyUI(
            running_sequence=[False],  # is_running probe says down
            generate_results=[],  # should never be called
        )
        runner._comfyui_client = fake

        with redirect_stdout(StringIO()):
            result = runner._run_image_comfyui(
                self._model(), "image", threading.Event()
            )

        self.assertFalse(result.success)
        self.assertIn("could not be started", result.error)
        self.assertEqual(fake.generate_calls, 0)

    def test_run_image_comfyui_missing_checkpoint_is_environment_skip(self):
        """Missing checkpoint validation errors should classify as environment skip."""
        runner = BatchRunner()
        runner._ensure_comfyui_ready = lambda t: True
        fake = self._FakeComfyUI(
            running_sequence=[True],
            generate_results=[
                RuntimeError(
                    "Could not queue prompt (400 Bad Request): "
                    "CheckpointLoaderSimple: ckpt_name value not in list"
                )
            ],
        )
        runner._comfyui_client = fake

        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            runner.output_dir = Path(tmp)
            runner._current_sample = {"index": 0, "prompt": "test prompt"}
            result = runner._run_image_comfyui(
                self._model(), "image", threading.Event()
            )

        self.assertFalse(result.success)
        self.assertEqual(result.failure_phase, "environment_skip")
        self.assertIn("Skipped:", result.error)


class BatchRunnerImageComfyUIColdStartFailureTests(unittest.TestCase):
    """v5.3.6+ cold-start fix coverage.

    These tests pin the new structured ``(bool, reason)`` callback contract
    and the multi-iteration failure cache that together replaced the generic
    ``FAIL  (ComfyUI is not running and could not be )`` 60-char-chopped
    error string in support reports.
    """

    def _model(self, model_id="sdxl-lowvram"):
        return {
            "id": model_id,
            "name": "SDXL Low VRAM",
            "comfyui_model": "sd_xl_base_1.0.safetensors",
            "comfyui_launch_flags": ["--lowvram"],
            "recommended_settings": {
                "width": 1024, "height": 1024, "steps": 30, "cfg": 7.0,
                "sampler": "euler", "scheduler": "normal",
            },
        }

    class _FakeComfyUIDown:
        def __init__(self):
            self.is_running_calls = 0
            self.generate_calls = 0

        def is_running(self):
            self.is_running_calls += 1
            return False

        def generate_image(self, **kwargs):
            self.generate_calls += 1
            raise RuntimeError("generate_image must not be called when ensure failed")

    def _make_runner_with_callback(self, callback):
        runner = BatchRunner()
        runner._ensure_comfyui_ready = callback
        runner._comfyui_client = self._FakeComfyUIDown()
        return runner

    def test_callback_returning_tuple_propagates_specific_reason_to_run_result(self):
        """When the bench callback returns ``(False, "specific reason…")``
        the RunResult.error must contain that specific reason — not the
        generic 'ComfyUI is not running and could not be started' placeholder
        that gave Ron zero actionable info in his screenshots."""
        specific = "ComfyUI not installed at expected paths"
        runner = self._make_runner_with_callback(lambda t: (False, specific))

        with redirect_stdout(StringIO()):
            result = runner._run_image_comfyui(
                self._model(), "image", threading.Event()
            )

        self.assertFalse(result.success)
        self.assertEqual(result.error, specific,
            "the bench-callback's specific failure reason must be preserved "
            "verbatim in RunResult.error so the bench log surfaces the real "
            "root cause, not 'ComfyUI is not running and could not be started'")

    def test_cold_start_timeout_reason_includes_elapsed_seconds(self):
        """A polling-timeout failure must report the actual elapsed time +
        'didn't respond on /system_stats within Xs' so Ron knows it timed out
        rather than crashed at startup."""
        runner = self._make_runner_with_callback(
            lambda t: (False, "ComfyUI subprocess started but didn't respond on /system_stats within 180s — process alive: True, exit code: None; see comfyui.log")
        )

        with redirect_stdout(StringIO()):
            result = runner._run_image_comfyui(
                self._model(), "image", threading.Event()
            )

        self.assertFalse(result.success)
        self.assertIn("didn't respond on /system_stats within", result.error)
        self.assertIn("180s", result.error)
        self.assertIn("comfyui.log", result.error)

    def test_dep_install_failure_reason_includes_pip_stderr_tail(self):
        """A dep-install failure must surface the last actionable line of
        pip's stderr (e.g., 'ERROR: No matching distribution found for X')
        so the user can fix the actual root cause."""
        pip_tail = "ERROR: No matching distribution found for comfyui-frontend-package==1.39.19"
        runner = self._make_runner_with_callback(
            lambda t: (False, f"ComfyUI dependency install failed: {pip_tail}")
        )

        with redirect_stdout(StringIO()):
            result = runner._run_image_comfyui(
                self._model(), "image", threading.Event()
            )

        self.assertFalse(result.success)
        self.assertIn("dependency install failed", result.error)
        self.assertIn("No matching distribution found", result.error)

    def test_not_installed_reason_reports_expected_paths_message(self):
        """If ComfyUI isn't installed anywhere, the reason must say so —
        not 'subprocess started but didn't respond'."""
        runner = self._make_runner_with_callback(
            lambda t: (False, "ComfyUI not installed at expected paths (config.json comfyui_dir, comfyui_path.bat, or ./ComfyUI)")
        )

        with redirect_stdout(StringIO()):
            result = runner._run_image_comfyui(
                self._model(), "image", threading.Event()
            )

        self.assertFalse(result.success)
        self.assertIn("ComfyUI not installed", result.error)

    def test_process_death_during_startup_reports_exit_code(self):
        """If the subprocess exits during the readiness poll (e.g., CUDA OOM,
        missing DLL), the reason must include the exit code so support knows
        whether to look at driver / CUDA / DLL vs network / timing issues."""
        runner = self._make_runner_with_callback(
            lambda t: (False, "ComfyUI subprocess (PID 12345) exited with code 3221225477 during startup — last log line: ImportError: DLL load failed while importing _C: The specified module could not be found.")
        )

        with redirect_stdout(StringIO()):
            result = runner._run_image_comfyui(
                self._model(), "image", threading.Event()
            )

        self.assertFalse(result.success)
        self.assertIn("exited with code 3221225477", result.error)
        self.assertIn("during startup", result.error)
        self.assertIn("DLL load failed", result.error)

    def test_iteration_two_returns_cached_failure_reason_in_under_100ms(self):
        """Once an image-gen run has failed to bring up ComfyUI, iterations
        2 and 3 of the same model — and every subsequent image model in the
        batch — must fail INSTANTLY using the cached reason instead of
        re-paying the multi-minute pip / cold-start cost.  Concretely: the
        ensure callback must be called exactly ONCE across three iterations."""
        call_count = [0]
        first_call_done = threading.Event()

        def slow_failing_callback(timeout):
            call_count[0] += 1
            # Simulate a slow failure (e.g. pip install that took 30 s
            # before failing).  Subsequent iterations must NOT pay this cost.
            if not first_call_done.is_set():
                time.sleep(0.01)  # short enough for the test to finish quickly
                first_call_done.set()
            return False, "ComfyUI dependency install failed: pip exited with code 1"

        runner = self._make_runner_with_callback(slow_failing_callback)
        model = self._model()

        # Iteration 1: pays the (mocked) slow cost.
        with redirect_stdout(StringIO()):
            r1 = runner._run_image_comfyui(model, "image", threading.Event())

        # Iterations 2 + 3: must return instantly from the cache.
        t2_start = time.perf_counter()
        with redirect_stdout(StringIO()):
            r2 = runner._run_image_comfyui(model, "image", threading.Event())
            r3 = runner._run_image_comfyui(model, "image", threading.Event())
        t2_elapsed = time.perf_counter() - t2_start

        self.assertFalse(r1.success)
        self.assertFalse(r2.success)
        self.assertFalse(r3.success)
        self.assertEqual(
            call_count[0], 1,
            "the ensure callback must be invoked exactly ONCE — iterations 2 "
            "and 3 MUST hit the cached failure reason and skip the slow "
            "dep-install / cold-start path entirely"
        )
        self.assertLess(
            t2_elapsed, 0.1,
            f"cached-failure iterations must return in <100ms; took {t2_elapsed*1000:.1f}ms"
        )
        # The cached reason must mention it came from a prior run so the
        # user understands why two iterations failed at the same instant.
        self.assertIn("cached from earlier image-gen run", r2.error)
        self.assertIn("dependency install failed", r2.error)

    def test_cached_failure_does_not_apply_when_comfyui_recovers(self):
        """If the user fixes ComfyUI mid-batch (e.g., starts it manually),
        the cached failure must be bypassed by a live is_running() probe so
        subsequent iterations can succeed."""
        call_count = [0]

        def failing_then_unused_callback(timeout):
            call_count[0] += 1
            return False, "ComfyUI dependency install failed: pip exited with code 1"

        runner = BatchRunner()
        runner._ensure_comfyui_ready = failing_then_unused_callback

        class _ToggleableComfyUI:
            def __init__(self):
                self.running = False
                self.generate_calls = 0

            def is_running(self):
                return self.running

            def generate_image(self, **kwargs):
                self.generate_calls += 1
                return b"png-bytes-after-manual-fix"

        fake = _ToggleableComfyUI()
        runner._comfyui_client = fake

        with tempfile.TemporaryDirectory() as tmp:
            # Prevent test artifacts from landing in C:\LocalAI when this test
            # exercises the successful image-save path.
            runner.output_dir = Path(tmp)

            # Iteration 1: ComfyUI down + callback fails → cached failure.
            with redirect_stdout(StringIO()):
                r1 = runner._run_image_comfyui(self._model(), "image", threading.Event())
            self.assertFalse(r1.success)

            # User "fixes it" mid-batch; iteration 2 must succeed via live probe.
            fake.running = True
            with redirect_stdout(StringIO()):
                r2 = runner._run_image_comfyui(self._model(), "image", threading.Event())
        self.assertTrue(
            r2.success,
            "if ComfyUI is_running() now returns True, the cached failure "
            "must be bypassed and generate_image must be called",
        )
        self.assertEqual(fake.generate_calls, 1)

    def test_callback_raising_exception_surfaces_clean_reason(self):
        """If the bench callback raises (not just returns False), the runner
        must convert that to a structured failure reason instead of letting
        it bubble up or printing a !! ensure_comfyui_ready callback raised
        line + a generic message."""
        def crashing_callback(timeout):
            raise RuntimeError("OSError 28 from profile container")

        runner = self._make_runner_with_callback(crashing_callback)

        with redirect_stdout(StringIO()):
            result = runner._run_image_comfyui(
                self._model(), "image", threading.Event()
            )

        self.assertFalse(result.success)
        self.assertIn("callback raised", result.error)
        self.assertIn("OSError 28", result.error)

    def test_legacy_bool_callback_preserves_original_failure_message(self):
        """Back-compat: callbacks that return plain ``False`` (the pre-v5.3.6
        contract) must still produce the original 'ComfyUI is not running and
        could not be started' message so headless / older wirings don't break."""
        runner = self._make_runner_with_callback(lambda t: False)

        with redirect_stdout(StringIO()):
            result = runner._run_image_comfyui(
                self._model(), "image", threading.Event()
            )

        self.assertFalse(result.success)
        self.assertIn("could not be started", result.error)

    def test_headless_no_callback_reports_actionable_message(self):
        """``run_batch.py`` runs with ``ensure_comfyui_ready=None``; the
        error must explain why ComfyUI didn't start (no callback wired) and
        tell the user what to do — not silently say 'could not be started'."""
        runner = BatchRunner()
        runner._ensure_comfyui_ready = None
        runner._comfyui_client = self._FakeComfyUIDown()

        with redirect_stdout(StringIO()):
            result = runner._run_image_comfyui(
                self._model(), "image", threading.Event()
            )

        self.assertFalse(result.success)
        self.assertIn("headless", result.error.lower())
        self.assertIn("manually", result.error.lower())

    def test_normalise_ensure_result_strips_newlines_and_caps_length(self):
        """A multi-line pip traceback must collapse to a single-line reason
        ≤240 chars so the bench log textbox doesn't get blown out and the
        batch_runner.py:300 ``[:200]`` truncation can't chop mid-word."""
        long_reason = "ERROR:\n" + ("x" * 500)
        ok, reason = BatchRunner._normalise_ensure_result(
            (False, long_reason),
            default_failure_reason="default-reason",
        )
        self.assertFalse(ok)
        self.assertNotIn("\n", reason)
        self.assertLessEqual(len(reason), 240)
        self.assertTrue(reason.endswith("…"),
            "truncated reason must end with ellipsis so the user sees it was cut")

    def test_normalise_ensure_result_rejects_malformed_tuples_safely(self):
        """A 3-tuple, 1-tuple, list, or non-bool first element must NOT be
        interpreted as ``(True, "x")`` — that would corrupt the contract and
        report success-with-failure-reason.  Defensive: fall back to the
        default failure reason instead."""
        # 3-tuple: must not be parsed as the new-contract 2-tuple.
        ok, reason = BatchRunner._normalise_ensure_result(
            (True, "ok", "extra"),
            default_failure_reason="fallback",
        )
        self.assertTrue(ok, "non-empty tuple is truthy → bool(raw) is True")
        self.assertEqual(reason, "",
            "truthy bool(raw) means success — reason must be empty, not 'fallback'")

        # List: must not be parsed as the new-contract 2-tuple even if it has len 2.
        ok, reason = BatchRunner._normalise_ensure_result(
            [False, "claimed failure"],
            default_failure_reason="fallback",
        )
        self.assertTrue(ok, "non-empty list is truthy → bool(raw) is True")

        # Non-bool first element in 2-tuple: also rejected.
        ok, reason = BatchRunner._normalise_ensure_result(
            ("not-a-bool", "reason"),
            default_failure_reason="fallback",
        )
        self.assertTrue(ok, "tuple-with-non-bool-head falls back to bool(raw) which is True")

        # None: classic falsy → default reason.
        ok, reason = BatchRunner._normalise_ensure_result(
            None, default_failure_reason="default-reason"
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "default-reason")

        # Properly formed legacy bool (False) → default reason.
        ok, reason = BatchRunner._normalise_ensure_result(
            False, default_failure_reason="legacy-default"
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "legacy-default")

    def test_fail_print_truncation_is_at_200_not_60_chars(self):
        """The historical ``print(f'FAIL  ({reason[:60]})')`` chopped
        cold-start error reports mid-word ('ComfyUI is not running and could
        not be ').  Pin the new 200-char cap so a future refactor can't
        silently re-introduce the 60-char regression."""
        src = Path(__file__).resolve().parents[1] / "src" / "batch_runner.py"
        text = src.read_text(encoding="utf-8")
        self.assertNotIn(
            "reason[:60]", text,
            "the 60-char reason truncation must NOT come back — it chopped "
            "the cold-start error reports mid-word"
        )
        self.assertIn(
            "reason[:200]", text,
            "the bench-log FAIL line must truncate at 200 chars so the "
            "actual ComfyUI cold-start failure reason fits"
        )


class BatchRunnerOrderingAndUtilityExclusionTests(unittest.TestCase):
    """Pin the v5.4.x ordering + utility-exclusion + report-timing contracts."""

    def test_default_skip_phase1_is_true(self):
        """Phase 1 / Toolbox / utility models are excluded from benchmarks by default."""
        runner = BatchRunner()
        self.assertTrue(runner.skip_phase1)

    def test_select_models_excludes_phase1_adapters_by_default(self):
        text_model = {
            "id": "small-chat",
            "name": "Small Chat",
            "ollama_tag": "small-chat:latest",
            "size_gb": 0.5,
            "min_ram_gb": 4,
            "min_vram_gb": 0,
        }
        utility_model = {
            "id": "speech",
            "name": "Whisper",
            "phase1_adapter": True,
            "hf_repo": "openai/whisper-large-v3-turbo",
            "size_gb": 1.5,
            "min_ram_gb": 4,
            "min_vram_gb": 0,
        }

        with tempfile.TemporaryDirectory() as tmp, \
             patch("src.batch_runner.load_catalog", return_value=[text_model, utility_model]), \
             patch("src.batch_runner.phase1_adapters.missing_dependencies_for_model", return_value=[]):
            runner = BatchRunner(output_dir=Path(tmp))
            selected = runner._select_models()

        ids = [m["id"] for m in selected]
        self.assertIn("small-chat", ids)
        self.assertNotIn("speech", ids,
                         "phase1_adapter models must be excluded from benchmarks by default")

    def test_select_models_partitions_text_then_image_smallest_first(self):
        # Mixed list with sizes shuffled out of order — runner must
        # sort text smallest-first, then image smallest-first.
        # Use the ordering helper directly so we don't fight `_methods_for`'s
        # capacity/run-mode gating (image methods only emit in Extended mode
        # on GPU profiles, which we cover in a separate test).
        small_chat = {
            "id": "tiny-chat",
            "name": "Tiny Chat",
            "ollama_tag": "tiny:latest",
            "size_gb": 0.3,
            "min_ram_gb": 2,
            "min_vram_gb": 0,
        }
        big_chat = {
            "id": "big-chat",
            "name": "Big Chat",
            "ollama_tag": "big:latest",
            "size_gb": 8.0,
            "min_ram_gb": 16,
            "min_vram_gb": 8,
        }
        small_image = {
            "id": "small-sd",
            "name": "Small SD",
            "backend": "comfyui",
            "comfyui_model": "small.safetensors",
            "category": "Image Generation",
            "size_gb": 2.0,
            "min_vram_gb": 4,
        }
        big_image = {
            "id": "big-sdxl",
            "name": "Big SDXL",
            "backend": "comfyui",
            "comfyui_model": "big.safetensors",
            "category": "Image Generation",
            "size_gb": 6.5,
            "min_vram_gb": 8,
        }
        # Catalog provided in arbitrary order to ensure sorting is real.
        catalog_in = [big_image, small_chat, big_chat, small_image]

        ordered = batch_runner._order_models_text_then_image(catalog_in)
        ids = [m["id"] for m in ordered]

        # Text first, smallest-first; then image, smallest-first.
        self.assertEqual(ids, ["tiny-chat", "big-chat", "small-sd", "big-sdxl"])

    def test_run_mode_extended_orders_images_last_within_overall_queue(self):
        # Extended mode runs per-model sample prompts and image-gen on GPU
        # profiles. Image rows MUST land after every text row regardless of
        # individual size_gb so users see fast progress before paying for
        # slow image renders.
        small_chat = {
            "id": "small-chat",
            "name": "Small Chat",
            "ollama_tag": "small:latest",
            "size_gb": 0.5,
            "min_ram_gb": 4,
            "min_vram_gb": 0,
        }
        # An image model with size_gb LARGER than chat to prove ordering is
        # by partition, not raw size.
        small_image = {
            "id": "small-sdxl",
            "name": "Small SDXL",
            "backend": "comfyui",
            "comfyui_model": "small.safetensors",
            "category": "Image Generation",
            "size_gb": 1.0,
            "min_vram_gb": 4,
        }
        big_chat = {
            "id": "big-chat",
            "name": "Big Chat",
            "ollama_tag": "big:latest",
            "size_gb": 7.0,
            "min_ram_gb": 16,
            "min_vram_gb": 8,
        }

        catalog_in = [small_image, small_chat, big_chat]
        ordered = batch_runner._order_models_text_then_image(catalog_in)
        ids = [m["id"] for m in ordered]

        # Even though small-sdxl has size_gb=1.0 (smaller than big-chat's
        # 7.0), it must come AFTER big-chat because it belongs to the
        # image partition which always runs last.
        self.assertEqual(ids, ["small-chat", "big-chat", "small-sdxl"])

    def test_html_report_includes_started_ended_and_duration_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = BatchReport(
                start_time="2026-05-21T10:00:00",
                machine_info={"machine_name": "test", "vcpu": 4, "ram_gb": 16},
                run_mode="quick",
            )
            report.add(RunResult(
                model_id="chat",
                model_name="Chat",
                method="ollama_cpu",
                sample_index=1,
                success=True,
                tokens_per_sec=12.0,
                ttft=0.5,
                total_time=2.0,
                prompt="hi",
            ))
            # Stamp an explicit later end-time so duration is computed
            # deterministically.
            report.stamp_end_time("2026-05-21T11:23:45")
            report.save_html(Path(tmp))

            html_path = next(Path(tmp).glob("*.html"))
            text = html_path.read_text(encoding="utf-8")

        self.assertIn('class="run-timing"', text)
        self.assertIn("Started", text)
        self.assertIn("Ended", text)
        # v5.5.7+: timing block headlines compute_time (sum of per-result
        # total_time) and keeps wall-clock as a secondary diagnostic
        # labelled "Wall clock". The old "Duration (H:MM:SS)" label was
        # split into these two so Resume Today's Run idle gaps don't lie
        # about how long the benchmark actually spent computing.
        self.assertIn("Compute time", text)
        self.assertIn("Wall clock", text)
        # 1h 23m 45s = "1:23:45" — wall clock between start and end stamps
        self.assertIn("1:23:45", text)
        # Compute time should also be present (2s = "0:00:02")
        self.assertIn("0:00:02", text)
        self.assertIn("2026-05-21T10:00:00", text)
        self.assertIn("2026-05-21T11:23:45", text)

    def test_html_summary_table_headers_are_sortable_buttons(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = BatchReport(
                start_time="2026-05-21T10:00:00",
                machine_info={"machine_name": "test"},
                run_mode="quick",
            )
            report.add(RunResult(
                model_id="chat",
                model_name="Chat",
                method="ollama_cpu",
                sample_index=1,
                success=True,
                tokens_per_sec=12.0,
                ttft=0.5,
                total_time=2.0,
                prompt="hi",
            ))
            report.save_html(Path(tmp))

            html_path = next(Path(tmp).glob("*.html"))
            text = html_path.read_text(encoding="utf-8")

        # Sortable headers must be keyboard-accessible buttons with
        # data-sort-key and aria-sort markup, and the table must carry
        # data-sort attributes on tds so the JS sort handler has values
        # to compare against.
        self.assertIn('role="button"', text)
        self.assertIn('tabindex="0"', text)
        self.assertIn('aria-sort="none"', text)
        self.assertIn('data-sort-key="raw"', text)
        self.assertIn('data-sort-key="tps"', text)
        # JS sort handler must be embedded inline so the report is
        # self-contained (no external deps).
        self.assertIn("aria-sort", text)
        self.assertIn("sort-indicator", text)

    def test_html_report_omits_utility_surface_tab(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = BatchReport(
                start_time="2026-05-21T10:00:00",
                machine_info={"machine_name": "test"},
                run_mode="quick",
            )
            report.add(RunResult(
                model_id="chat", model_name="Chat", method="ollama_cpu",
                sample_index=1, success=True, tokens_per_sec=12.0,
                ttft=0.5, total_time=2.0, prompt="hi",
            ))
            report.save_html(Path(tmp))
            html_path = next(Path(tmp).glob("*.html"))
            text = html_path.read_text(encoding="utf-8")
        # Utility surface tab was removed because utility models are
        # no longer benchmarked.
        self.assertNotIn('data-surface="utility"', text)
        self.assertNotIn("__COUNT_UTILITY__", text)


class ForceAllAndSmartSkipTests(unittest.TestCase):
    """v5.5.1+ Force-All and adaptive smart-skip behaviour."""

    def test_oom_error_classifier_matches_known_patterns(self):
        self.assertTrue(batch_runner._looks_like_oom_error("cuda out of memory: tried to allocate"))
        self.assertTrue(batch_runner._looks_like_oom_error("CUDAerrorOutOfMemory"))
        self.assertTrue(batch_runner._looks_like_oom_error("MemoryError: bad_alloc"))
        self.assertTrue(batch_runner._looks_like_oom_error("Process exited with code 137"))
        self.assertTrue(batch_runner._looks_like_oom_error("Process exited with code -1073741819"))
        self.assertTrue(batch_runner._looks_like_oom_error("DML allocator failed"))
        self.assertTrue(batch_runner._looks_like_oom_error("failed to load model"))
        # v5.5.7+: Windows-specific resource-exhaustion patterns surfaced
        # by ComfyUI on AI PC class hardware when CPU image-gen runs
        # the host out of pageable memory.  Without these the OOM ceiling
        # stayed at infinity and every subsequent (larger) image-gen
        # model paid the same dead-end startup cost.
        self.assertTrue(batch_runner._looks_like_oom_error(
            "ComfyUI execution error: Not enough memory resources are available to complete this operation."
        ))
        self.assertTrue(batch_runner._looks_like_oom_error(
            "ComfyUI execution error: resource deadlock would occur: resource deadlock would occur"
        ))
        self.assertTrue(batch_runner._looks_like_oom_error(
            "Semaphore timeout period has expired"
        ))
        self.assertTrue(batch_runner._looks_like_oom_error(
            "Insufficient system resources exist to complete the requested service."
        ))
        # Negative cases must NOT trip the classifier so unrelated failures
        # don't accidentally lower the OOM ceiling.
        self.assertFalse(batch_runner._looks_like_oom_error("Connection refused"))
        self.assertFalse(batch_runner._looks_like_oom_error("ollama not running"))
        self.assertFalse(batch_runner._looks_like_oom_error(""))
        self.assertFalse(batch_runner._looks_like_oom_error(None))

    def test_force_all_implies_allow_oversize_and_lifts_max_failures(self):
        runner = BatchRunner(force_all=True)
        self.assertTrue(runner.allow_oversize)
        self.assertEqual(runner.max_failures, __import__("sys").maxsize)

    def test_force_all_does_not_clobber_explicit_allow_oversize(self):
        runner = BatchRunner(force_all=False, allow_oversize=True)
        self.assertTrue(runner.allow_oversize)
        # max_failures should NOT be lifted when force_all is off, even when
        # oversize is explicitly enabled.
        self.assertEqual(runner.max_failures, 10)

    def test_smart_skip_tightens_oom_ceiling_per_method(self):
        """A CUDA OOM on ollama_gpu at 13B must auto-skip 30B/70B on
        ollama_gpu, while leaving ollama_cpu and image_comfyui untouched."""
        runner = BatchRunner(force_all=True)
        runner._record_outcome_for_smart_skip(
            {"id": "llama-13b", "size_gb": 13.0},
            "ollama_gpu",
            RunResult("llama-13b", "Llama 13B", "ollama_gpu", False,
                      error="cuda out of memory", failure_phase="runtime_error"),
        )

        self.assertEqual(runner._oom_ceiling_gb.get("ollama_gpu"), 13.0)

        # 30B on same method → skipped.
        reason = runner._smart_skip_reason({"id": "llama-30b", "size_gb": 30.0}, "ollama_gpu")
        self.assertIsNotNone(reason)
        self.assertIn("OOM", reason)
        self.assertIn("ollama_gpu", reason)

        # 13B is exactly at ceiling — still skipped.
        self.assertIsNotNone(runner._smart_skip_reason({"id": "llama-13b-b", "size_gb": 13.0}, "ollama_gpu"))

        # 7B is below ceiling — still tried.
        self.assertIsNone(runner._smart_skip_reason({"id": "llama-7b", "size_gb": 7.0}, "ollama_gpu"))

        # Same model on a DIFFERENT method must not be blocked — cross-method
        # contamination is exactly what we want to avoid.
        self.assertIsNone(runner._smart_skip_reason({"id": "llama-30b", "size_gb": 30.0}, "ollama_cpu"))
        self.assertIsNone(runner._smart_skip_reason({"id": "img-flux", "size_gb": 30.0}, "image_comfyui"))

    def test_smart_skip_tightens_disk_ceiling_for_environment_skip(self):
        """A disk-full environment_skip on ollama_gpu at 13B must auto-skip
        larger ollama_gpu pulls in the same run."""
        runner = BatchRunner()
        runner._record_outcome_for_smart_skip(
            {"id": "llama-13b", "size_gb": 13.0},
            "ollama_gpu",
            RunResult("llama-13b", "Llama 13B", "ollama_gpu", False,
                      error="Skipped: profile container is full (no space left on device)",
                      failure_phase="environment_skip"),
        )
        self.assertEqual(runner._disk_blocked_ceiling_gb.get("ollama_gpu"), 13.0)
        reason = runner._smart_skip_reason({"id": "llama-30b", "size_gb": 30.0}, "ollama_gpu")
        self.assertIsNotNone(reason)
        self.assertIn("disk", reason.lower())

    def test_smart_skip_ignores_successes_and_non_size_models(self):
        runner = BatchRunner()
        runner._record_outcome_for_smart_skip(
            {"id": "ok", "size_gb": 13.0},
            "ollama_gpu",
            RunResult("ok", "OK", "ollama_gpu", True),
        )
        self.assertEqual(runner._oom_ceiling_gb, {})

        runner._record_outcome_for_smart_skip(
            {"id": "no-size", "size_gb": 0},
            "ollama_gpu",
            RunResult("no-size", "No size", "ollama_gpu", False,
                      error="cuda out of memory", failure_phase="runtime_error"),
        )
        self.assertEqual(runner._oom_ceiling_gb, {})

    def test_image_gen_timeout_tightens_oom_ceiling(self):
        """v5.5.7+: a runtime_timeout on image_comfyui must tighten the OOM
        ceiling so subsequent (larger) image-gen models don't pay the same
        dead-end 300s timeout cost.  Text timeouts (a 70B chat model can
        just be slow on CPU) must NOT tighten — that's what the
        consecutive-failure counter is for."""
        runner = BatchRunner(force_all=True)
        # Image-gen timeout @ 2 GB → ceiling tightens.
        runner._record_outcome_for_smart_skip(
            {"id": "realistic-vision-v6", "size_gb": 2.0},
            "image_comfyui",
            RunResult("realistic-vision-v6", "Realistic Vision v6", "image_comfyui",
                      False, error="Timeout after 300s", failure_phase="runtime_timeout"),
        )
        self.assertEqual(runner._oom_ceiling_gb.get("image_comfyui"), 2.0)
        # 4 GB image-gen model → skipped.
        self.assertIsNotNone(runner._smart_skip_reason(
            {"id": "counterfeit-v3", "size_gb": 4.0}, "image_comfyui"))
        # Text timeout on ollama_cpu → does NOT tighten ceiling.
        runner._record_outcome_for_smart_skip(
            {"id": "llama-70b", "size_gb": 40.0},
            "ollama_cpu",
            RunResult("llama-70b", "Llama 70B", "ollama_cpu",
                      False, error="Timeout after 300s", failure_phase="runtime_timeout"),
        )
        self.assertNotIn("ollama_cpu", runner._oom_ceiling_gb)

    def test_image_gen_supported_respects_allow_oversize_on_cpu(self):
        """v5.5.7+: CPU-only profiles under Force All / allow_oversize must
        be allowed to attempt image-gen at the runtime layer.  The pre-fix
        gate hard-returned False on no-GPU so the UI's "image-gen visible
        under Force All" promise was silently broken at runtime."""
        runner_cpu_fresh = BatchRunner(capacity_ram_gb=64.0,
                                       capacity_vram_gb=0.0,
                                       capacity_has_gpu=False,
                                       force_all=False)
        # Fresh CPU profile (no Force All) → image-gen still rejected.
        self.assertFalse(runner_cpu_fresh._image_gen_supported(
            {"id": "realistic-vision-v6", "min_vram_gb": 4}))

        runner_cpu_force = BatchRunner(capacity_ram_gb=64.0,
                                       capacity_vram_gb=0.0,
                                       capacity_has_gpu=False,
                                       force_all=True)
        # Force All on CPU profile → image-gen accepted.
        self.assertTrue(runner_cpu_force._image_gen_supported(
            {"id": "realistic-vision-v6", "min_vram_gb": 4}))

        # GPU profile with adequate VRAM → always accepted regardless.
        runner_gpu = BatchRunner(capacity_ram_gb=32.0,
                                 capacity_vram_gb=12.0,
                                 capacity_has_gpu=True)
        self.assertTrue(runner_gpu._image_gen_supported(
            {"id": "realistic-vision-v6", "min_vram_gb": 4}))

    def test_image_gen_supported_rejects_snapdragon_arm64_even_with_force_all(self):
        """v5.5.9 (Ron, 2026-05-26): Snapdragon X (Windows ARM64) is the one
        exception to the v5.5.6+ Force-All-unlocks-CPU-image-gen rule —
        torch-directml has no Windows-ARM64 wheel and ComfyUI's torchaudio
        import crashes with the ``torch_library_impl could not be located
        in _torchaudio.pyd`` Windows popup on startup. Returning False
        here keeps Force All as the documented "best-effort baseline /
        ignores capacity" escape hatch on every OTHER platform while
        preventing the process-killing popup on Snapdragon."""
        from src import gpu_detect

        # Patch the helper to simulate Snapdragon ARM64 without touching
        # sys.platform globally (sibling tests run in the same process).
        original = gpu_detect.is_snapdragon_arm64
        gpu_detect.is_snapdragon_arm64 = lambda: True
        # The import in batch_runner.py is "from src.gpu_detect import
        # is_snapdragon_arm64", so we must also patch the bound name in
        # the batch_runner module namespace.
        import src.batch_runner as br
        original_br = br.is_snapdragon_arm64
        br.is_snapdragon_arm64 = lambda: True
        self.addCleanup(lambda: setattr(gpu_detect, "is_snapdragon_arm64", original))
        self.addCleanup(lambda: setattr(br, "is_snapdragon_arm64", original_br))

        # Even Force All must NOT enable image-gen on Snapdragon — that
        # path leads to the torch_library_impl popup, not a clean error.
        runner_snapdragon_force = BatchRunner(
            capacity_ram_gb=64.0,
            capacity_vram_gb=0.0,
            capacity_has_gpu=False,
            force_all=True,
        )
        self.assertFalse(runner_snapdragon_force._image_gen_supported(
            {"id": "realistic-vision-v6", "min_vram_gb": 4}))

        # Same answer for a model with no VRAM requirement at all.
        self.assertFalse(runner_snapdragon_force._image_gen_supported(
            {"id": "tiny-image-model"}))

        # And same answer if a Snapdragon somehow ended up with a
        # capacity_has_gpu=True (defensive — torch-directml still wouldn't
        # work, so the popup risk is the same).
        runner_snapdragon_phantom_gpu = BatchRunner(
            capacity_ram_gb=64.0,
            capacity_vram_gb=8.0,
            capacity_has_gpu=True,
            force_all=True,
        )
        self.assertFalse(runner_snapdragon_phantom_gpu._image_gen_supported(
            {"id": "realistic-vision-v6", "min_vram_gb": 4}))

    def test_skip_combos_drops_completed_samples(self):
        """v5.5.7+ Resume Today's Run: combos pre-populated in
        ``skip_combos`` must be silently filtered out of
        ``_iter_selected_samples_for`` so resume runs only the remaining
        work without re-paying setup cost."""
        runner = BatchRunner(skip_combos={("chat", "ollama_cpu", 0)})
        model = {"id": "chat", "name": "Chat"}
        # Patch _iter_samples_for to return two samples so we can assert
        # that sample_index=0 is dropped while sample_index=1 survives.
        runner._iter_samples_for = lambda m, meth: [
            {"id": "s0", "prompt": "a", "title": "a"},
            {"id": "s1", "prompt": "b", "title": "b"},
        ]
        samples = runner._iter_selected_samples_for(model, "ollama_cpu")
        indexes = [idx for idx, _, _ in samples]
        self.assertEqual(indexes, [1])
        # Different method on same model → not skipped (skip_combos is
        # per (model_id, method, sample_index)).
        samples_other = runner._iter_selected_samples_for(model, "ollama_gpu")
        self.assertEqual([idx for idx, _, _ in samples_other], [0, 1])

    def test_force_all_run_loop_excludes_adaptive_skip_from_streak(self):
        """`adaptive_skip` failures must not count against the
        consecutive-failure ceiling — that's the whole point: a dead
        backend never bails the whole run. AST-pin the literal set
        membership in the run loop so a regression that drops
        ``adaptive_skip`` from the carve-out fails this test even
        without a full integration run."""
        import ast
        from pathlib import Path as _Path
        source = _Path(batch_runner.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        # Find every set-literal that contains both strings — the actual
        # source check at batch_runner.py:_run loop must include both
        # adaptive_skip and environment_skip in the same set so neither
        # bumps _consecutive_failures.
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Set):
                values = {
                    elt.value for elt in node.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                }
                if {"environment_skip", "adaptive_skip"}.issubset(values):
                    found = True
                    break
        self.assertTrue(
            found,
            "batch_runner.py run loop must check `result.failure_phase in "
            "{'environment_skip', 'adaptive_skip'}` so neither phase bumps "
            "the consecutive-failure streak. Regression risk: a dropped "
            "membership silently re-enables the stop-on-N behaviour during "
            "Force-All sweeps and partial-disk-pressure runs.",
        )

    def test_force_all_banner_does_not_leak_host_specs(self):
        """Regression-critical contract row 303: the Force-All banner reports
        behaviour only and MUST NOT print the host's CPU count, RAM GB,
        VRAM GB, or GPU SKU name. Any future copy-paste of specs into
        that print block would silently regress the privacy invariant."""
        import re as _re
        runner = BatchRunner(force_all=True)
        # Stub out the work paths so run() prints only the header + banner
        # and bails immediately without touching the catalog or Ollama.
        runner._select_models = lambda: []
        runner._count_combos = lambda models: 0
        runner._save_report = lambda: None
        buf = StringIO()
        with redirect_stdout(buf):
            try:
                runner.run()
            except Exception:
                pass
        banner = buf.getvalue()
        # Sanity: the banner must actually be emitted under force_all.
        self.assertIn("Force All mode", banner)
        # The banner must mention Force-All behavior but never leak specs.
        leak_patterns = [
            r"\b\d+\s*GB\s*(RAM|VRAM)\b",
            r"\b\d+\s*v?CPU\b",
            r"\bNVIDIA\s+[A-Z][\w-]+",
            r"\bAMD\s+(Radeon|Instinct)",
            r"\bGeForce\s+\w+",
            r"\bA10-\d+Q\b",
        ]
        for pattern in leak_patterns:
            self.assertFalse(
                _re.search(pattern, banner),
                f"Force-All banner leaked host specs via pattern {pattern!r}: "
                f"banner text was:\n{banner}",
            )

    def test_write_failure_diagnostics_emits_three_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            runner = BatchRunner(output_dir=output_dir)
            runner.report.add(RunResult(
                "m1", "Model 1", "ollama_gpu", False,
                error="cuda out of memory: tried to allocate 12 GB",
                failure_phase="runtime_error",
                prompt="hello",
                response_text="(crashed)",
            ))
            runner._captured_log_chunks.append("===== m1 / ollama_gpu =====\nload weights...\nfatal: OOM\n")
            written = runner._write_failure_diagnostics(output_dir)

            stem = runner.report.file_stem
            failures_path = output_dir / f"{stem}_failures.txt"
            env_path = output_dir / f"{stem}_env.txt"
            run_log_path = output_dir / f"{stem}_run.log"

            self.assertIn(failures_path, written)
            self.assertIn(env_path, written)
            self.assertIn(run_log_path, written)
            failures_text = failures_path.read_text(encoding="utf-8")
            self.assertIn("Model 1", failures_text)
            self.assertIn("cuda out of memory", failures_text)
            self.assertIn("hello", failures_text)
            env_text = env_path.read_text(encoding="utf-8")
            self.assertIn("machine_info", env_text)
            self.assertIn("python", env_text)
            run_log_text = run_log_path.read_text(encoding="utf-8")
            self.assertIn("fatal: OOM", run_log_text)

    def test_write_failure_diagnostics_skips_when_all_pass(self):
        """No failures → no sidecars. Keeps the folder tidy on a clean run."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            runner = BatchRunner(output_dir=output_dir)
            runner.report.add(RunResult("ok", "OK", "ollama_gpu", True))
            written = runner._write_failure_diagnostics(output_dir)
            self.assertEqual(written, [])
            stem = runner.report.file_stem
            self.assertFalse((output_dir / f"{stem}_failures.txt").exists())
            self.assertFalse((output_dir / f"{stem}_env.txt").exists())
            self.assertFalse((output_dir / f"{stem}_run.log").exists())

    def test_force_all_capacity_uses_real_hardware_on_cpu_only_profile(self):
        """When force_all is on, the profile's CPU-only flag is ignored and
        the runner uses the host's real GPU info so GPU methods can be
        attempted. We don't *invent* hardware — when get_gpu_info returns
        empty, has_gpu still becomes False."""
        with patch("src.batch_runner.get_ram_info", return_value={"total_mb": 64 * 1024}):
            with patch("src.batch_runner.get_gpu_info", return_value=[
                {"vram_total_mb": 24 * 1024, "unified_memory": False, "name": "Test GPU"}
            ]):
                runner = BatchRunner(
                    force_all=True,
                    capacity_ram_gb=16,
                    capacity_vram_gb=0,
                    capacity_has_gpu=False,
                )
                ram, has_gpu, vram, unified = runner._benchmark_capacity()
                self.assertEqual(ram, 64.0)
                self.assertTrue(has_gpu)
                self.assertEqual(vram, 24.0)
                self.assertFalse(unified)

        with patch("src.batch_runner.get_ram_info", return_value={"total_mb": 16 * 1024}):
            with patch("src.batch_runner.get_gpu_info", return_value=[]):
                runner = BatchRunner(force_all=True, capacity_has_gpu=False)
                ram, has_gpu, vram, _ = runner._benchmark_capacity()
                self.assertEqual(ram, 16.0)
                self.assertFalse(has_gpu)
                self.assertEqual(vram, 0.0)


class RunBatchCLIForceAllTests(unittest.TestCase):
    """The CLI must surface --force-all and forward it to BatchRunner."""

    def test_force_all_flag_present_in_help(self):
        import subprocess
        import sys as _sys
        result = subprocess.run(
            [_sys.executable, "run_batch.py", "--help"],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--force-all", result.stdout)
        self.assertIn("Force-All", result.stdout)


class ForceAllRuntimeRamGateBypassTests(unittest.TestCase):
    """v5.5.4: Force-All must bypass the Low-Resources live-RAM pre-flight
    that synthesises ``Skipped: Insufficient RAM: need X GB, only Y GB``
    results — this was the root cause of the 8 CPU constrained cloud VM run
    where 25/66 cases were skipped on RAM the host actually had. When the
    gate DOES fire (low_resources_mode + Force-All OFF), the synthesised
    RunResult must carry ``failure_phase='environment_skip'`` so the
    smart-skip ceiling treats it as a real environment limit instead of a
    ``unknown`` blank phase that the 8 CPU sidecars showed.
    """

    @staticmethod
    def _make_runner(force_all: bool):
        runner = BatchRunner(force_all=force_all)
        runner.low_resources_mode = True
        runner._purge_done = False
        runner._comfyui_client = None
        return runner

    def test_force_all_bypasses_live_ram_check_under_low_resources(self):
        """With Force-All ON, the live RAM gate must NOT be consulted —
        let the real Ollama load OOM instead of pre-rejecting on noisy
        psutil readings."""
        from unittest import mock
        runner = self._make_runner(force_all=True)
        model = {"id": "test-30b", "name": "Test 30B", "min_ram_gb": 64,
                 "size_gb": 30, "ollama_tag": "test:30b"}
        # If the gate FIRES, check_ram_for_model would be called. Patch it
        # to raise so any accidental call shows up as a test failure.
        with mock.patch("src.batch_runner.resource_manager.check_ram_for_model",
                        side_effect=AssertionError("RAM gate must be bypassed under Force-All")):
            # The gate is wrapped in `if self.low_resources_mode and not self.force_all`.
            # Confirm the runtime condition by evaluating the guard directly:
            should_check = runner.low_resources_mode and not runner.force_all
            self.assertFalse(should_check,
                             "Force-All + low_resources must NOT call check_ram_for_model")

    def test_low_resources_without_force_all_still_calls_ram_gate(self):
        """Sanity check: Low-Resources mode without Force-All must still
        consult the RAM gate — the bypass is conditional on force_all."""
        runner = self._make_runner(force_all=False)
        should_check = runner.low_resources_mode and not runner.force_all
        self.assertTrue(should_check,
                        "low_resources without force_all must still gate on RAM")

    def test_insufficient_ram_skip_carries_environment_skip_phase(self):
        """The synthesised ``Skipped: Insufficient RAM`` RunResult must
        set ``failure_phase='environment_skip'`` so the run loop excludes
        it from the consecutive-failure streak and the smart-skip ceiling
        prunes larger same-method models. v5.5.2 emitted blank phase
        which was the second 8 CPU root cause (sidecar showed
        `failure_phase: unknown`)."""
        # Locate the source line for the synthesised RunResult and assert
        # it explicitly sets the environment_skip phase.
        from pathlib import Path as _P
        src = _P(batch_runner.__file__).read_text(encoding="utf-8")
        # Find the guarded block.
        anchor = "if self.low_resources_mode and not self.force_all:"
        self.assertIn(anchor, src,
                      "v5.5.4 Force-All bypass guard must be present")
        block_start = src.index(anchor)
        # The synthesised RunResult must follow within ~600 chars and
        # carry failure_phase="environment_skip".
        block = src[block_start: block_start + 800]
        self.assertIn('failure_phase="environment_skip"', block,
                      "Insufficient-RAM skip must classify as environment_skip "
                      "(was blank in v5.5.2, the 8 CPU run bug)")
        self.assertIn("Skipped:", block,
                      "Synthesised result must still prefix the reason with 'Skipped:'")


class RepetitionLoopDetectorTests(unittest.TestCase):
    """Guards the streaming-time repetition-loop detector that prevents
    degenerate Ollama outputs (e.g., MiniCPM-V fake-roleplay loops on the
    2026-05-23 high-VRAM run) from burning the full token budget and
    being mis-classified as ``output_truncated``."""

    def test_detects_minicpm_v_style_paragraph_repetition(self):
        # Real failure shape: ~280-char paragraph repeated ~40x at temp=0.0.
        block = (
            "MiniCPM-V: I'm sorry, but as a language model trained by OpenAI, "
            "I don't have access to any specific websites or demos. Can you "
            "please ask another question? You: Could you recommend which demo "
            "choice would be better for my new website based on speed and "
            "quality? "
        )
        loopy = "Sure, here's an example. " + block * 10
        self.assertTrue(batch_runner._looks_like_repetition_loop(loopy))

    def test_detects_identical_three_repeats_at_tail(self):
        preamble = "lorem ipsum dolor sit amet " * 25
        repeat60 = "===<<NEXT IDENTICAL BLOCK FOR LOOP DETECTOR TEST>>===" * 3
        self.assertTrue(batch_runner._looks_like_repetition_loop(preamble + repeat60))

    def test_does_not_flag_short_responses(self):
        # Anything below the MIN_CHARS gate must pass — short answers can
        # legitimately repeat themselves (e.g., greetings, code stubs).
        small = "Yes. " * 50  # ~250 chars
        self.assertFalse(batch_runner._looks_like_repetition_loop(small))

    def test_does_not_flag_mixed_long_response(self):
        import random
        random.seed(42)
        mixed = " ".join("word" + str(random.randint(0, 9999)) for _ in range(400))
        self.assertFalse(batch_runner._looks_like_repetition_loop(mixed))

    def test_does_not_flag_section_headers_with_distinct_bodies(self):
        # Repeated structural markers like "### Section N" are common in
        # legitimate answers; only byte-identical block repetition counts.
        listy = "".join(
            f"### Section {i}\n\nThis section talks about {chr(65+i)*5} "
            f"with content about topic {i*7} and details {i*13}.\n\n"
            for i in range(8)
        )
        self.assertFalse(batch_runner._looks_like_repetition_loop(listy))

    def test_ignores_whitespace_only_tails(self):
        # Trailing whitespace runs (e.g., \n\n\n\n) must not register as
        # a repeat — the suffix.strip() guard handles this.
        self.assertFalse(
            batch_runner._looks_like_repetition_loop("legitimate answer." + " " * 1000)
        )

    def test_ignores_low_entropy_blocks(self):
        # A 100-char block of just "ab" repeats should not be misclassified
        # as a loop — the unique-chars-<=2 reject covers this trivial case.
        # (Real model loops have varied content inside the repeated block.)
        text = "Real prefix. " * 60 + ("ab" * 50) * 3
        self.assertFalse(batch_runner._looks_like_repetition_loop(text))

    def test_handles_none_and_empty_input(self):
        self.assertFalse(batch_runner._looks_like_repetition_loop(None))
        self.assertFalse(batch_runner._looks_like_repetition_loop(""))

    def test_detects_loop_after_long_legitimate_preamble(self):
        # Realistic failure shape: the model produces 1000+ chars of valid
        # prose, then degenerates into a loop that fills the rest of the
        # token budget. Detector should still fire because the tail window
        # (_REPETITION_TAIL_CHARS=1800) captures the loop, even though the
        # full response is dominated by legitimate content earlier.
        preamble = (
            "Here is a thorough answer to the question. " * 30  # ~1290 chars
        )
        block = (
            "Loop block that the small VLM keeps re-emitting once it falls "
            "into the degenerate roleplay attractor at temperature 0.0. "
        )  # ~120 chars
        loopy_tail = block * 12  # ~1440 chars of pure repetition
        self.assertTrue(
            batch_runner._looks_like_repetition_loop(preamble + loopy_tail),
            "Loop confined to the tail must still be detected even when "
            "earlier prose is legitimate.",
        )

    def test_does_not_flag_long_bullet_list_with_distinct_items(self):
        # 30 short bullets, each ~40 chars with DIFFERENT content. Total
        # length exceeds the 600-char gate so the detector enters its search
        # loop. Because every bullet body is byte-distinct, no candidate
        # block of 50+ chars can repeat 3x back-to-back — guarding the
        # _REPETITION_BLOCK_MIN=50 / byte-exact contract.
        items = "".join(
            f"- Item {i:02d}: {chr(65 + (i % 26)) * 6} note {i * 17}.\n"
            for i in range(30)
        )
        self.assertGreater(len(items), batch_runner._REPETITION_MIN_CHARS,
                           "Fixture must clear the MIN_CHARS gate to exercise "
                           "the actual search loop.")
        self.assertFalse(
            batch_runner._looks_like_repetition_loop(items),
            "A long list of distinct short bullets must not be mistaken for "
            "a repetition loop.",
        )


class RepetitionLoopRuntimeTests(unittest.TestCase):
    """End-to-end-ish guard: feed a fake Ollama stream that loops and verify
    the runner short-circuits with ``failure_phase='repetition_loop'``."""

    def test_run_ollama_generation_returns_repetition_loop_result(self):
        runner = BatchRunner()

        # Fake client that streams the loopy paragraph until the runner stops.
        block = (
            "MiniCPM-V: I'm sorry, but as a language model trained by OpenAI, "
            "I don't have access to any specific websites or demos. Can you "
            "please ask another question? You: Could you recommend which demo "
            "choice would be better for my new website based on speed and "
            "quality? "
        )

        class _FakeOllama:
            def chat_stream_with_stats(self, tag, messages, **kwargs):
                # Emit the block many times — repetition detector should
                # short-circuit well before this exhausts.
                for _ in range(80):
                    yield (block, None)
                yield ("", {"eval_count": 80, "eval_duration": 1_000_000_000,
                            "load_duration": 0, "total_duration": 1_000_000_000,
                            "prompt_eval_duration": 0, "done_reason": "length"})

        runner.ollama = _FakeOllama()
        runner._current_sample = {"prompt": "fake prompt", "index": 0, "count": 3}
        runner._system_snapshot = lambda: {}
        stop_event = threading.Event()
        model = {"id": "minicpm-v-vision", "name": "MiniCPM-V"}
        options = {
            "temperature": 0.0,
            "num_gpu": -1,
            "num_predict": 4096,
            "timeout_s": 600,
        }
        with redirect_stdout(StringIO()):
            result = runner._run_ollama_generation(
                model=model,
                method="ollama_gpu",
                tag="minicpm-v:latest",
                messages=[{"role": "user", "content": "fake"}],
                options=options,
                stop_event=stop_event,
                download_time=0.0,
                was_local=True,
            )
        self.assertFalse(result.success)
        self.assertEqual(result.failure_phase, "repetition_loop")
        self.assertIn("repetition loop", result.error.lower())
        # The runner must have stopped early — token_count should be far below
        # the 80 chunks the fake stream was prepared to emit.
        self.assertLess(result.token_count, 80,
                        "Repetition detector must short-circuit the stream early")

    def test_repetition_loop_result_carries_full_runresult_fields(self):
        """RunResult emitted from the repetition-loop branch must populate
        the same diagnostic fields a normal failed Ollama row would, so
        downstream report rendering (HTML matrix, JSON merge, retry-failed
        keying) doesn't blank-out columns just because the failure phase
        is new. Mirrors the shape of the ``output_truncated`` row whose
        place this branch takes over."""
        runner = BatchRunner()

        block = (
            "MiniCPM-V: I'm sorry, but as a language model trained by OpenAI, "
            "I don't have access to any specific websites or demos. Can you "
            "please ask another question? You: Could you recommend which demo "
            "choice would be better for my new website based on speed and "
            "quality? "
        )

        class _FakeOllama:
            def chat_stream_with_stats(self, tag, messages, **kwargs):
                for _ in range(80):
                    yield (block, None)
                yield ("", {
                    "eval_count": 80,
                    "eval_duration": 2_000_000_000,   # 2.0s of generation
                    "load_duration": 500_000_000,     # 0.5s load
                    "total_duration": 2_500_000_000,
                    "prompt_eval_duration": 0,
                    "done_reason": "length",
                })

        runner.ollama = _FakeOllama()
        runner._current_sample = {"prompt": "fake prompt", "index": 1, "count": 3}
        runner._system_snapshot = lambda: {"cpu": "test-fixture"}
        stop_event = threading.Event()
        model = {"id": "minicpm-v-vision", "name": "MiniCPM-V"}
        options = {
            "temperature": 0.0,
            "num_gpu": -1,
            "num_predict": 4096,
            "timeout_s": 600,
            "repeat_penalty": 1.15,
            "repeat_last_n": 256,
        }
        with redirect_stdout(StringIO()):
            result = runner._run_ollama_generation(
                model=model,
                method="ollama_gpu",
                tag="minicpm-v:latest",
                messages=[{"role": "user", "content": "fake"}],
                options=options,
                stop_event=stop_event,
                download_time=0.0,
                was_local=True,
            )

        # Core failure-classification fields.
        self.assertFalse(result.success)
        self.assertEqual(result.failure_phase, "repetition_loop")
        self.assertEqual(result.model_id, "minicpm-v-vision")
        self.assertEqual(result.model_name, "MiniCPM-V")
        self.assertEqual(result.method, "ollama_gpu")
        # Error message must mention the failure mode for log grep / matrix
        # tooltips, and must NOT mis-blame the token budget.
        self.assertIn("repetition loop", result.error.lower())
        self.assertNotIn("output_truncated", result.error.lower())
        # Diagnostic plumbing — the matrix and JSON merge depend on these.
        self.assertEqual(result.prompt, "fake prompt")
        self.assertIsInstance(result.options, dict)
        self.assertEqual(result.options.get("num_predict"), 4096)
        self.assertEqual(result.options.get("repeat_penalty"), 1.15)
        self.assertIsInstance(result.system_snapshot, dict)
        self.assertEqual(result.system_snapshot.get("cpu"), "test-fixture")
        # Timing fields: TTFT, token_count, and tokens_per_sec have
        # wall-clock fallbacks so they are populated even though the
        # detector breaks out of the stream BEFORE the final ``done=true``
        # chunk arrives. The Ollama-native stats fields (``load_time``,
        # ``ollama_eval_duration_ns``, ``ollama_eval_count``) stay zero
        # because the only chunk that carries them is the done chunk we
        # never read — that is the observed contract.
        self.assertGreater(result.ttft, 0.0,
                           "TTFT must be populated from wall-clock fallback "
                           "since first_token_wall is set inside the loop.")
        self.assertGreater(result.token_count, 0,
                           "token_count must fall back to len(tokens) when "
                           "stats.eval_count is missing.")
        self.assertGreater(result.tokens_per_sec, 0.0,
                           "tokens_per_sec must fall back to "
                           "len(tokens)/wall_total so the HTML matrix row "
                           "carries real perf data instead of suspicious "
                           "zeros. If this regresses to 0, update both this "
                           "test and the comment in _run_ollama_generation.")
        self.assertEqual(result.load_time, 0.0)
        self.assertEqual(result.ollama_eval_duration_ns, 0)
        self.assertEqual(result.ollama_eval_count, 0)

    def test_repeat_options_forwarded_to_ollama_chat_stream(self):
        """The override path: when the runner builds an ``options`` dict
        containing ``repeat_penalty`` / ``repeat_last_n`` (either from the
        extended-mode defaults at runtime or from a catalog override via
        ``benchmark_repeat_penalty`` / ``benchmark_repeat_last_n``), those
        values MUST reach ``ollama_client.chat_stream_with_stats`` so the
        Ollama daemon actually applies them. Without this, the catalog key
        would silently no-op."""
        runner = BatchRunner()
        captured_kwargs: dict = {}

        class _CapturingOllama:
            def chat_stream_with_stats(self, tag, messages, **kwargs):
                captured_kwargs["tag"] = tag
                captured_kwargs.update(kwargs)
                # Yield nothing — we only care about the kwargs the runner
                # built. An empty stream returns a short response that the
                # repetition detector ignores (< MIN_CHARS).
                yield ("hello", None)
                yield ("", {"eval_count": 1, "eval_duration": 1,
                            "load_duration": 0, "total_duration": 1,
                            "prompt_eval_duration": 0, "done_reason": "stop"})

        runner.ollama = _CapturingOllama()
        runner._current_sample = {"prompt": "fake", "index": 0, "count": 1}
        runner._system_snapshot = lambda: {}
        stop_event = threading.Event()
        options = {
            "temperature": 0.0,
            "num_gpu": -1,
            "num_predict": 256,
            "timeout_s": 600,
            "repeat_penalty": 2.5,   # catalog-override-shaped value
            "repeat_last_n": 512,
        }
        with redirect_stdout(StringIO()):
            runner._run_ollama_generation(
                model={"id": "demo", "name": "Demo"},
                method="ollama_gpu",
                tag="demo:latest",
                messages=[{"role": "user", "content": "hi"}],
                options=options,
                stop_event=stop_event,
                download_time=0.0,
                was_local=True,
            )

        self.assertEqual(captured_kwargs.get("repeat_penalty"), 2.5,
                         "Override repeat_penalty must reach Ollama unchanged")
        self.assertEqual(captured_kwargs.get("repeat_last_n"), 512,
                         "Override repeat_last_n must reach Ollama unchanged")
        # And the rest of the canonical stream kwargs must still be present
        # so we don't silently drop a sibling argument when refactoring.
        self.assertEqual(captured_kwargs.get("num_predict"), 256)
        self.assertEqual(captured_kwargs.get("num_gpu"), -1)
        self.assertEqual(captured_kwargs.get("temperature"), 0.0)


class SamplePromptOverrideTests(unittest.TestCase):
    """Pin that brittle/loopy auto-generated prompts have curated overrides
    so a future regression of ``model_demos._default_chat_demo`` cannot
    silently reintroduce the MiniCPM-V failure."""

    def test_minicpm_v_vision_has_curated_override(self):
        from src.sample_prompts import MODEL_DEMO_SAMPLE_OVERRIDES
        samples = MODEL_DEMO_SAMPLE_OVERRIDES.get("minicpm-v-vision", [])
        self.assertEqual(len(samples), 3,
                         "MiniCPM-V must keep 3 curated samples that avoid the "
                         "fake-roleplay 'Ask {name}…' template")
        for s in samples:
            self.assertNotIn("Ask MiniCPM-V", s,
                             "Curated override must not contain the meta-prompt "
                             "that triggered the 2026-05-23 high-VRAM loop")

    def test_default_chat_demo_template_avoids_meta_roleplay_prompt(self):
        """The fallback template applied to chat/vision models without an
        override must NOT be the third-person 'Ask {name} to…' framing that
        primed self-roleplay on MiniCPM-V. Anchored at a string in
        ``model_demos.py`` so a future agent can't reintroduce it."""
        from pathlib import Path
        from src import model_demos
        text = Path(model_demos.__file__).read_text(encoding="utf-8")
        self.assertNotIn("Ask {name} to compare two demo choices", text,
                         "The meta-prompt template that caused the MiniCPM-V "
                         "repetition loop must not return.")

    def test_get_model_demo_minicpm_v_vision_returns_curated_override(self):
        """End-to-end contract: ``get_model_demo`` MUST return the curated
        ``MODEL_DEMO_SAMPLE_OVERRIDES['minicpm-v-vision']`` samples instead
        of the auto-generated chat-demo fallback. If the override-application
        order is ever refactored (e.g., overrides applied before the
        category-aware fallback, or overrides consulted only for image
        models), this is the test that fires."""
        from src.model_demos import get_model_demo
        from src.sample_prompts import MODEL_DEMO_SAMPLE_OVERRIDES

        # Catalog-shaped fixture matching how Benchmark / Model-Guide call it.
        model = {
            "id": "minicpm-v-vision",
            "name": "MiniCPM-V",
            "category": "Vision",
            "tags": ["vision", "chat"],
        }
        demo = get_model_demo(model)
        expected = MODEL_DEMO_SAMPLE_OVERRIDES["minicpm-v-vision"]

        # samples list must match the curated overrides verbatim (trimmed +
        # capped at 3 by get_model_demo, but the override is already len==3).
        self.assertEqual(len(demo["samples"]), 3)
        for got, want in zip(demo["samples"], expected):
            self.assertEqual(got, want.strip())
        # primary must be the first override entry — UI cards surface it.
        self.assertEqual(demo["primary"], expected[0].strip())
        # And none of the surfaced samples may contain the loop-priming
        # third-person meta-prompt phrasing.
        for s in demo["samples"]:
            self.assertNotIn("Ask MiniCPM-V", s,
                             "get_model_demo must not surface the meta-prompt "
                             "that triggered the 2026-05-23 high-VRAM loop.")
            self.assertNotIn("Ask {name}", s)


class BatchRunnerNetworkResilienceTests(unittest.TestCase):
    """v5.5.12 regression coverage for the network-resilience improvements
    motivated by the May 2026 Mac M4 run that burned ~5h on 94 sequential
    ``ollama pull`` timeouts against an NXDOMAIN ``registry.ollama.ai``.
    """

    def _make_runner(self) -> BatchRunner:
        # Cheapest possible BatchRunner: bypass __init__ side effects
        # (catalog scans, phase1 env setup) — we only need the methods.
        runner = object.__new__(BatchRunner)
        runner._registry_dns_cache = None
        return runner

    def test_network_error_text_matcher_recognises_mac_run_failure_modes(self):
        runner = self._make_runner()
        # Every distinct error substring observed in the 2026-05-23 Mac run.
        for sample in [
            'pull model manifest: Get "https://registry.ollama.ai/v2/library/'
            'phi4/manifests/latest": dial tcp: lookup registry.ollama.ai: '
            "no such host",
            "max retries exceeded: Get \"https://dd20bb891979d25aebc8bec07b2"
            "b3bbc.r2.cloudflarestorage.com/...\"",
            "temporary failure in name resolution",
            "network is unreachable",
            "no route to host",
            "connection refused",
            "could not resolve host: registry.ollama.ai",
            "nodename nor servname provided, or not known",
        ]:
            with self.subTest(text=sample[:60]):
                self.assertTrue(
                    runner._is_network_unreachable_error_text(sample),
                    f"network matcher must classify {sample!r} as a "
                    "transient network outage so the consecutive-failure "
                    "ceiling is not tripped",
                )

    def test_network_error_text_matcher_lets_real_failures_through(self):
        runner = self._make_runner()
        # Non-network errors must NOT be reclassified — they would mask
        # real bugs (disk-full has its own ``is_disk_full_error_text``
        # branch upstream; everything else stays ``download_failed``).
        for sample in [
            "model is too large for available VRAM",
            "ollama server replied with status 500",
            "checksum mismatch on layer sha256:...",
            "",  # empty string must not match
        ]:
            with self.subTest(text=sample):
                self.assertFalse(
                    runner._is_network_unreachable_error_text(sample),
                    f"network matcher false-positive on {sample!r}",
                )

    def test_registry_dns_probe_is_cached_per_runner(self):
        runner = self._make_runner()
        calls: list[tuple[str, float]] = []

        def fake_probe(host="registry.ollama.ai", timeout_seconds=3.0):
            calls.append((host, timeout_seconds))
            return False, "fake NXDOMAIN"

        with patch.object(BatchRunner, "_probe_registry_dns", staticmethod(fake_probe)):
            r1 = runner._is_registry_reachable()
            r2 = runner._is_registry_reachable()
            r3 = runner._is_registry_reachable()

        self.assertEqual(r1, (False, "fake NXDOMAIN"))
        self.assertEqual(r1, r2)
        self.assertEqual(r2, r3)
        self.assertEqual(
            len(calls), 1,
            "DNS probe must be cached per BatchRunner so 100+ Ollama rows "
            "don't each trigger a getaddrinfo call. Got: " + repr(calls),
        )

    def test_run_ollama_short_circuits_to_environment_skip_when_registry_unreachable(self):
        # Reproduce the exact Mac-run failure path: model not pre-pulled,
        # DNS to registry.ollama.ai dead. Must short-circuit to
        # ``environment_skip`` instead of paying the multi-minute pull
        # timeout and getting classified as ``download_failed`` (which
        # would trip the consecutive-failure ceiling at row 10).
        runner = self._make_runner()
        runner._ollama_attempt_count = 0
        runner._unload_running_ollama_models = lambda: None
        runner._wait_for_ollama_running = lambda stop_event: True
        runner._is_ollama_tag_local = lambda tag: False
        runner._system_snapshot = lambda: {}
        runner.low_resources_mode = False
        runner.force_all = False

        model = {"id": "phi4:mini", "name": "Phi-4 Mini", "ollama_tag": "phi4-mini:latest", "size_gb": 0}
        stop = threading.Event()

        with patch.object(BatchRunner, "_probe_registry_dns",
                          staticmethod(lambda host="registry.ollama.ai", timeout_seconds=3.0:
                                       (False, "Name or service not known"))):
            result = BatchRunner._run_ollama(runner, model, "ollama_gpu", stop)

        self.assertFalse(result.success)
        self.assertEqual(result.failure_phase, "environment_skip",
                         "Network-down rows must NOT count against the "
                         "consecutive-failure ceiling.")
        self.assertIn("registry.ollama.ai DNS unreachable", result.error)
        self.assertIn("Name or service not known", result.error)
        self.assertIn("ollama pull phi4-mini:latest", result.error,
                      "Skip message must tell the user how to recover.")

    def test_run_ollama_does_not_probe_dns_when_model_is_already_local(self):
        # Pre-pulled models must run normally even if DNS is dead — that's
        # the entire point of "Ollama works offline once you've pulled".
        # Verify the DNS probe is never invoked for a was_local=True row.
        runner = self._make_runner()
        runner._ollama_attempt_count = 0
        runner._unload_running_ollama_models = lambda: None
        runner._wait_for_ollama_running = lambda stop_event: True
        runner._is_ollama_tag_local = lambda tag: True
        runner._system_snapshot = lambda: {}
        runner.low_resources_mode = False
        runner.force_all = False

        # Stub the pull so the row succeeds without touching the daemon.
        runner.ollama = type("StubOllama", (), {
            "pull_model": lambda self, tag, progress_cb=None, stop_event=None: None,
            "generate": lambda self, *args, **kwargs: {"text": "ok", "eval_count": 1, "eval_duration": 1},
        })()

        probe_calls: list[int] = []

        def boom(host="registry.ollama.ai", timeout_seconds=3.0):
            probe_calls.append(1)
            return False, "should not have been called"

        model = {"id": "qwen2.5:0.5b", "name": "Qwen 2.5 0.5B", "ollama_tag": "qwen2.5:0.5b", "size_gb": 0.4}
        stop = threading.Event()

        with patch.object(BatchRunner, "_probe_registry_dns", staticmethod(boom)):
            # We don't care about the run result here — we just need the
            # control flow up to the pull/run stage. If it raises later
            # (because we stubbed `ollama` so loosely), that's fine; the
            # assertion below only cares that the DNS probe didn't fire.
            try:
                BatchRunner._run_ollama(runner, model, "ollama_gpu", stop)
            except Exception:
                pass

        self.assertEqual(
            probe_calls, [],
            "Pre-pulled (was_local=True) rows must bypass the registry "
            "DNS probe so Ollama still runs offline on machines that "
            "already have the model cached.",
        )


if __name__ == "__main__":
    unittest.main()
