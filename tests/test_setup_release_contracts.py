import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="ignore")


# Maintainer-only files are gitignored and absent in public clones.
# Tests that read them are gated with @SKIP_WHEN_PUBLIC so the suite
# still passes against a fresh public clone (e.g. a contributor running
# `pytest tests/` after a `git clone`).
MAINTAINER_FILES_PRESENT = (
    (ROOT / "AGENTS.md").exists()
    and (ROOT / "CLAUDE.md").exists()
    and (ROOT / ".claude" / "agents" / "localai-setup-release-engineer.md").exists()
)
SKIP_WHEN_PUBLIC = unittest.skipUnless(
    MAINTAINER_FILES_PRESENT,
    "maintainer-only files (AGENTS.md / CLAUDE.md / .claude/agents/) "
    "are gitignored and absent in public clones",
)


class SetupReleaseContractTests(unittest.TestCase):
    def test_setup_detects_npu_and_defaults_openvino_yes(self):
        """setup1.bat must detect an Intel NPU (SETUP_HAS_NPU) and default the
        OpenVINO GenAI prompt to Yes when one is present — mirroring how
        DirectML auto-installs when an iGPU is detected."""
        setup = read("setup1.bat")
        self.assertIn("SETUP_HAS_NPU", setup)
        # Detection probes Intel NPU PCI IDs and/or the AI Boost friendly name.
        self.assertIn("B03E", setup, "Panther Lake NPU PCI id missing")
        self.assertIn("AI Boost", setup)
        # The OpenVINO install branch must key off SETUP_HAS_NPU and default
        # to y on that branch.
        npu_branch = setup[setup.index('"!SETUP_HAS_NPU!"=="1"'):]
        self.assertIn("INSTALL_OPENVINO", npu_branch)
        self.assertIn("set INSTALL_OPENVINO=y", npu_branch)

    def test_python_path_variable_is_consistent_across_launchers(self):
        setup = read("setup1.bat")
        self.assertIn("LOCALAI_PYTHON", setup)
        for rel in ["run.bat", "run_batch.bat", "verify_models.bat", "fix_nvidia_pytorch.bat", "fix_directml_pytorch.bat", "uninstall.bat"]:
            text = read(rel)
            self.assertIn("LOCALAI_PYTHON", text, rel)

    @SKIP_WHEN_PUBLIC
    def test_runtime_contract_files_are_in_publish_guidance(self):
        agents = read("AGENTS.md")
        plan_path = Path(r"C:\Plans\Publish.md")
        plan = plan_path.read_text(encoding="utf-8", errors="ignore") if plan_path.exists() else ""
        required = [
            "src\\", "docs\\", "AGENTS.md", "main.py", "run.bat", "run.sh",
            "run_batch.bat", "run_batch.py", "setup.bat", "setup1.bat",
            "setup.ps1", "setup.sh",
            "uninstall.bat", "uninstall.sh", "fix_nvidia_pytorch.bat",
            "fix_directml_pytorch.bat",
            "set_ollama_models_dir.bat",
            "verify_models.py", "verify_models.bat", "requirements.txt",
            "models_catalog.json", "content_blocklist.txt",
        ]
        for item in required:
            self.assertIn(item, agents, item)
            if plan:
                self.assertIn(item.rstrip("\\"), plan, item)
        if plan:
            self.assertIn(r"^LocalAI_v\d+(\.\d+)+\.zip$", plan)
            self.assertIn("Archive\\", plan)
            self.assertIn("(^|/)Archive/", plan)
            self.assertIn("toolbox_outputs\\", plan)
            self.assertIn("(^|/)toolbox_outputs/", plan)

    @SKIP_WHEN_PUBLIC
    def test_publish_guidance_requires_pre_edit_backups_and_recycle_bin_cleanup(self):
        agents = read("AGENTS.md")
        setup_agent = read(".claude/agents/localai-setup-release-engineer.md")
        plan_path = Path(r"C:\Plans\Publish.md")
        plan = plan_path.read_text(encoding="utf-8", errors="ignore") if plan_path.exists() else ""

        for label, text in [
            ("AGENTS.md", agents),
            ("localai-setup-release-engineer.md", setup_agent),
        ]:
            self.assertRegex(text.lower(), r"back ?up.*before edit", label)
            self.assertRegex(text.lower(), r"before (the )?first (edit|write)|before modifying any file", label)
            self.assertIn("timestamped", text, label)
            self.assertIn("Archive\\", text, label)
            self.assertIn("Recycle Bin", text, label)
            self.assertIn("benchmark", text, label)
            self.assertIn("latest", text.lower(), label)
            self.assertIn("never permanently delete", text.lower(), label)

        if plan:
            self.assertIn("Always back up before changing files", plan)
            self.assertIn("before the first write", plan)
            self.assertIn("pre-publish-backup", plan)
            self.assertIn("Recycle Bin", plan)
            self.assertIn("SendToRecycleBin", plan)
            self.assertIn("benchmark_results\\", plan)
            self.assertIn("previous-run staging folders", plan)
            self.assertIn('LocalAI_publish_${newVer}_$stamp', plan)
            self.assertNotIn('LocalAI_publish_$newVer"', plan)
            self.assertNotIn("Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue", plan)
            for protected in [
                "skus.json",
                "config.json",
                "model caches",
                "Ollama models",
                "ComfyUI checkpoints",
                "active backup folder",
            ]:
                self.assertIn(protected, plan, protected)
            self.assertIn("latest", plan.lower())
            self.assertIn("Never use permanent deletion", plan)

    def test_toolbox_package_sets_include_core_hidden_dependencies(self):
        app = read("src/app.py")
        setup = read("setup1.bat")
        uninstall = read("uninstall.bat")
        requirements = read("requirements.txt")
        match = re.search(r"full_package_set = \[(.*?)\]\s+dep_to_package", app, re.S)
        self.assertIsNotNone(match, "Toolbox install package list not found")
        packages = re.findall(r'"([^">=]+)(?:[>=][^"]*)?"', match.group(1))
        self.assertIn("accelerate", packages)
        self.assertIn("safetensors", packages)
        self.assertIn("sentence-transformers", packages)
        self.assertIn("diffusers", packages)
        for package in packages:
            self.assertIn(package, setup, package)
            self.assertIn(package, uninstall, package)
            self.assertIn(package, requirements, package)
        utility_start = setup.index("Installing utility model demo packages")
        utility_end = setup.index(":done_utility", utility_start)
        self.assertIn("torchvision", setup[utility_start:utility_end])

    @SKIP_WHEN_PUBLIC
    def test_huggingface_hub_is_pinned_below_one_in_install_paths(self):
        setup = read("setup1.bat")
        setup_sh = read("setup.sh")
        app = read("src/app.py")
        main = read("main.py")
        requirements = read("requirements.txt")
        agents = read("AGENTS.md")
        onnx_client = read("src/onnx_client.py")
        openvino_client = read("src/openvino_client.py")
        batch_runner = read("src/batch_runner.py")

        pin = "huggingface-hub>=0.34.0,<1.0"
        for label, text in [
            ("setup1.bat", setup),
            ("setup.sh", setup_sh),
            ("src/app.py", app),
            ("main.py", main),
            ("requirements.txt", requirements),
            ("AGENTS.md", agents),
            ("src/onnx_client.py", onnx_client),
            ("src/openvino_client.py", openvino_client),
            ("src/batch_runner.py", batch_runner),
        ]:
            self.assertIn(pin, text, label)

        openvino_lines = [
            line for line in setup.splitlines()
            if "openvino openvino-genai" in line and "-m pip install" in line
        ]
        self.assertTrue(openvino_lines, "OpenVINO install line not found")
        for line in openvino_lines:
            self.assertNotIn("huggingface-hub", line)
        self.assertIn("-m pip check", setup)

    def test_setup_repairs_orphaned_gguf_dependencies_before_core_install(self):
        setup = read("setup1.bat")
        repair_start = setup.index("Older uninstall.bat versions could remove gguf")
        core_start = setup.index("Installing required packages")
        self.assertLess(repair_start, core_start)
        repair_block = setup[repair_start:core_start]
        self.assertIn("find_spec('gguf')", repair_block)
        self.assertIn('"PyYAML>=5.1"', repair_block)
        self.assertIn('"tqdm>=4.27"', repair_block)

    def test_setup_pip_check_runs_after_image_generation_repairs(self):
        setup = read("setup1.bat")
        self.assertLess(
            setup.index("ComfyUI startup dependencies verified"),
            setup.index("Verifying Python dependency consistency"),
        )
        self.assertLess(
            setup.index(":img_done"),
            setup.index("Verifying Python dependency consistency"),
        )

    def test_gitignore_protects_private_runtime_and_artifact_files(self):
        gitignore = read(".gitignore")
        for pattern in (
            "skus.json",
            "config.json",
            "localai.log",
            "comfyui.log",
            "vm_sku_detection_*.txt",
            "toolbox_outputs/",
            "Archive/",
            "Scratchpad/",
            ".cache/",
            "*.bak2",
            "*.bak3",
        ):
            self.assertIn(pattern, gitignore)

    def test_huggingface_downloads_do_not_use_deprecated_symlink_argument(self):
        for rel in ["src/onnx_client.py", "src/openvino_client.py"]:
            self.assertNotIn("local_dir_use_symlinks", read(rel), rel)

    def test_gpu_detect_uses_localai_logger(self):
        gpu_detect = read("src/gpu_detect.py")
        self.assertIn("from src import logger", gpu_detect)
        self.assertNotIn("import logging", gpu_detect)

    @SKIP_WHEN_PUBLIC
    def test_release_baseline_ignores_backup_zip_names(self):
        agents = read("AGENTS.md")
        self.assertIn("LocalAI_v{version}.zip", agents)
        self.assertIn("Do **not** create patch zips", agents)
        self.assertIn("Do not delete backups, archives, or subdirectory zips.", agents)

    def test_setup_all_features_defaults_to_applicable_installs(self):
        setup = read("setup1.bat")
        self.assertIn('set /p SETUP_ALL_FEATURES="Setup all features? [Y/n]: "', setup)
        self.assertIn('if "!SETUP_ALL_FEATURES!"=="" set SETUP_ALL_FEATURES=y', setup)
        self.assertIn("call :detect_setup_hardware", setup)

    def test_setup_all_features_prompt_is_first_setup_prompt(self):
        setup = read("setup1.bat")
        prompts = re.findall(r"set /p ([A-Z_]+)=", setup)
        self.assertGreater(len(prompts), 1)
        self.assertEqual(prompts[0], "SETUP_ALL_FEATURES")
        self.assertLess(
            setup.index('set /p SETUP_ALL_FEATURES="Setup all features? [Y/n]: "'),
            setup.index(":: ── Pre-flight"),
        )

    def test_setup_all_features_yes_bypasses_individual_feature_prompts(self):
        setup = read("setup1.bat")
        prompt_by_var = {
            "INSTALL_ONNX": "Install NPU/DirectML (ONNX Runtime) support? [y/N]: ",
            "INSTALL_OPENVINO": "Install OpenVINO GenAI support? [y/N]: ",
            "INSTALL_UTILITY": "Install OCR/speech/embedding demo support? [y/N]: ",
            "INSTALL_IMG": "Set up image generation support? [y/N]: ",
            "INSTALL_COMFY": "Download and install ComfyUI now? [y/N]: ",
        }
        for var, prompt in prompt_by_var.items():
            pattern = (
                r'(?s)if "!SETUP_ALL_FEATURES!"=="1" \(\s*'
                rf"set {var}=y\b"
                r".*?\) else \(\s*"
                rf'set /p {var}="{re.escape(prompt)}"'
                r"\s*\)"
            )
            self.assertRegex(setup, pattern, var)

    def test_setup_all_features_gpu_acceleration_is_hardware_gated(self):
        setup = read("setup1.bat")
        start = setup.index('if "!SETUP_ALL_FEATURES!"=="1" (\n    if "!SETUP_HAS_GPU!"=="1" (')
        end = setup.index('if /i "!INSTALL_GPU_ACCEL!"=="y" (', start)
        block = setup[start:end]
        self.assertIn('if "!SETUP_HAS_GPU!"=="1"', block)
        self.assertIn('if "!SETUP_HAS_NVIDIA!"=="1"', block)
        self.assertIn("set INSTALL_GPU_ACCEL=y", block)
        self.assertIn("set INSTALL_GPU_ACCEL=n", block)
        self.assertIn("installing NVIDIA CUDA GPU acceleration", block)
        self.assertIn("installing DirectML GPU acceleration", block)
        self.assertIn("skipped GPU image acceleration because no supported local GPU was detected", block)
        self.assertIn('set /p INSTALL_GPU_ACCEL="Install GPU image acceleration (CUDA for NVIDIA, DirectML for AMD/Intel)? [y/N]: "', block)
        self.assertIn('if "!SETUP_HAS_GPU!"=="1"', block)
        install_start = setup.index('if /i "!INSTALL_GPU_ACCEL!"=="y" (', start)
        install_end = setup.index('(echo set LOCALAI_COMFYUI', install_start)
        install_block = setup[install_start:install_end]
        self.assertIn('if "!SETUP_HAS_NVIDIA!"=="1"', install_block)
        self.assertIn('else if "!SETUP_HAS_DML_GPU!"=="1"', install_block)
        self.assertIn("No supported local GPU detected; skipping GPU image acceleration.", install_block)

    def test_setup_installs_comfyui_next_to_app_not_user_profile(self):
        setup = read("setup1.bat")
        start = setup.index(":: 6b. Locate or install ComfyUI")
        end = setup.index(":comfyui_found", start)
        block = setup[start:end]
        self.assertIn('for %%F in ("%~dp0.") do set "APP_ROOT=%%~fF"', block)
        self.assertIn('set "COMFYUI_DEFAULT=!APP_ROOT!\\ComfyUI"', block)
        self.assertNotIn("%LocalAppData%\\LocalAI\\ComfyUI", block)
        self.assertNotIn("%LocalAppData%\\LocalAI", block)

        # The Mac launcher (setup.sh) is intentionally absent from the
        # public Windows-only clone. Skip the bash-side assertions when
        # the file isn't present so the suite still passes there.
        if not (ROOT / "setup.sh").exists():
            return
        setup_sh = read("setup.sh")
        # COMFY_ROOT/COMFY_DIR are defined once at the top of setup.sh and
        # reused throughout the install block, so check the assignments
        # against the full file text.
        self.assertIn('COMFY_ROOT="${APP_DIR}"', setup_sh)
        self.assertIn('COMFY_DIR="${APP_DIR}/ComfyUI"', setup_sh)
        sh_start = setup_sh.index('if is_yes "$INSTALL_COMFY"; then')
        sh_end = setup_sh.index('say_ok "ComfyUI saved at $COMFY_DIR"', sh_start)
        sh_block = setup_sh[sh_start:sh_end]
        self.assertNotIn('${HOME}/Library/Application Support/LocalAI', sh_block)

    def test_cuda_pytorch_failure_echo_escapes_parentheses_inside_batch_block(self):
        setup = read("setup1.bat")
        start = setup.index("\n:install_cuda_pytorch")
        end = setup.index(":install_directml_pytorch", start)
        block = setup[start:end]
        self.assertIn("nothing ^(antivirus,", block)
        self.assertIn("a profile virtualization policy^) is locking files", block)
        self.assertNotIn("nothing (antivirus,", block)
        self.assertNotIn("a profile virtualization policy) is locking files", block)

    def test_shipped_bat_echo_lines_escape_parens(self):
        """Every shipped .bat must escape ( and ) in `echo` lines INSIDE a
        nested `if (...)` / `for (...)` / `else (...)` block. cmd.exe tokenizes
        ( and ) as block delimiters inside such blocks even within echo text,
        and an unescaped paren aborts the enclosing block with
        "was unexpected at this time."

        Pinned by the uninstall.bat HuggingFace-cache crash (v5.5.0): unescaped
        parens in `echo This is LocalAI's own download cache (toolbox models,
        HF Hub).` and `echo (now stale ... missing path)` made cmd.exe abort
        the enclosing `if exist ... (` block before any HF cache was deleted.

        Top-level echo lines with parens are harmless and intentionally exempt
        — cmd.exe only special-cases parens inside parenthesized blocks.
        """
        bat_files = [
            "setup.bat", "setup1.bat", "run.bat", "run_batch.bat", "uninstall.bat",
            "fix_nvidia_pytorch.bat", "fix_directml_pytorch.bat",
            "verify_models.bat",
            "set_ollama_models_dir.bat",
        ]
        echo_line = re.compile(r"^\s*echo\b", re.IGNORECASE)
        comment_line = re.compile(r"^\s*(::|rem\b)", re.IGNORECASE)
        unescaped_paren = re.compile(r"(?<!\^)[()]")
        # For block-depth tracking on non-echo lines, strip "double quoted" text
        # so quoted parens (e.g. in set /p prompts) do not affect depth.
        quoted = re.compile(r'"[^"]*"')

        for rel in bat_files:
            text = read(rel)
            depth = 0
            offenders = []
            for lineno, raw in enumerate(text.splitlines(), start=1):
                if comment_line.match(raw):
                    continue
                if echo_line.match(raw):
                    # Flag only if currently inside a (...) block
                    if depth > 0 and unescaped_paren.search(raw):
                        offenders.append(f"  {rel}:{lineno}: {raw.rstrip()}")
                    continue
                # Non-echo, non-comment: track block depth via unescaped parens
                stripped = quoted.sub("", raw)
                opens = len([m for m in unescaped_paren.finditer(stripped) if m.group() == "("])
                closes = len([m for m in unescaped_paren.finditer(stripped) if m.group() == ")"])
                depth += opens - closes
                if depth < 0:
                    depth = 0  # tolerate single-line `if () else ()` shape
            self.assertEqual(
                offenders, [],
                f"{rel} has echo lines INSIDE a (...) block with unescaped "
                f"parens that break cmd.exe block parsing.\n"
                f"Fix by replacing ( with ^( and ) with ^) in the echo body.\n"
                + "\n".join(offenders),
            )

    def test_comfyui_path_resolution_keeps_config_and_bat_in_sync(self):
        app = read("src/app.py")
        start = app.index("def _comfyui_installed_path")
        end = app.index("def _check_comfyui_async", start)
        block = app[start:end]
        self.assertIn("self._sync_comfyui_path_bat(p)", block)
        self.assertIn('self.cfg["comfyui_dir"] = str(p)', block)
        self.assertIn("config.save(self.cfg)", block)

    def test_setup_hardware_detection_filters_virtual_adapters(self):
        setup = read("setup1.bat")
        start = setup.index("\n:detect_setup_hardware")
        end = setup.index(":find_python", start)
        block = setup[start:end]
        self.assertIn(":detect_setup_hardware", setup)
        self.assertIn("Win32_VideoController", block)
        self.assertIn("Microsoft|Basic|Remote|Hyper-V", block)
        self.assertIn("NVIDIA|GeForce|RTX|Quadro|AMD|Radeon|Intel|Arc|Iris|UHD|Xe", block)
        self.assertIn("SETUP_HAS_NVIDIA", block)
        self.assertIn("SETUP_HAS_DML_GPU", block)
        self.assertIn("SETUP_NVIDIA_GPU_NAMES", block)
        self.assertIn("SETUP_DML_GPU_NAMES", block)

    def test_nvidia_setup_uses_cuda_not_directml(self):
        setup = read("setup1.bat")
        self.assertIn("call :install_cuda_pytorch", setup)
        self.assertIn("call :install_directml_pytorch", setup)
        cuda_start = setup.index("\n:install_cuda_pytorch")
        cuda_end = setup.index("\n:install_directml_pytorch", cuda_start)
        cuda_block = setup[cuda_start:cuda_end]
        self.assertIn("torch-directml", cuda_block)
        self.assertIn("--index-url https://download.pytorch.org/whl/cu128", cuda_block)
        self.assertIn("--no-input", cuda_block)
        # v5.3.10: --no-cache-dir was replaced by PIP_CACHE_DIR redirection at
        # the top of the script. The wheel cache now lives on the install drive
        # so a small profile drive can't run out of space; the asserts below
        # pin the redirection invariant instead.
        self.assertIn("--disable-pip-version-check", cuda_block)
        self.assertIn("assert torch.cuda.is_available()", cuda_block)
        self.assertIn("Progress will be shown below", cuda_block)
        self.assertNotIn("--quiet", cuda_block)

    def test_fix_nvidia_pytorch_streams_and_verifies_cuda(self):
        fix = read("fix_nvidia_pytorch.bat")
        self.assertIn("setlocal enabledelayedexpansion", fix)
        self.assertIn("--no-input", fix)
        # v5.3.10: --no-cache-dir replaced by PIP_CACHE_DIR redirection.
        self.assertIn("PIP_CACHE_DIR=%~dp0.cache\\pip", fix)
        self.assertIn("--disable-pip-version-check", fix)
        self.assertIn("--index-url https://download.pytorch.org/whl/cu128", fix)
        self.assertIn("assert torch.cuda.is_available()", fix)
        self.assertIn("Progress will be shown below", fix)

    def test_fix_directml_pytorch_streams_and_verifies_directml(self):
        """v5.5.9 (Ron, 2026-05-26): DirectML recovery script for Intel /
        AMD / AI PC users hitting torch <-> torchaudio ABI drift
        (symptom: ``Entry Point Not Found: torch_library_impl could not be
        located in _torchaudio.pyd``). Mirrors fix_nvidia_pytorch.bat shape
        but uses a matched-set install (see v5.5.11 below) so versions
        resolve compatibly. Refuses to run on Windows ARM64 because
        torch-directml has no ARM64 wheel on PyPI.

        v5.5.11 (Ron, 2026-05-26): the original v5.5.9 single-line install
        ``pip install ... torch-directml torch torchvision torchaudio`` let
        pip's resolver pick the LATEST torchaudio (e.g. 2.11.0) against
        torch-directml's pinned torch (2.4.1), reintroducing the WinError
        127 / ``torch_library_impl in _torchaudio.pyd`` failure. The script
        now installs torch-directml alone first, discovers the pinned torch
        version, then installs torchaudio==<that-version> + torchvision.
        Detailed matched-set assertions live in
        ``DirectMlMatchedSetInstallContractTests`` below.
        """
        fix = read("fix_directml_pytorch.bat")
        self.assertIn("setlocal enabledelayedexpansion", fix)
        self.assertIn("--no-input", fix)
        self.assertIn("PIP_CACHE_DIR=%~dp0.cache\\pip", fix)
        self.assertIn("--disable-pip-version-check", fix)
        # v5.5.11: the broken v5.5.9 single-line install must NOT come back.
        self.assertNotIn(
            "torch-directml torch torchvision torchaudio",
            fix,
            "fix_directml_pytorch.bat must NOT bundle torch+torchvision+torchaudio "
            "with torch-directml on the same pip install line: pip's resolver picks "
            "the LATEST torchaudio against torch-directml's pinned torch and "
            "reintroduces WinError 127. Use the v5.5.11 matched-set pattern instead.",
        )
        # v5.5.11: torch-directml must still be installed first (so it pins torch),
        # then torchaudio==<discovered torch version> + torchvision.
        self.assertIn("pip install --upgrade --no-input --disable-pip-version-check torch-directml", fix)
        self.assertIn('"torchaudio==%TORCH_VER%" torchvision', fix)
        self.assertIn("torch_directml.is_available()", fix)
        self.assertIn("import torch, torchaudio, torch_directml", fix)
        self.assertIn("Progress will be shown below", fix)
        # ARM64 refusal: must bail BEFORE the pip install.
        first_arm_check = fix.find("ARM64")
        first_pip = fix.find("pip install")
        self.assertGreater(first_arm_check, 0,
                           "fix_directml_pytorch.bat must check for Windows ARM64")
        self.assertGreater(first_pip, 0)
        self.assertLess(
            first_arm_check, first_pip,
            "fix_directml_pytorch.bat must refuse to run on Windows ARM64 "
            "BEFORE attempting any pip install (no torch-directml wheel for ARM64).",
        )

    def test_comfyui_startup_dependencies_are_repaired(self):
        app = read("src/app.py")
        setup = read("setup1.bat")
        # The Mac launcher (setup.sh) is intentionally absent from the
        # public Windows-only clone. Defer reading it until after the
        # Windows assertions so the bash-side checks can be skipped
        # cleanly when the file isn't present.
        setup_sh = read("setup.sh") if (ROOT / "setup.sh").exists() else None

        self.assertIn('"sqlalchemy": "SQLAlchemy"', app)
        self.assertIn('"alembic": "alembic"', app)
        self.assertIn('"torchsde": "torchsde"', app)
        self.assertIn('"av": "av"', app)
        self.assertIn('"comfy_kitchen": "comfy-kitchen"', app)
        self.assertIn('"comfy_aimdo": "comfy-aimdo"', app)
        self.assertIn('"simpleeval": "simpleeval"', app)
        self.assertIn('"gguf": "gguf>=0.13.0"', app)
        self.assertIn('"yaml": "PyYAML>=5.1"', app)
        self.assertIn('"tqdm": "tqdm>=4.27"', app)
        self.assertIn('"google.protobuf": "protobuf"', app)
        self.assertIn('"pydantic_settings": "pydantic-settings~=2.0"', app)
        self.assertIn('"spandrel": "spandrel"', app)
        self.assertIn('"kornia": "kornia>=0.7.1"', app)
        self.assertIn('"OpenGL": "PyOpenGL"', app)
        self.assertIn('"glfw": "glfw"', app)
        self.assertIn('"dist:comfyui-frontend-package": "comfyui-frontend-package==1.39.19"', app)
        self.assertIn('"dist:comfyui-workflow-templates": "comfyui-workflow-templates==0.9.11"', app)
        self.assertIn('"dist:comfyui-embedded-docs": "comfyui-embedded-docs==0.4.3"', app)
        self.assertIn("importlib.metadata.version", app)
        self.assertIn("def _ensure_comfyui_core_dependencies", app)
        self.assertIn("_ensure_comfyui_core_dependencies(python_exe)", app)
        start = app.index("def _start_comfyui_process")
        source = app[start:app.index("# Build command with GPU", start)]
        self.assertIn("_ensure_comfyui_core_dependencies(python_exe)", source)
        helper_start = app.index("def _ensure_comfyui_core_dependencies")
        helper_source = app[helper_start:app.index("def download_comfyui_model", helper_start)]
        self.assertIn("threading.Lock", helper_source)
        self.assertIn("CREATE_NO_WINDOW", helper_source)
        self.assertNotIn("requirements.txt", helper_source)
        self.assertNotIn('"-r"', helper_source)
        self.assertIn(
            "SQLAlchemy alembic torchsde av comfy-kitchen comfy-aimdo simpleeval "
            '"gguf>=0.13.0" "PyYAML>=5.1" "tqdm>=4.27" protobuf "pydantic-settings~=2.0" spandrel "kornia>=0.7.1" PyOpenGL glfw '
            "comfyui-frontend-package==1.39.19 comfyui-workflow-templates==0.9.11 "
            "comfyui-embedded-docs==0.4.3",
            setup,
        )
        self.assertIn("ComfyUI startup dependencies verified", setup)
        if setup_sh is None:
            return
        self.assertIn(
            "SQLAlchemy alembic torchsde av comfy-kitchen comfy-aimdo simpleeval "
            '"gguf>=0.13.0" "PyYAML>=5.1" "tqdm>=4.27" protobuf "pydantic-settings~=2.0" spandrel "kornia>=0.7.1" PyOpenGL glfw '
            "comfyui-frontend-package==1.39.19 comfyui-workflow-templates==0.9.11 "
            "comfyui-embedded-docs==0.4.3",
            setup_sh,
        )
        self.assertLess(
            setup.index('-r "!COMFYUI_PATH!\\requirements.txt"'),
            setup.index("SQLAlchemy alembic torchsde av comfy-kitchen comfy-aimdo simpleeval"),
        )
        self.assertLess(
            setup_sh.index('-r "$COMFY_DIR/requirements.txt"'),
            setup_sh.index("SQLAlchemy alembic torchsde av comfy-kitchen comfy-aimdo simpleeval"),
        )

    def test_uninstall_removes_full_comfyui_startup_dependency_set(self):
        uninstall = read("uninstall.bat")
        all_start = uninstall.index('if /i "!PIP_CHOICE!"=="a"')
        all_end = uninstall.index(') else if /i "!PIP_CHOICE!"=="b"', all_start)
        image_start = uninstall.index(') else if /i "!PIP_CHOICE!"=="e"')
        image_end = uninstall.index(') else (', image_start)
        for block in (uninstall[all_start:all_end], uninstall[image_start:image_end]):
            for package in (
                "gguf",
                "PyYAML",
                "tqdm",
                "SQLAlchemy",
                "alembic",
                "torchsde",
                "av",
                "comfy-kitchen",
                "comfy-aimdo",
                "simpleeval",
                "protobuf",
                "pydantic-settings",
                "spandrel",
                "kornia",
                "PyOpenGL",
                "glfw",
                "comfyui-frontend-package",
                "comfyui-workflow-templates",
                "comfyui-embedded-docs",
            ):
                self.assertIn(package, block)

    def test_comfyui_dependency_repair_installs_only_targeted_db_packages(self):
        from src import app as app_module
        from src.app import App

        class Result:
            def __init__(self, returncode, stdout="", stderr=""):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr

        app = object.__new__(App)
        app.after = lambda _delay, callback=None: callback() if callback else None
        app._set_comfyui_status = lambda *_args, **_kwargs: None
        app._comfyui_dependency_lock = app_module.threading.Lock()

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            if len(calls) in (1, 2):
                return Result(
                    1,
                    "sqlalchemy\nalembic\ntorchsde\nav\ncomfy_kitchen\ncomfy_aimdo\n"
                    "simpleeval\ngguf\nyaml\ntqdm\ngoogle.protobuf\npydantic_settings\nspandrel\n"
                    "kornia\nOpenGL\nglfw\ndist:comfyui-frontend-package\n"
                    "dist:comfyui-workflow-templates\ndist:comfyui-embedded-docs\n",
                )
            if len(calls) == 3:
                return Result(0)
            return Result(0)

        original_run = app_module.subprocess.run
        try:
            app_module.subprocess.run = fake_run
            self.assertTrue(App._ensure_comfyui_core_dependencies(app, "python"))
        finally:
            app_module.subprocess.run = original_run

        self.assertEqual(len(calls), 4)
        install_cmd = calls[2][0]
        self.assertEqual(install_cmd[:5], ["python", "-m", "pip", "install", "--upgrade"])
        self.assertIn("--no-input", install_cmd)
        self.assertIn("--disable-pip-version-check", install_cmd)
        self.assertIn("SQLAlchemy", install_cmd)
        self.assertIn("alembic", install_cmd)
        self.assertIn("torchsde", install_cmd)
        self.assertIn("av", install_cmd)
        self.assertIn("comfy-kitchen", install_cmd)
        self.assertIn("comfy-aimdo", install_cmd)
        self.assertIn("simpleeval", install_cmd)
        self.assertIn("gguf>=0.13.0", install_cmd)
        self.assertIn("PyYAML>=5.1", install_cmd)
        self.assertIn("tqdm>=4.27", install_cmd)
        self.assertIn("protobuf", install_cmd)
        self.assertIn("pydantic-settings~=2.0", install_cmd)
        self.assertIn("spandrel", install_cmd)
        self.assertIn("kornia>=0.7.1", install_cmd)
        self.assertIn("PyOpenGL", install_cmd)
        self.assertIn("glfw", install_cmd)
        self.assertIn("comfyui-frontend-package==1.39.19", install_cmd)
        self.assertIn("comfyui-workflow-templates==0.9.11", install_cmd)
        self.assertIn("comfyui-embedded-docs==0.4.3", install_cmd)
        self.assertNotIn("-r", install_cmd)
        self.assertFalse(any("requirements.txt" in str(part) for part in install_cmd))

        calls.clear()

        def fake_run_no_missing(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return Result(0)

        try:
            app_module.subprocess.run = fake_run_no_missing
            self.assertTrue(App._ensure_comfyui_core_dependencies(app, "python"))
        finally:
            app_module.subprocess.run = original_run

        self.assertIn(len(calls), (0, 1))

    def test_image_validation_harness_exercises_app_paths_and_cleans_up(self):
        tool = read("tools/validate_image_gen_models.py")

        self.assertIn("app = App()", tool)
        self.assertIn('app._switch_page("image_gen")', tool)
        self.assertIn("app._start_comfyui_process()", tool)
        self.assertIn("app.comfyui.generate_image(", tool)
        self.assertIn('entry.get("supports_img2img")', tool)
        self.assertIn("content_filter.check_prompt(SAFE_BLOCKED_FIXTURE_PROMPT)", tool)
        self.assertIn('"explicit_nsfw_generation": False', tool)
        self.assertIn("blocked_fixture_non_explicit", tool)
        self.assertIn("report.json", tool)
        self.assertIn("report.csv", tool)
        self.assertIn("report.md", tool)
        self.assertIn("def _stop_owned_comfyui", tool)
        self.assertIn("proc.terminate()", tool)
        self.assertIn("proc.kill()", tool)
        self.assertIn("app.comfyui.free_vram()", tool)
        self.assertIn("_create_synthetic_reference_image", tool)
        self.assertIn('--safety-fixture-mode", choices=("once", "all", "skip"), default="once"', tool)
        fixture = re.search(r"SAFE_BLOCKED_FIXTURE_PROMPT = \((.*?)\)", tool, re.S)
        self.assertIsNotNone(fixture)
        self.assertNotIn("nude", fixture.group(1).lower())

    def test_setup_utility_install_preserves_directml_onnx(self):
        """v2026.06.01.7 (Ron, 2026-06-01): the reconcile step is now
        variant-aware. Pre-v.7 this unconditionally re-installed
        onnxruntime-directml even on NVIDIA boxes, which collided with the
        -gpu wheel and broke onnxruntime.__file__. Post-v.7 the message
        carries the SKU label, the chosen variant is templated via
        !SETUP_ONNX_PKG! / !SETUP_ONNX_GENAI_PKG!, and the provider check
        uses !SETUP_ONNX_EP!. DirectML still works on AMD/Intel boxes via
        the SETUP_HAS_DML_GPU branch.
        """
        setup = read("setup1.bat")
        # DirectML pair must still be referenced (in the purge list AND in
        # the SETUP_HAS_DML_GPU branch of :detect_setup_hardware).
        self.assertIn("onnxruntime-directml", setup)
        self.assertIn("onnxruntime-genai-directml", setup)
        # Reconcile message is now variant-aware (uses !SETUP_ONNX_LABEL!).
        self.assertIn(
            "Reconciling !SETUP_ONNX_LABEL! ONNX runtime after utility packages",
            setup,
            "v2026.06.01.7: reconcile message must use !SETUP_ONNX_LABEL! "
            "so NVIDIA boxes don't print misleading 'DirectML' text.",
        )
        # Provider check is variable, not hard-coded.
        self.assertIn("!SETUP_ONNX_EP!", setup)
        # Success message is now variant-aware.
        self.assertIn(
            "[OK] !SETUP_ONNX_LABEL! ONNX provider verified after utility install.",
            setup,
        )
        # DmlExecutionProvider must still be a possible target (set in the
        # SETUP_HAS_DML_GPU branch of :detect_setup_hardware).
        self.assertIn("DmlExecutionProvider", setup)

    def test_setup_uninstalls_bare_onnxruntime_before_directml(self):
        """v2026.06.01.7 (Ron, 2026-06-01): superset of the original v.6
        contract. The bug originally observed 2026-05-31 (CPU `onnxruntime`
        from `optimum[onnxruntime]` shadowing `onnxruntime-directml`) was a
        symptom of a broader namespace-collision class: any TWO of the
        three runtime wheels (`onnxruntime`, `onnxruntime-gpu`,
        `onnxruntime-directml`) installed side by side leave
        `onnxruntime.__file__ == None` and `InferenceSession` gone. v.7
        purges ALL SIX mutually-exclusive packages (3 runtimes + 3 genai)
        immediately before each install. This test pins the broader
        contract: every `pip install` of an onnxruntime runtime variant
        must be preceded by an `uninstall -y` of all six within a small
        window. The earlier narrow contract (uninstall bare onnxruntime
        before -directml) was the v.6 fix that turned out to be
        insufficient on NVIDIA boxes — Toolbox Speak regressed on a
        workstation-class GPU with v.6 because -gpu was the colliding
        variant, not bare CPU.
        """
        setup_lines = read("setup1.bat").splitlines()
        # Any line that pip-installs a runtime variant (templated via
        # !SETUP_ONNX_PKG! OR hard-coded onnxruntime-directml etc.).
        runtime_tokens = (
            "!SETUP_ONNX_PKG!",
            "onnxruntime-directml",
            "onnxruntime-gpu",
        )
        install_indices = []
        for i, line in enumerate(setup_lines):
            if "pip install" not in line or "pip uninstall" in line:
                continue
            if any(tok in line for tok in runtime_tokens):
                install_indices.append(i)
        self.assertGreaterEqual(
            len(install_indices), 2,
            "expected at least 2 onnxruntime runtime install lines "
            "(initial ONNX step + utility reconcile); found %d" % len(install_indices),
        )
        for idx in install_indices:
            preamble = "\n".join(setup_lines[max(0, idx - 10):idx])
            # Must purge all SIX variants, not just bare onnxruntime.
            for pkg in (
                "onnxruntime",
                "onnxruntime-gpu",
                "onnxruntime-directml",
                "onnxruntime-genai",
                "onnxruntime-genai-cuda",
                "onnxruntime-genai-directml",
            ):
                self.assertRegex(
                    preamble,
                    rf"pip\s+uninstall\s+-y[^\n]*\b{re.escape(pkg)}\b",
                    msg=(
                        f"setup.bat line {idx + 1}: must `pip uninstall -y "
                        f"... {pkg} ...` immediately before installing an "
                        f"onnxruntime runtime variant. v2026.06.01.7 requires "
                        f"all six mutually-exclusive packages be purged so a "
                        f"prior-run half-install can't shadow the chosen wheel. "
                        f"Preamble was:\n{preamble}"
                    ),
                )

    def test_interactive_onnx_loader_uses_genai_for_phi_bundles(self):
        app = read("src/app.py")
        self.assertIn("OnnxGenAISession", app)
        self.assertIn("has_genai_config", app)
        load_start = app.index("def _load_onnx_model")
        load_end = app.index("def _onnx_load_done", load_start)
        load_source = app[load_start:load_end]
        self.assertIn("GENAI_AVAILABLE", load_source)
        self.assertIn('load_provider == "OpenVINOExecutionProvider"', load_source)
        self.assertIn("onnx_cpu_subfolder", load_source)
        self.assertIn("OnnxGenAISession(model_dir, subfolder=load_subfolder)", load_source)
        onnx_client = read("src/onnx_client.py")
        self.assertIn("class OnnxGenAISession", onnx_client)
        self.assertIn("def generate_stream(", onnx_client)

    def test_manual_onnx_docs_include_genai_directml(self):
        docs = read("docs/index.html")
        self.assertGreaterEqual(docs.count("onnxruntime-genai-directml"), 3)

    @SKIP_WHEN_PUBLIC
    def test_macos_setup_guidance_matches_setup_sh(self):
        claude = read("CLAUDE.md")
        setup_agent = read(".claude/agents/localai-setup-release-engineer.md")
        docs = read("docs/index.html")
        setup_sh = read("setup.sh")
        run_sh = read("run.sh")
        for label, text in [("CLAUDE.md", claude), ("setup agent", setup_agent), ("docs/index.html", docs)]:
            normalized = text.lower()
            self.assertIn("requires an existing python 3.10+", normalized, label)
            self.assertIn("does not install python", normalized, label)
            self.assertNotIn(".venv/bin/python3", normalized, label)
            self.assertNotIn("setup script does this automatically", normalized, label)
        self.assertIn("sys.version_info >= (3, 10)", setup_sh)
        self.assertIn("Python 3.10+ required", setup_sh)
        self.assertIn('"$PYTHON_BASE" -m venv "$APP_DIR/.venv"', setup_sh)
        self.assertIn('PYTHON="$APP_DIR/.venv/bin/python"', setup_sh)
        self.assertIn('"$PYTHON" -m pip install --upgrade -r "$APP_DIR/requirements.txt"', setup_sh)
        self.assertIn('"$PYTHON" -m pip check', setup_sh)
        self.assertIn("import customtkinter, requests, psutil", setup_sh)
        self.assertIn('"$APP_DIR/.venv/bin/python"', run_sh)
        self.assertNotIn('command -v python3', run_sh)
        self.assertIn("LocalAI virtual environment not found", run_sh)
        self.assertIn('"$PYTHON" "$APP_DIR/main.py" "$@"', run_sh)
        self.assertIn("[ERROR] App exited with error code", run_sh)
        self.assertNotIn('exec "$PYTHON" "$APP_DIR/main.py" "$@"', run_sh)
        self.assertIn(".venv/bin/python main.py", docs)
        self.assertIn("Windows / active environment", docs)

    @SKIP_WHEN_PUBLIC
    def test_cuda_and_directml_installs_redirect_pip_cache_off_profile_drive(self):
        """v5.3.10 replaces ``--no-cache-dir`` with an explicit cache
        redirection block at the top of ``setup.bat`` and
        ``fix_nvidia_pytorch.bat`` that points
        ``PIP_CACHE_DIR`` / ``TMP`` / ``TEMP`` / ``HF_HOME`` /
        ``HUGGINGFACE_HUB_CACHE`` / ``TORCH_HOME`` / ``XDG_CACHE_HOME`` /
        ``OLLAMA_MODELS`` at ``%~dp0.cache\\...`` (and ``%~dp0Ollama``) so the
        2.8 GB CUDA wheel + every other runtime download lands on the install
        drive, not the (possibly tiny) profile-virtualization-capped %USERPROFILE%.

        The original ``--no-cache-dir`` workaround was a partial fix — it
        avoided the cachecontrol spool but still let ``%TMP%`` (a side effect
        of pip's atomic-rename behavior) live on the profile drive. The
        redirection block eliminates that entire class of failure for setup
        AND every helper that touches pip/HF/torch caches.

        The pre-existing disk preflight subroutine (``:check_disk_space``)
        must remain absent — it was the misdiagnosed earlier draft and would
        give false reassurance.
        """
        setup = read("setup1.bat")
        fix = read("fix_nvidia_pytorch.bat")
        fix_dml = read("fix_directml_pytorch.bat")
        agents = read("AGENTS.md")

        self.assertIsNotNone(re.search(r"(?m)^:install_cuda_pytorch\b", setup))
        self.assertIsNotNone(re.search(r"(?m)^:install_directml_pytorch\b", setup))
        # No misdiagnosed disk preflight.
        self.assertIsNone(re.search(r"(?m)^:check_disk_space\b", setup))

        # Required redirection block (BEFORE any pip / python call).
        for env_line in (
            'set "PIP_CACHE_DIR=%~dp0.cache\\pip"',
            'set "TMP=%~dp0.cache\\tmp"',
            'set "TEMP=%~dp0.cache\\tmp"',
            'set "HF_HOME=%~dp0.cache\\huggingface"',
            'set "HUGGINGFACE_HUB_CACHE=%~dp0.cache\\huggingface\\hub"',
            'set "TORCH_HOME=%~dp0.cache\\torch"',
            'set "XDG_CACHE_HOME=%~dp0.cache"',
            'set "OLLAMA_MODELS=%~dp0Ollama"',
        ):
            self.assertIn(env_line, setup)
            self.assertIn(env_line, fix)
            self.assertIn(env_line, fix_dml)

        # Redirection block must appear BEFORE the first pip install in both.
        first_pip_setup = setup.find('pip install')
        first_env_setup = setup.find('PIP_CACHE_DIR=%~dp0.cache')
        self.assertGreater(first_pip_setup, 0)
        self.assertGreater(first_env_setup, 0)
        self.assertLess(
            first_env_setup, first_pip_setup,
            "PIP_CACHE_DIR redirect must come BEFORE the first pip install in setup.bat",
        )

        first_pip_fix = fix.find('pip install')
        first_env_fix = fix.find('PIP_CACHE_DIR=%~dp0.cache')
        self.assertGreater(first_pip_fix, 0)
        self.assertGreater(first_env_fix, 0)
        self.assertLess(
            first_env_fix, first_pip_fix,
            "PIP_CACHE_DIR redirect must come BEFORE the first pip install in fix_nvidia_pytorch.bat",
        )

        first_pip_dml = fix_dml.find('pip install')
        first_env_dml = fix_dml.find('PIP_CACHE_DIR=%~dp0.cache')
        self.assertGreater(first_pip_dml, 0)
        self.assertGreater(first_env_dml, 0)
        self.assertLess(
            first_env_dml, first_pip_dml,
            "PIP_CACHE_DIR redirect must come BEFORE the first pip install in fix_directml_pytorch.bat",
        )

        # pip cache purge on failure is still the right recovery hatch.
        self.assertIn("pip cache purge", setup)
        self.assertIn("pip cache purge", fix)
        self.assertIn("pip cache purge", fix_dml)
        # Get-PSDrive preflight must NOT exist anywhere.
        self.assertNotIn("Get-PSDrive", fix)
        self.assertNotIn("Get-PSDrive", fix_dml)
        self.assertNotIn("Get-PSDrive", setup)

        # AGENTS.md DO NOT REGRESS row carries the new invariant. v5.3.10
        # replaces the --no-cache-dir requirement; the row text is rewritten
        # to pin the cache-redirection invariant instead. We assert on the
        # canonical phrasing tokens to make a row deletion fail loudly.
        self.assertIn("PIP_CACHE_DIR", agents)
        self.assertIn(".cache\\pip", agents)
        self.assertIn("OLLAMA_MODELS", agents)


class DirectMlAbiSelfHealContractTests(unittest.TestCase):
    """v5.5.10: setup.bat + run.bat must self-heal DirectML torch/torchaudio
    ABI drift. The drift surfaces as a Windows ``Entry Point Not Found:
    torch_library_impl ... _torchaudio.pyd`` dialog at import time, killing
    image generation on Intel / AMD / AI PC machines whose torchaudio
    got bumped independently of torch (pip ``--upgrade`` cannot repair it
    because torch-directml's torch pin freezes torch).

    Setup must run an ABI pre-check BEFORE the install and uninstall-then-
    reinstall on drift. Run must run the same pre-check BEFORE launching
    the app, with SetErrorMode-suppressed imports so the popup never blocks
    the user. The pre-check is the same Python snippet as fix_directml_pytorch.bat
    so the three scripts agree on what "healthy" means.
    """

    def test_setup_install_directml_pytorch_self_heals_on_abi_drift(self):
        setup = read("setup1.bat")
        start = setup.index("\n:install_directml_pytorch")
        end = setup.index("\n:find_python", start)
        block = setup[start:end]

        # The ABI pre-check must SetErrorMode-suppress the OS popup so the
        # broken-import failure surfaces as a clean nonzero exit instead of
        # a blocking dialog the user cannot dismiss without losing state.
        self.assertIn(
            "SetErrorMode(0x8001)",
            block,
            ":install_directml_pytorch must SetErrorMode(0x8001) to suppress "
            "the OS popup before the torch/torchaudio import attempt.",
        )
        self.assertIn(
            "import torch, torchaudio, torch_directml",
            block,
            ":install_directml_pytorch ABI pre-check must import ALL THREE of "
            "torch, torchaudio, torch_directml (drift fires on torchaudio).",
        )
        self.assertIn(
            "torch_directml.is_available()",
            block,
            ":install_directml_pytorch must verify torch_directml.is_available() "
            "in the ABI pre-check, not just that the import succeeded.",
        )

        # The ABI pre-check must appear BEFORE the pip install, mirroring
        # how :install_cuda_pytorch gates on torch.cuda.is_available() before
        # the (re)install. Without the pre-check, --upgrade re-runs every
        # time and cannot repair an existing drift.
        precheck_pos = block.find("Checking DirectML PyTorch ABI health")
        install_pos = block.find("pip install --upgrade")
        self.assertGreater(precheck_pos, 0,
                           ":install_directml_pytorch must announce the ABI pre-check.")
        self.assertGreater(install_pos, 0,
                           ":install_directml_pytorch must still install torch-directml.")
        self.assertLess(
            precheck_pos, install_pos,
            ":install_directml_pytorch ABI pre-check must run BEFORE the pip install "
            "so an already-healthy environment is left alone and a drifted environment "
            "is detected before the unconditional --upgrade.",
        )

        # On drift we must uninstall BEFORE reinstalling. --upgrade alone
        # cannot realign torch and torchaudio because torch-directml's torch
        # pin keeps pip from touching torch. The uninstall must enumerate
        # all four packages in the same order as fix_directml_pytorch.bat.
        self.assertIn(
            "pip uninstall -y torch torchvision torchaudio torch-directml",
            block,
            ":install_directml_pytorch must uninstall torch+torchvision+torchaudio+"
            "torch-directml BEFORE reinstalling so the new packages land as a matched "
            "ABI set (fix_directml_pytorch.bat does the same and the two must agree).",
        )
        uninstall_pos = block.find("pip uninstall -y torch torchvision torchaudio torch-directml")
        self.assertLess(
            uninstall_pos, install_pos,
            ":install_directml_pytorch must uninstall BEFORE the pip install --upgrade "
            "(fix_directml_pytorch.bat's clean-reinstall flow).",
        )

        # Post-install verify with SetErrorMode-suppressed import.
        verify_pos = block.find("Verifying DirectML PyTorch ABI alignment")
        self.assertGreater(verify_pos, install_pos,
                           ":install_directml_pytorch must verify ABI alignment AFTER the install.")
        post_install_block = block[install_pos:]
        self.assertIn("SetErrorMode(0x8001)", post_install_block,
                      "Post-install verify must also use SetErrorMode to suppress popup.")

    def test_run_bat_runs_directml_abi_preflight_before_app_launch(self):
        run = read("run.bat")

        # Preflight must use importlib.util.find_spec to skip the check on
        # CPU-only / CUDA-only setups (where torch-directml is not installed
        # and importing it would fail for unrelated reasons). The find_spec
        # gate keeps the warning targeted at the actual DirectML drift case.
        self.assertIn(
            "importlib.util.find_spec('torch_directml')",
            run,
            "run.bat DirectML preflight must gate on find_spec('torch_directml') "
            "so CPU-only and CUDA-only setups are not warned spuriously.",
        )

        # SetErrorMode(0x8001) is the linchpin: without it, the import that
        # detects the drift would TRIGGER the popup we are trying to prevent.
        self.assertIn(
            "SetErrorMode(0x8001)",
            run,
            "run.bat DirectML preflight must SetErrorMode(0x8001) so the import "
            "that detects drift does not itself trigger the torch_library_impl popup.",
        )
        self.assertIn(
            "import torch, torchaudio, torch_directml",
            run,
            "run.bat DirectML preflight must import ALL THREE of torch+torchaudio+"
            "torch_directml under SetErrorMode (drift fires on torchaudio).",
        )

        # The preflight must steer users to the dedicated fix script when
        # drift is detected.
        self.assertIn(
            "fix_directml_pytorch.bat",
            run,
            "run.bat preflight warning must direct users to fix_directml_pytorch.bat.",
        )
        self.assertIn(
            "torch_library_impl",
            run,
            "run.bat preflight warning must name the exact error string the user "
            "would otherwise see in the popup, so search engines and chat logs "
            "connect symptom -> fix.",
        )

        # The preflight must pre-clear CONTINUE before the set /p prompt so an
        # inherited environment value (e.g. CONTINUE=y exported by a parent
        # shell) cannot silently bypass the warning and launch into the broken
        # state. Pressing Enter at the prompt with a pre-cleared variable
        # leaves CONTINUE empty, which fails the /i "y" check and exits 1.
        preclear_pos = run.find('set "CONTINUE="')
        prompt_pos = run.find("set /p CONTINUE=")
        self.assertGreater(
            preclear_pos, 0,
            "run.bat preflight must pre-clear CONTINUE via 'set \"CONTINUE=\"' "
            "so an inherited env var cannot silently auto-accept the prompt.",
        )
        self.assertGreater(prompt_pos, 0,
                           "run.bat preflight must prompt with set /p CONTINUE.")
        self.assertLess(
            preclear_pos, prompt_pos,
            "run.bat preflight must pre-clear CONTINUE BEFORE the set /p prompt.",
        )

        # The preflight must run BEFORE the app launches; otherwise the dialog
        # fires from main.py before our warning gets a chance.
        preflight_pos = run.find("DirectML ABI preflight")
        launch_pos = run.rfind('"%PYTHON_EXE%" main.py')
        self.assertGreater(preflight_pos, 0,
                           "run.bat must contain the DirectML ABI preflight block.")
        self.assertGreater(launch_pos, 0, "run.bat must still launch main.py.")
        self.assertLess(
            preflight_pos, launch_pos,
            "run.bat DirectML preflight must run BEFORE 'python main.py' so the "
            "popup is intercepted before the app starts.",
        )


class DirectMlMatchedSetInstallContractTests(unittest.TestCase):
    """v5.5.11: setup.bat :install_directml_pytorch AND fix_directml_pytorch.bat
    must install torch-directml ALONE first, then discover the pinned torch
    version at runtime and install ``torchaudio==<that-version>`` + torchvision
    against it.

    The v5.5.10 install line was ``pip install --upgrade torch-directml torch
    torchvision torchaudio`` — a single resolver group. torch-directml's strict
    ``torch==2.4.1`` pin held, but torchaudio's loose pin let pip pick the LATEST
    torchaudio (e.g. 2.11.0) which needs torch 2.7+ ABI. Result on Intel Core
    Ultra 5 325 AI PC: ``OSError: [WinError 127] The specified procedure
    could not be found`` from ``_load_lib('_torchaudio')`` — the suppressed
    version of the same ``torch_library_impl`` popup v5.5.10 was trying to
    self-heal. The fix repaired DETECTION but not REPAIR.

    The two-step install closes the gap: install torch-directml first so its
    torch pin lands; then read the installed torch version via a temp-file
    redirect (the only batch idiom that survives nested quote-stripping in
    cmd /c) and pin torchaudio to the same version. Future-proof: when MS
    bumps torch-directml to a new torch, the script auto-adapts.
    """

    def _assert_matched_set_install(self, block, script_label, py_var):
        # Step 1: torch-directml must appear ALONE on a pip install line, not
        # bundled with torch+torchvision+torchaudio (the v5.5.10 bug). Match
        # only actual pip install invocation lines (which start with a quoted
        # python path), not `::` batch comments that may quote the old bug.
        bundled_pattern = re.compile(
            r'^[ \t]*"[^"\r\n]+"\s+-m\s+pip\s+install[^\n\r]*\btorch-directml\b[^\n\r]*\btorchaudio\b',
            re.MULTILINE | re.IGNORECASE,
        )
        self.assertIsNone(
            bundled_pattern.search(block),
            f"{script_label} must NOT install torch-directml and torchaudio on the same pip "
            "install line. That lets pip's resolver pick the LATEST torchaudio (e.g. 2.11.0) "
            "against torch-directml's strict torch pin (e.g. 2.4.1), producing "
            "WinError 127 / 'torch_library_impl in _torchaudio.pyd' (v5.5.10 bug).",
        )
        reverse_pattern = re.compile(
            r'^[ \t]*"[^"\r\n]+"\s+-m\s+pip\s+install[^\n\r]*\btorchaudio\b[^\n\r]*\btorch-directml\b',
            re.MULTILINE | re.IGNORECASE,
        )
        self.assertIsNone(
            reverse_pattern.search(block),
            f"{script_label} must not bundle torchaudio with torch-directml in either order.",
        )

        # Step 2: torchaudio MUST be pinned to the discovered torch version via
        # the %TORCH_VER% variable. A bare 'pip install torchaudio' (or
        # --upgrade torchaudio) would re-introduce the drift.
        #
        # v5.5.11 (SQT test-engineer nit, 2026-05-26): Anchor this to the
        # actual `-m pip install ... torchaudio==%TORCH_VER%` line, not just
        # any line containing the string — both scripts also echo
        # "[ERROR] Matched torchaudio==%TORCH_VER% / torchvision install failed."
        # on the failure path, so a plain assertIn would silently pass even
        # if a regression hardcoded the version on the actual install line
        # and only the echo string survived. The regex requires the substring
        # to live on the pip install invocation itself.
        pinned_install_pattern = re.compile(
            r'-m\s+pip\s+install[^\n\r]*"torchaudio==%TORCH_VER%"',
            re.IGNORECASE,
        )
        self.assertRegex(
            block,
            pinned_install_pattern,
            f"{script_label} must install torchaudio pinned to the discovered torch version "
            "(\"torchaudio==%TORCH_VER%\") on the actual `-m pip install` line so it matches "
            "torch-directml's torch pin. A diagnostic echo containing the same string is "
            "not sufficient.",
        )

        # Step 3: the torch version discovery must use the temp-file redirect
        # pattern. cmd /c quote-stripping breaks the `for /f ('python -c ...')`
        # form when the python command contains both quoted args AND a
        # redirect, so the only reliable batch idiom is:
        #
        #   "%PYTHON_EXE%" -c "..." > "%VFILE%" 2>nul
        #   set /p TORCH_VER=<"%VFILE%"
        self.assertIn(
            "torch.__version__.split(chr(43))[0]",
            block,
            f"{script_label} must discover the BASE torch version (strip the +cpu/+cu* "
            "suffix using chr(43)='+' to avoid an embedded single-quote that would "
            "collide with cmd quoting).",
        )
        self.assertIn(
            "set /p TORCH_VER=",
            block,
            f"{script_label} must capture the discovered torch version via 'set /p "
            "TORCH_VER=<file' (the for /f ('cmd') form is unreliable here due to "
            "cmd /c quote-stripping on nested double-quoted args).",
        )

        # The two-step install ordering: torch-directml first, version
        # discovery, then matched torchaudio install.
        first_install_pos = block.find("pip install --upgrade")
        # The python -c that captures the torch version
        discover_pos = block.find("torch.__version__.split(chr(43))[0]")
        # The pinned torchaudio install
        matched_install_pos = block.find("torchaudio==%TORCH_VER%")
        self.assertGreater(first_install_pos, 0,
                           f"{script_label} must still issue 'pip install --upgrade ... torch-directml'.")
        self.assertGreater(discover_pos, 0,
                           f"{script_label} must include the torch version discovery snippet.")
        self.assertGreater(matched_install_pos, 0,
                           f"{script_label} must include the pinned torchaudio install.")
        self.assertLess(
            first_install_pos, discover_pos,
            f"{script_label} must install torch-directml BEFORE discovering the torch version.",
        )
        self.assertLess(
            discover_pos, matched_install_pos,
            f"{script_label} must discover the torch version BEFORE installing "
            "torchaudio==%TORCH_VER%.",
        )

        # The Python executable variable name must match the script's
        # convention. setup.bat uses %PYTHON_EXE%, fix_directml_pytorch.bat
        # uses %LOCALAI_PYTHON%. Confirm the discovery line uses the right one.
        py_var_pattern = re.compile(
            r'"%' + re.escape(py_var) + r'%"\s+-c\s+"import torch;\s*print\(torch\.__version__\.split\(chr\(43\)\)\[0\]\)"',
        )
        self.assertRegex(
            block,
            py_var_pattern,
            f"{script_label} torch-version discovery must invoke %{py_var}% (the script's "
            "own Python executable variable), not a different name.",
        )

    def test_setup_install_directml_pytorch_uses_matched_set_install(self):
        setup = read("setup1.bat")
        start = setup.index("\n:install_directml_pytorch")
        end = setup.index("\n:find_python", start)
        block = setup[start:end]
        self._assert_matched_set_install(block, "setup.bat :install_directml_pytorch", "PYTHON_EXE")

    def test_fix_directml_pytorch_uses_matched_set_install(self):
        fix = read("fix_directml_pytorch.bat")
        self._assert_matched_set_install(fix, "fix_directml_pytorch.bat", "LOCALAI_PYTHON")


class FixDirectMlPytorchSetErrorModeContractTests(unittest.TestCase):
    """v5.5.11 (Ron, 2026-05-26): fix_directml_pytorch.bat post-install
    verify MUST wrap its `import torch, torchaudio, torch_directml` line
    with `ctypes.windll.kernel32.SetErrorMode(0x8001)` to suppress the
    Windows "The specified procedure could not be found" popup dialog
    (WinError 127) if a stale _torchaudio.pyd somehow still mismatches.

    setup.bat:install_directml_pytorch has done this since v5.5.10; this
    test pins the SQT setup-release nit catch for the fix script.
    """

    def test_verify_import_wraps_seterrormode(self):
        fix = read("fix_directml_pytorch.bat")
        verify_pattern = re.compile(
            r'-c\s+"[^"\n\r]*ctypes\.windll\.kernel32\.SetErrorMode\(0x8001\)[^"\n\r]*'
            r'import\s+torch,\s*torchaudio,\s*torch_directml',
            re.IGNORECASE,
        )
        self.assertRegex(
            fix,
            verify_pattern,
            "fix_directml_pytorch.bat post-install verify must call "
            "ctypes.windll.kernel32.SetErrorMode(0x8001) BEFORE importing "
            "torch/torchaudio/torch_directml so WinError 127 is suppressed "
            "to the console (no modal popup). See setup.bat:install_directml_pytorch "
            "for the matching pattern.",
        )


class IntegratedGpuUnifiedMemoryContractTests(unittest.TestCase):
    """v5.5.11 (Ron, 2026-05-26): Windows integrated GPUs (Intel Iris/UHD/Arc
    Graphics, AMD Radeon Graphics, Snapdragon Adreno) MUST be treated as
    unified memory in system_info, with vram_total/vram_free reported as
    system RAM total/available rather than the small dedicated AdapterRAM
    carve-out. Without this, can_run_model fires "Not enough GPU VRAM: only
    1.7 GB free on Intel Arc Graphics" on AI PCs that actually have
    13+ GB of shared system memory available to the iGPU through DXGI.

    These tests stub the powershell + RAM lookup so the assertion runs on
    any platform (CI / Linux / Mac dev boxes) without needing real Intel
    hardware.
    """

    def test_wmi_fallback_marks_intel_igpu_as_unified(self):
        import json as _json
        from unittest.mock import patch
        from src import system_info as si

        fake_wmi = _json.dumps([
            {"Name": "Intel(R) Arc(TM) Graphics", "AdapterRAM": 2 * 1024 * 1024 * 1024},
        ])

        # The WMI fallback only runs if nvidia-smi + rocm-smi return nothing
        # parseable. Make the _run side_effect dispatch by command so other
        # probes don't accidentally yield the WMI JSON or crash on the wrong
        # shape (the rocm-smi parser does data.items(), which AttributeErrors
        # on a list payload and isn't caught by its except clause).
        def _fake_run(cmd, *args, **kwargs):
            if cmd and cmd[0] == "powershell":
                return fake_wmi
            return ""

        fake_ram = {"total_mb": 32_000, "available_mb": 13_000}

        with patch.object(si, "_run", side_effect=_fake_run), \
             patch.object(si, "get_ram_info", return_value=fake_ram):
            gpus = si._get_gpu_info_windows()

        self.assertTrue(gpus, "WMI fallback must surface the Intel Arc iGPU.")
        gpu = gpus[0]
        self.assertEqual(gpu.get("vendor"), "Intel")
        self.assertTrue(
            gpu.get("unified_memory"),
            "Intel iGPU detected via WMI fallback must be marked unified_memory=True "
            "so can_run_model uses the system-RAM gate instead of the dedicated AdapterRAM gate.",
        )
        self.assertEqual(
            gpu.get("vram_total_mb"), 32_000,
            "Unified iGPU vram_total_mb must report system RAM total, not AdapterRAM.",
        )
        self.assertEqual(
            gpu.get("vram_free_mb"), 13_000,
            "Unified iGPU vram_free_mb must report system RAM available.",
        )
        self.assertEqual(
            gpu.get("dedicated_vram_mb"), 2048,
            "Unified iGPU must preserve the WMI AdapterRAM value in dedicated_vram_mb "
            "for system-page display.",
        )

    def test_wmi_fallback_marks_amd_radeon_graphics_as_unified(self):
        import json as _json
        from unittest.mock import patch
        from src import system_info as si

        fake_wmi = _json.dumps([
            {"Name": "AMD Radeon(TM) Graphics", "AdapterRAM": 512 * 1024 * 1024},
        ])

        def _fake_run(cmd, *args, **kwargs):
            if cmd and cmd[0] == "powershell":
                return fake_wmi
            return ""

        fake_ram = {"total_mb": 16_000, "available_mb": 9_000}

        with patch.object(si, "_run", side_effect=_fake_run), \
             patch.object(si, "get_ram_info", return_value=fake_ram):
            gpus = si._get_gpu_info_windows()

        self.assertTrue(gpus)
        gpu = gpus[0]
        self.assertEqual(gpu.get("vendor"), "AMD")
        self.assertTrue(gpu.get("unified_memory"))
        self.assertEqual(gpu.get("vram_total_mb"), 16_000)
        self.assertEqual(gpu.get("dedicated_vram_mb"), 512)

    def test_can_run_model_uses_ram_gate_for_unified_igpu(self):
        """End-to-end gate: a 7 GB model on an Intel Arc iGPU with 13 GB
        shared RAM available MUST be allowed (not the v5.5.10 false-fail
        path that complained about 1.7 GB free dedicated VRAM).

        Designed to be behaviorally distinguishable from the dedicated-VRAM
        branch: vram_free_mb is set BELOW min_vram so the dedicated-VRAM
        gate would FAIL with "only 2 GB free", and only the unified-memory
        branch (which reads from system RAM via fake_ram) can return OK.
        This guards against typo/branch-removal regressions where the
        unified path is silently bypassed (the gap caught by SQT mutation
        testing in v5.5.11)."""
        from unittest.mock import patch
        from src import system_info as si

        model = {"name": "qwen2.5-7b", "size_gb": 4.5, "min_ram_gb": 8, "min_vram_gb": 6}
        fake_gpus = [{
            "name": "Intel(R) Arc(TM) Graphics",
            "vendor": "Intel",
            "vram_total_mb": 32_000,
            # 2 GB free is BELOW the 6 GB min_vram requirement. If the
            # unified-memory branch is bypassed, the dedicated-VRAM gate
            # will fire and return (False, "Not enough GPU VRAM..."). Only
            # the unified branch (which uses system RAM total/available)
            # can return OK here.
            "vram_free_mb": 2_048,
            "vram_used_mb": 30_000,
            "dedicated_vram_mb": 2_048,
            "type": "GPU",
            "unified_memory": True,
        }]
        fake_ram = {"total_mb": 32_000, "available_mb": 13_000}

        with patch.object(si, "get_gpu_info", return_value=fake_gpus), \
             patch.object(si, "get_ram_info", return_value=fake_ram):
            ok, msg = si.can_run_model(model, gpu_index=0)

        self.assertTrue(
            ok,
            f"can_run_model must allow a 6 GB-VRAM-required model on a 32 GB "
            f"unified iGPU with 13 GB free SYSTEM RAM, but failed with: {msg!r}. "
            f"This usually means the unified-memory branch was bypassed and the "
            f"dedicated-VRAM gate fired against vram_free_mb=2_048 (only 2 GB).",
        )

    def test_wmi_fallback_does_not_mark_intel_arc_discrete_as_unified(self):
        """SQT setup-release NIT-1: discrete Intel Arc A### / B### cards
        (A380, A580, A750, A770, B570, B580, B770) are NOT detected by
        nvidia-smi or rocm-smi and fall through the WMI fallback. They
        have real, dedicated GDDR6 VRAM — they MUST NOT be flagged as
        unified_memory or can_run_model will route through the system-RAM
        gate and block models that would actually fit in dedicated VRAM."""
        import json as _json
        from unittest.mock import patch
        from src import system_info as si

        # AdapterRAM is uint32-capped near 4 GB for >= 4 GB cards. Use the
        # actual cap value to mimic what WMI returns for an A770 (16 GB).
        fake_wmi = _json.dumps([
            {"Name": "Intel(R) Arc(TM) A770 Graphics", "AdapterRAM": 4_293_918_720},
        ])

        def _fake_run(cmd, *args, **kwargs):
            if cmd and cmd[0] == "powershell":
                return fake_wmi
            return ""

        fake_ram = {"total_mb": 32_000, "available_mb": 13_000}

        with patch.object(si, "_run", side_effect=_fake_run), \
             patch.object(si, "get_ram_info", return_value=fake_ram):
            gpus = si._get_gpu_info_windows()

        self.assertTrue(gpus, "WMI fallback must still surface the Intel Arc A770.")
        gpu = gpus[0]
        self.assertEqual(gpu.get("vendor"), "Intel")
        self.assertFalse(
            gpu.get("unified_memory"),
            "Intel Arc A770 is a DISCRETE GDDR6 GPU — must NOT be marked "
            "unified_memory. Marking it unified routes through the system-RAM "
            "gate, blocking models that would otherwise fit in the dedicated VRAM.",
        )
        # vram_total_mb should reflect dedicated VRAM (uint32-capped), not system RAM.
        self.assertLess(
            gpu.get("vram_total_mb"), 5_000,
            "Discrete Intel Arc vram_total_mb must report AdapterRAM (~4 GB), "
            "not system RAM (32 GB).",
        )


class ModelDropdownSpeedSortContractTests(unittest.TestCase):
    """v5.5.11 (Ron, 2026-05-26): Both the Chat page model dropdown and the
    Image Generation page model dropdown MUST sort fastest -> slowest using
    catalog `size_gb` as the speed proxy. Smaller models load and infer
    faster on the same hardware, so this gives users a consistent
    "fastest at the top" experience without needing per-SKU TPS lookup.
    """

    def test_chat_model_entries_sorts_by_size_gb_ascending(self):
        """Inspect the _chat_model_entries source to confirm size_gb is the
        primary sort key (not name/category). This is a source-shape test
        because constructing a full App() requires Tk + customtkinter."""
        app_src = read("src/app.py")
        # Find the _chat_model_entries function body
        start = app_src.find("def _chat_model_entries(self)")
        self.assertGreater(start, 0, "Could not locate _chat_model_entries in src/app.py")
        end = app_src.find("\n    def ", start + 1)
        body = app_src[start:end]
        # The sort key tuple must lead with _size_gb(m) so size_gb is the
        # primary axis. A regression that reorders the tuple (e.g.
        # name -> category -> size_gb) would silently break fastest-first.
        self.assertRegex(
            body,
            re.compile(
                r"sorted\(\s*models\s*,\s*key=lambda\s+m\s*:\s*\(\s*_size_gb\(m\)\s*,",
                re.DOTALL,
            ),
            "_chat_model_entries must sort with _size_gb(m) as the FIRST key "
            "(fastest models first). DO NOT reorder the key tuple.",
        )

    def test_populate_image_model_menu_sorts_by_size_gb_ascending(self):
        """Inspect _populate_image_model_menu source to confirm it sorts by
        catalog size_gb ascending rather than alphabetically by label."""
        app_src = read("src/app.py")
        start = app_src.find("def _populate_image_model_menu(self")
        self.assertGreater(start, 0, "Could not locate _populate_image_model_menu in src/app.py")
        end = app_src.find("\n    def ", start + 1)
        body = app_src[start:end]
        # The new sort key must lead with size_gb (a float). The old v5.5.10
        # implementation was `pairs.sort(key=lambda p: p[0].lower())` which
        # sorted purely alphabetically by display label.
        self.assertNotIn(
            "pairs.sort(key=lambda p: p[0].lower())",
            body,
            "_populate_image_model_menu must NOT sort alphabetically by display label "
            "(v5.5.10 behavior). It must sort by catalog size_gb ascending so the "
            "fastest image models surface at the top of the dropdown.",
        )
        self.assertRegex(
            body,
            re.compile(
                r"_find_catalog_entry_for_model\(fn\)",
                re.DOTALL,
            ),
            "_populate_image_model_menu must look up each model's catalog "
            "entry (via _find_catalog_entry_for_model) to obtain size_gb for sorting.",
        )
        self.assertRegex(
            body,
            re.compile(
                r"triples\.sort\(\s*key=lambda\s+t\s*:\s*\(\s*t\[0\]\s*,",
                re.DOTALL,
            ),
            "_populate_image_model_menu must sort the (size_gb, label, filename) "
            "triples with size_gb as the first key. DO NOT reorder the tuple.",
        )

    def test_chat_dropdown_pushes_zero_or_missing_size_gb_to_end(self):
        """v5.5.12 (Ron, 2026-05-27): 10 catalog entries (Llama 3.3 70B,
        Qwen3 30B-A3B, Mistral Nemo, etc.) ship with ``size_gb=0`` because
        their actual quantized size is sourced at download time. Treating 0
        as "smallest" floats huge unsized models to the TOP of the dropdown,
        which is exactly what Ron hit on his Intel Core Ultra 5 325
        ("Llama 3.3 70B was first, DeepSeek-R1 32B was last"). The contract:
        ``_size_gb`` helper inside ``_chat_model_entries`` must map 0, None,
        and non-numeric values to ``float('inf')`` so they sort to the END,
        with real-sized small models (Qwen 2.5 0.5B at 0.4 GB, etc.) leading.
        """
        app_src = read("src/app.py")
        start = app_src.find("def _chat_model_entries(self)")
        self.assertGreater(start, 0)
        end = app_src.find("\n    def ", start + 1)
        body = app_src[start:end]
        # The v5.5.11 regression returned 0.0 from _size_gb for None/0
        # (because `model.get("size_gb") or 0` collapses both to 0 and the
        # function returned that directly). The fix MUST gate the return
        # on `val > 0` (or equivalent) and substitute float('inf') so
        # unsized models sort to the end.
        self.assertRegex(
            body,
            re.compile(r"val\s*>\s*0\s*else\s*float\([\"']inf[\"']\)"),
            "_chat_model_entries._size_gb must explicitly map val<=0 to "
            "float('inf'). Pattern required: `return val if val > 0 else "
            "float('inf')`. Without this gate, 10 catalog entries with "
            "size_gb=0 (Llama 3.3 70B, Qwen3 30B-A3B, Mistral Nemo, etc.) "
            "float to the TOP of the dropdown — Ron's v5.5.11 bug.",
        )
        self.assertRegex(
            body,
            re.compile(r"float\([\"']inf[\"']\)"),
            "_chat_model_entries._size_gb must return float('inf') for "
            "size_gb<=0 / None / non-numeric so unsized models sort to the END, "
            "not the TOP, of the chat dropdown.",
        )

    def test_image_dropdown_pushes_zero_or_missing_size_gb_to_end(self):
        """v5.5.12 (Ron, 2026-05-27): Same contract as the chat dropdown,
        applied to the image dropdown. SDXL Low VRAM has ``size_gb=0`` in
        the catalog (size depends on UNet variant) — it must sort to the
        END, not the TOP. The existing v5.5.11 image dropdown code already
        falls back to ``float('inf')`` when the catalog entry is missing
        AND when the catalog entry has size_gb=None, but should also do so
        when size_gb is explicitly 0.
        """
        app_src = read("src/app.py")
        start = app_src.find("def _populate_image_model_menu(self")
        self.assertGreater(start, 0)
        end = app_src.find("\n    def ", start + 1)
        body = app_src[start:end]
        # Locate the size_gb extraction block. Must convert 0/None/non-numeric
        # to inf so unsized image models don't surface at the top.
        self.assertIn(
            'float("inf")',
            body,
            "_populate_image_model_menu must use float('inf') as the size_gb "
            "fallback so unsized image models (SDXL Low VRAM with size_gb=0) "
            "sort to the END of the dropdown, not the TOP.",
        )


class IgpuViableImageModelContractTests(unittest.TestCase):
    """v5.5.12 (Ron, 2026-05-27): The catalog helper
    ``is_igpu_viable_image_model`` determines which image models can complete
    inside Windows' DXGI TDR (~2s per kernel) on integrated GPUs. SD 1.5
    family models pass; SDXL/Flux fail except SDXL Lightning (step-reduced).
    The helper backs the iGPU filter in ``_populate_image_model_menu``.
    """

    def test_cpu_viable_models_are_igpu_viable(self):
        from src.catalog import is_igpu_viable_image_model
        # cpu_viable=True models (SD 1.5 family) are automatically iGPU-viable
        # — their small UNet fits a single iGPU step inside TDR.
        self.assertTrue(
            is_igpu_viable_image_model({"backend": "comfyui", "cpu_viable": True}),
            "cpu_viable image models (SD 1.5 family) MUST be iGPU-viable "
            "because their per-step latency on iGPU is well inside DXGI TDR.",
        )

    def test_explicit_igpu_viable_flag_is_honored(self):
        from src.catalog import is_igpu_viable_image_model
        # SDXL Lightning has cpu_viable=False (SDXL UNet is too slow on CPU)
        # but is iGPU-viable at 512x512 / 4 steps. The igpu_viable flag is
        # the catalog's escape hatch for these step-reduced architectures.
        self.assertTrue(
            is_igpu_viable_image_model(
                {"backend": "comfyui", "cpu_viable": False, "igpu_viable": True}
            ),
            "Models flagged igpu_viable=True (SDXL Lightning) MUST be iGPU-viable "
            "even when cpu_viable=False.",
        )

    def test_unflagged_heavy_models_are_not_igpu_viable(self):
        from src.catalog import is_igpu_viable_image_model
        # Default SDXL/Flux models with neither cpu_viable nor igpu_viable
        # MUST NOT pass — they reliably TDR on iGPUs.
        self.assertFalse(
            is_igpu_viable_image_model({"backend": "comfyui"}),
            "Image models with neither cpu_viable nor igpu_viable MUST NOT be "
            "iGPU-viable. Default SDXL/Flux entries hitting iGPU TDR was Ron's "
            "v5.5.11 failure case.",
        )

    def test_non_comfyui_backends_are_not_igpu_viable(self):
        from src.catalog import is_igpu_viable_image_model
        # backend!=comfyui (e.g. external API) is out of scope for this
        # gating — they don't go through the dropdown's TDR-safety filter.
        self.assertFalse(
            is_igpu_viable_image_model(
                {"backend": "ollama", "cpu_viable": True, "igpu_viable": True}
            ),
            "Only backend=comfyui image models go through the iGPU TDR filter.",
        )

    def test_catalog_flags_three_models_igpu_viable(self):
        """The catalog MUST flag exactly the three iGPU-survivable models:
        Realistic Vision V6, Counterfeit V3.0, SDXL Lightning. Adding new
        models requires explicit benchmark validation on a real iGPU; this
        test pins the set so a careless flag add can't sneak through.
        """
        import json
        catalog_path = ROOT / "models_catalog.json"
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
        models = data.get("models", data) if isinstance(data, dict) else data
        igpu_flagged = sorted(
            m.get("name") for m in models if m.get("igpu_viable")
        )
        self.assertEqual(
            igpu_flagged,
            ["Counterfeit V3.0", "Realistic Vision v6.0", "SDXL Lightning"],
            "Catalog igpu_viable set drifted. v5.5.12 ships exactly these 3. "
            "Adding new models requires benchmark validation on a real Windows "
            "iGPU (Intel Arc Graphics / AMD Radeon Graphics / Snapdragon Adreno) "
            "to confirm they complete inside DXGI TDR. Update this test only when "
            "you have that validation in hand.",
        )


class IgpuImageGenSafetyContractTests(unittest.TestCase):
    """v5.5.12 (Ron, 2026-05-27): On Windows-integrated GPUs, the Image Gen
    UX MUST (1) filter the dropdown to iGPU-viable models, (2) clamp default
    steps/resolution to TDR-safe ranges, (3) surface a friendly error dialog
    when DXGI TDR fires. Ron's failure case: Intel Core Ultra 5 325 + Arc
    Graphics iGPU + Realistic Vision V6 at 768x512 / 30 steps -> "GPU device
    instance has been suspended" with no user-actionable guidance.
    """

    def test_windows_unified_igpu_flag_cached_at_gpu_detection(self):
        """``_apply_gpu_detection_result`` must set ``self._windows_unified_igpu``
        based on ``system_info.get_gpu_info()`` so subsequent ImageGen lookups
        don't re-query WMI. The cache must distinguish Windows iGPUs from
        Apple Silicon unified-memory GPUs (no TDR on Metal).
        """
        app_src = read("src/app.py")
        start = app_src.find("def _apply_gpu_detection_result")
        self.assertGreater(start, 0)
        end = app_src.find("\n    def ", start + 1)
        body = app_src[start:end]
        self.assertIn(
            "_windows_unified_igpu",
            body,
            "_apply_gpu_detection_result must set self._windows_unified_igpu so "
            "subsequent Image Gen calls don't re-query WMI.",
        )
        self.assertIn(
            'sys.platform == "win32"',
            body,
            "_apply_gpu_detection_result must gate the iGPU detection on "
            "sys.platform == 'win32' so macOS unified-memory hosts (Apple "
            "Silicon, no TDR) don't get falsely flagged.",
        )
        self.assertRegex(
            body,
            re.compile(r'vendor.*Apple', re.DOTALL),
            "Windows-iGPU detection must exclude Apple vendor — Apple Silicon "
            "is unified_memory but has no TDR (Metal has no per-kernel timeout).",
        )

    def test_image_dropdown_filters_to_igpu_viable_on_windows_unified(self):
        """``_populate_image_model_menu`` must filter to
        ``is_igpu_viable_image_model`` matches when
        ``self._windows_unified_igpu`` is True.
        """
        app_src = read("src/app.py")
        start = app_src.find("def _populate_image_model_menu(self")
        self.assertGreater(start, 0)
        end = app_src.find("\n    def ", start + 1)
        body = app_src[start:end]
        self.assertIn(
            "_windows_unified_igpu",
            body,
            "_populate_image_model_menu must check self._windows_unified_igpu to "
            "decide whether to apply the iGPU TDR-safety filter.",
        )
        self.assertIn(
            "is_igpu_viable_image_model",
            body,
            "_populate_image_model_menu must call catalog.is_igpu_viable_image_model "
            "to filter the dropdown on Windows-integrated GPUs.",
        )

    def test_igpu_safe_defaults_clamp_helper_exists(self):
        """A dedicated ``_clamp_image_params_for_igpu`` helper must clamp
        steps and resolution to TDR-safe ranges. Anchored to specific
        numeric ceilings (10 steps, 512px) so a regression that loosens
        the limits is caught.
        """
        app_src = read("src/app.py")
        start = app_src.find("def _clamp_image_params_for_igpu(self")
        self.assertGreater(
            start, 0,
            "_clamp_image_params_for_igpu helper missing. Must clamp steps<=10 "
            "and width/height<=512 on Windows-integrated GPUs to prevent DXGI "
            "TDR on first generation.",
        )
        end = app_src.find("\n    def ", start + 1)
        body = app_src[start:end]
        self.assertIn(
            "_windows_unified_igpu",
            body,
            "Clamp must early-return when _windows_unified_igpu is False (Apple "
            "Silicon and discrete GPUs don't need TDR-safe defaults).",
        )
        # Steps ceiling
        self.assertRegex(
            body,
            re.compile(r"steps\s*>\s*10"),
            "Clamp must cap steps at 10 on iGPUs (Ron's RV6 failure was at 30).",
        )
        self.assertIn(
            '"10"',
            body,
            "Clamp must SET steps to '10' (not just compare).",
        )
        # Resolution ceiling
        self.assertRegex(
            body,
            re.compile(r"val\s*>\s*512"),
            "Clamp must cap width/height at 512 on iGPUs (Ron's RV6 failure "
            "was at 768x512 — even one axis above 512 can TDR).",
        )

    def test_igpu_clamp_called_from_both_model_changed_paths(self):
        """Both the catalog-driven path AND the heuristic fallback path of
        ``_on_img_model_changed`` must call ``_clamp_image_params_for_igpu``
        AFTER applying defaults. A regression that adds clamp to only one
        path would silently leak unsafe defaults for user-installed
        checkpoints that lack a catalog entry.
        """
        app_src = read("src/app.py")
        start = app_src.find("def _on_img_model_changed(self")
        self.assertGreater(start, 0)
        end = app_src.find("\n    def ", start + 1)
        body = app_src[start:end]
        count = body.count("_clamp_image_params_for_igpu()")
        self.assertGreaterEqual(
            count, 2,
            f"_on_img_model_changed must call _clamp_image_params_for_igpu() "
            f"from BOTH the catalog path AND the heuristic path. Found {count} "
            "call(s); expected at least 2. Otherwise user-installed checkpoints "
            "(which take the heuristic path) bypass the iGPU safety clamp.",
        )

    def test_device_suspended_dialog_dispatched_in_image_failure(self):
        """``_img_generation_failed`` must surface a friendly dialog when the
        error contains DXGI TDR markers ("device instance has been suspended"
        or "GetDeviceRemovedReason"). The dialog must mention the 2-second
        per-kernel timeout AND suggest concrete remediations (lower steps,
        512x512, smaller model).
        """
        app_src = read("src/app.py")
        start = app_src.find("def _img_generation_failed(self")
        self.assertGreater(start, 0)
        end = app_src.find("\n    def ", start + 1)
        body = app_src[start:end]
        self.assertIn(
            "device instance has been suspended",
            body,
            "_img_generation_failed must detect the 'device instance has been "
            "suspended' DXGI TDR error string Ron's iGPU emits and route it to "
            "a friendly dialog instead of leaving users staring at a cryptic "
            "ComfyUI execution error.",
        )
        self.assertIn(
            "getdeviceremovedreason",
            body,
            "_img_generation_failed must also match 'GetDeviceRemovedReason' "
            "(the lowercase comparison form) — Win32's docs name this API "
            "explicitly in the error string.",
        )
        # Confirm dialog text hits the three remediations
        self.assertIn("Lower steps", body)
        self.assertIn("512", body)
        self.assertIn("SDXL Lightning", body)


class Phi4MiniSamplePromptContractTests(unittest.TestCase):
    """Sample-2 + sample-3 regressions (Ron, 2026-05-31 / 2026-06-01): on a
    24 GB-VRAM partition extended run (and reproduced on a workstation-class
    GPU), TWO phi4:mini prompts triggered the same *degenerate repetition
    loop* failure mode (failure_phase=output_truncated, ~265s wasted per
    sample, 4096-token budget exhausted before any stop reason).

    **Sample 2** (fixed 2026-05-31): code prompt that asked for ``outside
    2-8 C`` — the model parsed the hyphen as a minus sign, wrote a
    docstring claiming the range was -10..+50 and a wrong predicate
    ``(-20..+50) | (-1..+9)``, then drifted into a free-form ``Note:``
    explanation that spiraled into 4000+ tokens of word salad. Two root
    causes: ambiguous range notation, and a missing ``and no
    explanation`` stop signal.

    **Sample 3** (fixed 2026-06-01): advisory prompt that asked for
    "Give a three-bullet recommendation using evidence, tradeoff, and
    next experiment." The model wrote three reasonable bullets, then a
    "Recommendation:" section with three more sub-bullets (overshoot),
    then "In conclusion..." / "In summary..." paragraphs that repeated
    the same content verbatim 4+ times until truncation. Root cause:
    open-ended advisory prompt with NO hard length budget AND NO
    anti-drift stop signal — sample 3 was deliberately excluded from
    sample-2's ``no explanation`` rule (it's not a code prompt), but
    that left it with no failsafe at all.

    These tests lock the curated rewrite so the failure cannot resurface:
    sample-2 MUST use the function name ``flag_out_of_range``, MUST cite
    the unambiguous "2 to 8 (degrees Celsius)" range, BOTH code samples
    (1 and 2) MUST end with a terminating stop signal (``and stop``) and
    MUST NOT use the empirically-failing ``exactly <N> assert tests``
    self-test pattern (2026-06-17 verification on CPC-ronma-4GM0M), AND
    sample-3 MUST carry a hard word budget AND an explicit "do not write
    any introduction, conclusion, or summary" anti-drift directive.
    """

    def _phi4_mini_samples(self):
        from src.sample_prompts import MODEL_DEMO_SAMPLE_OVERRIDES
        samples = MODEL_DEMO_SAMPLE_OVERRIDES.get("phi4:mini", [])
        self.assertEqual(
            len(samples), 3,
            "phi4:mini override must keep exactly 3 curated samples "
            "(code-with-fn-signature, code-with-fn-signature, "
            "advisory-bullets). Adding/removing changes the benchmark "
            "comparison surface and must be a deliberate decision.",
        )
        return samples

    def test_phi4_mini_sample_2_uses_unambiguous_range_notation(self):
        """Sample-2 MUST NOT contain the ambiguous ``\\d+-\\d+ C`` pattern
        (e.g. ``2-8 C``). On phi4:mini the hyphen is read as a minus sign,
        producing wrong code and triggering the degenerate-loop failure
        Ron reported across multiple GPU configurations.
        """
        samples = self._phi4_mini_samples()
        sample_2 = samples[1]
        self.assertNotRegex(
            sample_2,
            re.compile(r"\d+\s*-\s*\d+\s*C\b"),
            f"phi4:mini sample-2 reverted to the ambiguous '\\d-\\d C' range "
            f"pattern. Use 'A to B (degrees Celsius)' or 'A°C to B°C' instead. "
            f"Sample text: {sample_2!r}",
        )

    def test_phi4_mini_sample_2_pins_curated_function_name(self):
        """The curated rewrite uses a concrete function name
        (``flag_out_of_range``) mirroring sample-1's ``fits_job``. The
        function-name + bounded-range + stop-signal combination is what
        empirically stops phi4:mini from drifting into the degenerate
        loop.
        """
        samples = self._phi4_mini_samples()
        sample_2 = samples[1]
        self.assertIn(
            "flag_out_of_range",
            sample_2,
            "phi4:mini sample-2 lost its concrete function-name anchor "
            "(`flag_out_of_range`). Without a named function the model "
            "writes a more verbose 'here is how you might' preamble that "
            "drifts toward the v5.5.x degenerate-loop failure.",
        )
        self.assertRegex(
            sample_2,
            re.compile(r"2\s*to\s*8.*[Cc]elsius|2\s*°?C\s*to\s*8\s*°?C", re.DOTALL),
            f"phi4:mini sample-2 must spell out the range explicitly "
            f"('2 to 8 (degrees Celsius)' or '2°C to 8°C'). Sample text: "
            f"{sample_2!r}",
        )

    def test_phi4_mini_code_prompts_have_terminating_stop_signal(self):
        """Sample-1 and sample-2 are both code-generation prompts. Each MUST
        end with an explicit terminating stop signal AND must NOT use the
        ``exactly <N> assert tests`` self-test pattern.

        History: the 2026-05-31 curation added an ``and no explanation`` stop
        signal to the code prompts. The 2026-06-17 extended-bench verification
        on a CPU-only Cloud PC (CPC-ronma-4GM0M) showed that was INSUFFICIENT:
        the code prompts that asked for ``exactly three assert tests and no
        explanation`` still spiralled — phi4:mini wrote a function, then
        second-guessed its own asserts (``Note: the above is incorrect,
        version 2...``) and looped to the 4096-token ceiling. The verified fix
        drops the self-test-critique trap and asks instead for a concrete,
        naturally-terminating deliverable ("show example calls and the value
        each returns, and stop"). That rewrite passed 3/3 on the same box.

        This test pins the verified shape so the failure cannot resurface:

          1. each code prompt ends with a terminating stop signal — ``and
             stop`` (the verified terminator) or the legacy ``no explanation``;
          2. neither code prompt uses the empirically-failing ``exactly <N>
             assert tests`` self-test pattern.
        """
        samples = self._phi4_mini_samples()
        stop_signal = re.compile(r"(and\s+stop|no\s+explanation)\b", re.IGNORECASE)
        assert_trap = re.compile(r"exactly\s+\S+\s+assert\s+tests?\b", re.IGNORECASE)
        for idx in (0, 1):
            text = samples[idx]
            self.assertRegex(
                text,
                stop_signal,
                f"phi4:mini sample-{idx+1} must end with a terminating stop "
                f"signal ('and stop' or 'no explanation'). Code prompts "
                f"without it let the model drift past the answer and risk the "
                f"degenerate-loop failure. Sample text: {text!r}",
            )
            self.assertNotRegex(
                text,
                assert_trap,
                f"phi4:mini sample-{idx+1} reverted to the 'exactly N assert "
                f"tests' self-test pattern that empirically triggered the "
                f"2026-06-17 degenerate loop on CPC-ronma-4GM0M. Ask for "
                f"'example calls and the value each returns, and stop' instead. "
                f"Sample text: {text!r}",
            )

    def test_phi4_mini_sample_3_has_hard_length_budget_and_anti_drift_signal(self):
        """Sample-3 is an advisory-bullets prompt (NOT a code prompt) and
        therefore cannot use the ``and no explanation`` rule that pins
        samples 1 and 2. To prevent the 2026-06-01 high-VRAM-workstation regression
        where the model spiraled into ``In conclusion / In summary``
        repetition until the 4096-token budget was exhausted, the
        curated prompt MUST carry BOTH:

        1. A hard length budget expressed in words (so the budget is
           parsed even by small models that mis-interpret token caps).
        2. An explicit anti-drift directive that names the failure
           shapes ("do not write any introduction, conclusion, or
           summary, and do not repeat any bullet").

        Both signals must appear together: an isolated word cap is not
        sufficient because phi4:mini will respect the bullet count but
        then add post-bullet prose; an isolated "no conclusion"
        directive is not sufficient because it inflates each bullet
        instead of looping.
        """
        samples = self._phi4_mini_samples()
        sample_3 = samples[2]
        self.assertRegex(
            sample_3.lower(),
            re.compile(r"under\s+\d+\s+words|\d+\s+words\s+or\s+(less|fewer)"),
            f"phi4:mini sample-3 must carry an explicit word budget such "
            f"as 'totaling under 60 words' or '60 words or less'. Without "
            f"a budget the model writes verbose bullets and triggers the "
            f"2026-06-01 degenerate-loop regression. Sample text: "
            f"{sample_3!r}",
        )
        self.assertRegex(
            sample_3.lower(),
            re.compile(
                r"do\s+not\s+write\s+any\s+(introduction|conclusion|summary)"
            ),
            f"phi4:mini sample-3 must include an explicit 'do not write "
            f"any introduction, conclusion, or summary' anti-drift "
            f"directive — these are the exact failure shapes the model "
            f"loops on. Sample text: {sample_3!r}",
        )
        self.assertRegex(
            sample_3.lower(),
            re.compile(r"do\s+not\s+repeat"),
            f"phi4:mini sample-3 must include a 'do not repeat any "
            f"bullet' directive. The 2026-06-01 failure showed the model "
            f"writing the same paragraph 4+ times verbatim. Sample text: "
            f"{sample_3!r}",
        )

    def test_phi4_mini_sample_3_does_not_use_known_failing_prompt(self):
        """Belt-and-suspenders: explicitly reject the literal prompt text
        that triggered the 2026-06-01 extended-bench failure. If somebody
        reverts the sample-3 rewrite by hand (or a merge brings back
        the old wording) this test catches it before the next bench run.
        """
        samples = self._phi4_mini_samples()
        sample_3 = samples[2]
        self.assertNotRegex(
            sample_3,
            re.compile(
                r"Give\s+a\s+three-bullet\s+recommendation\s+using\s+"
                r"evidence,\s*tradeoff,\s*and\s+next\s+experiment",
                re.IGNORECASE,
            ),
            f"phi4:mini sample-3 reverted to the known-failing "
            f"'three-bullet recommendation using evidence, tradeoff, "
            f"and next experiment' wording (the 2026-06-01 regression). "
            f"Use the curated rewrite with a hard word budget and "
            f"anti-drift directive instead. Sample text: {sample_3!r}",
        )


class RepoSyncContractTests(unittest.TestCase):
    """Pin the post-publish public-repo sync hook so a future edit to
    Publish.md / Publish_Mac.md or to the staging block cannot silently
    desync C:\\LocalAI from C:\\repos\\LocalAI.

    All tests guard external paths with Path(...).exists() so the suite
    still passes on a clean machine that has no Plans/ folder or repo
    checkout. They only enforce the contract on the maintainer's box
    where those files actually live.
    """

    SYNC_REPO_PATH = Path(r"C:\LocalAI\tools\sync_repo.ps1")
    PUBLISH_MD = Path(r"C:\Plans\Publish.md")
    PUBLISH_MAC_MD = Path(r"C:\Plans\Publish_Mac.md")
    REPO_MANIFEST = Path(r"C:\repos\LocalAI\tools\maintenance\manifest.txt")

    def test_sync_repo_wrapper_exists_and_has_expected_shape(self):
        path = self.SYNC_REPO_PATH
        if not path.exists():
            self.skipTest(f"{path} not present on this machine")
        text = path.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("C:\\repos\\LocalAI", text,
                      "sync_repo.ps1 must reference the canonical repo path")
        self.assertRegex(text, r"update_repo\.bat",
                         "sync_repo.ps1 must invoke update_repo.bat")
        self.assertRegex(text, r"diff_repo\.ps1",
                         "sync_repo.ps1 must invoke diff_repo.ps1 for the report-only path")
        self.assertRegex(text, r"\[switch\]\s*\$Apply",
                         "sync_repo.ps1 must expose -Apply as a switch parameter")
        # Repo-absent path must exit 0 (skip), not raise.
        self.assertRegex(text, r"Test-Path\s+\$Repo",
                         "sync_repo.ps1 must Test-Path the repo before doing anything")
        self.assertRegex(text, r"exit\s+0",
                         "sync_repo.ps1 must exit 0 on the skip path")

    def test_publish_md_describes_repo_sync_step(self):
        if not self.PUBLISH_MD.exists():
            self.skipTest(f"{self.PUBLISH_MD} not present on this machine")
        text = self.PUBLISH_MD.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("sync_repo.ps1", text,
                      "Publish.md must mention the sync_repo.ps1 hook")
        self.assertIn(r"C:\repos\LocalAI", text,
                      "Publish.md must reference the public-repo path")
        self.assertIn("manifest.txt", text,
                      "Publish.md must reference the manifest as the single source of truth")
        self.assertIn("-Apply", text,
                      "Publish.md must document the -Apply switch for actually copying")

    def test_publish_mac_md_describes_repo_sync_step(self):
        if not self.PUBLISH_MAC_MD.exists():
            self.skipTest(f"{self.PUBLISH_MAC_MD} not present on this machine")
        text = self.PUBLISH_MAC_MD.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("sync_repo.ps1", text,
                      "Publish_Mac.md must mention the sync_repo.ps1 hook")
        self.assertIn(r"C:\repos\LocalAI", text,
                      "Publish_Mac.md must reference the public-repo path")
        self.assertIn("manifest.txt", text,
                      "Publish_Mac.md must reference the manifest as the single source of truth")

    def test_publish_mac_staged_files_are_in_repo_manifest(self):
        """Every file the Mac publish stages (step 6 list) must be:
          (a) named in the public-repo manifest if it ships on Windows too, OR
          (b) explicitly Mac-only and intentionally absent from the public
              manifest (the public repo is Windows-only).
        Prevents desync between what publishes externally as a release zip
        vs. what's tracked in the public source repo.
        """
        if not self.PUBLISH_MAC_MD.exists():
            self.skipTest(f"{self.PUBLISH_MAC_MD} not present on this machine")
        if not self.REPO_MANIFEST.exists():
            self.skipTest(f"{self.REPO_MANIFEST} not present on this machine")
        publish_text = self.PUBLISH_MAC_MD.read_text(encoding="utf-8", errors="ignore")
        manifest_text = self.REPO_MANIFEST.read_text(encoding="utf-8", errors="ignore")

        # Cross-platform files — must appear in both Publish_Mac.md step 6
        # AND in the public-repo manifest.
        staged_shared_files = [
            "main.py", "run_batch.py", "verify_models.py", "requirements.txt",
            "models_catalog.json", "content_blocklist.txt",
        ]
        for fname in staged_shared_files:
            self.assertIn(fname, publish_text,
                          f"sanity: {fname} must still be in Publish_Mac.md step 6 staging block")
            self.assertIn(fname, manifest_text,
                          f"{fname} is staged by Publish_Mac.md but missing from the public-repo manifest "
                          f"(add it to manifest.txt, or add it to the NOT-shipped appendix with rationale)")

        # Mac-only files — must appear in Publish_Mac.md step 6 AND must
        # NOT appear in the public-repo manifest's tracked entries. The
        # public repo is intentionally Windows-only; *.sh launchers ship
        # via the Mac flow only.
        staged_mac_only_files = [
            "run.sh", "run_batch.sh", "setup.sh", "uninstall.sh",
        ]
        for fname in staged_mac_only_files:
            self.assertIn(fname, publish_text,
                          f"sanity: {fname} must still be in Publish_Mac.md step 6 staging block")
            # Allow the name to appear inside manifest *comments* (e.g.,
            # the NOT-shipped appendix) but not as a tracked path. The
            # manifest parser treats lines starting with '#' as comments,
            # so we check uncommented lines only.
            tracked_lines = [
                ln.split('#', 1)[0].strip()
                for ln in manifest_text.splitlines()
                if ln.strip() and not ln.strip().startswith('#')
            ]
            self.assertNotIn(fname, tracked_lines,
                             f"{fname} is Mac-only and must NOT appear as a tracked entry in the "
                             f"public-repo manifest (the public repo is intentionally Windows-only).")

        # Directory entries (src/, docs/) must also be covered.
        for dirent in ["src/", "docs/"]:
            self.assertIn(dirent, manifest_text,
                          f"{dirent} tree is staged by Publish_Mac.md but missing from the public-repo manifest")


class DiagnosticLogContractTests(unittest.TestCase):
    """v2026.06.01.4/5: localai.log diagnostic contracts.

    A high-VRAM-workstation first-run failure on 2026-06-01 surfaced two papercuts:

    1. When the app crashed at module load (broken onnxruntime), the
       error message told the user "Check localai.log" — but no such
       file existed because App.__init__ never ran.
    2. There was no record of what the setup script had actually done.

    v2026.06.01.4 attempted to fix BOTH by adding a PowerShell Tee-Object
    self-wrapper to setup.bat — that was REVERTED in v2026.06.01.5 because
    piping cmd's stdout through Tee-Object switches Windows to block
    buffering, so interactive `set /p` prompts never flush their prompt
    text before reading user input. Setup transcript capture is deferred
    to a future design that does not break interactivity.

    The localai.log contract still holds:

    * ``run.bat`` writes session start/end markers to ``localai.log`` so
      the file always exists. We deliberately do NOT tee ``python
      main.py`` itself, because PowerShell ``Tee-Object`` opens the file
      with ``FileShare.Read`` which would block the app's own logger from
      writing concurrently to the same file.
    * The crash-path error message references ``localai.log`` honestly,
      including the case where a launch-time module-import crash leaves
      no app-level events in the file.
    """

    def test_setup_bats_guard_against_zip_preview_and_temp_extractor_paths(self):
        """v2026.06.02.0: A user reported a confusing
        ``[Errno 2] No such file or directory: ...onnxruntime_genai_cuda...whl``
        mid-install error. Root cause: they double-clicked setup.bat from
        inside Windows Explorer's zip preview, which extracts to a path
        like ``%TEMP%\\<GUID>_<zipname>.zip.e2b\\`` — already ~135 chars,
        and setup1.bat's pip-cache redirection nests the wheel-unpack
        directory further until Windows MAX_PATH=260 is exceeded.

        Both setup.bat (the shim) and setup1.bat (the actual installer,
        defense-in-depth) must guard against running from such a path
        and fail fast with a friendly fix-it message ("Extract All... to a
        SHORT path like C:\\LocalAI") rather than letting pip blow up
        halfway through the install."""
        for rel in ("setup.bat", "setup1.bat"):
            with self.subTest(file=rel):
                text = read(rel)
                self.assertIn(
                    ".zip.e2b\\",
                    text,
                    f"{rel} must check for the Windows zip-preview marker "
                    "`.zip.e2b\\` in the script directory path so users "
                    "who double-click setup.bat from inside a zip preview "
                    "get a friendly error before pip blows past MAX_PATH.",
                )
                self.assertIn(
                    ".zip\\",
                    text,
                    f"{rel} must also check for the broader `.zip\\` "
                    "directory marker so 7-Zip / WinRAR temp extractors "
                    "(which produce paths like %TEMP%\\<name>.zip\\) are "
                    "caught too.",
                )
                self.assertIn(
                    "MAX_PATH",
                    text,
                    f"{rel} guard message must mention MAX_PATH so users "
                    "understand WHY installing from a deep temp path fails.",
                )
                self.assertIn(
                    "Extract All",
                    text,
                    f"{rel} guard message must tell users to use Windows "
                    "Explorer's `Extract All...` rather than just opening "
                    "the zip preview.",
                )

    def test_setup_bat_does_not_use_powershell_tee_wrapper(self):
        """v2026.06.01.5 revert: the self-tee wrapper added in v.4 broke
        interactive prompts. Make sure nobody adds it back without a
        non-interactive-breaking design.

        v2026.06.01.11 split: setup.bat is now a tiny shim that launches
        setup.ps1 (which uses Start-Transcript, not Tee-Object), and the
        actual install logic lives in setup1.bat. Both must remain clean
        of the LOCALAI_SETUP_TEED / Tee-Object pattern."""
        for rel in ("setup.bat", "setup1.bat"):
            with self.subTest(file=rel):
                text = read(rel)
                self.assertNotIn(
                    "LOCALAI_SETUP_TEED",
                    text,
                    f"{rel} must NOT re-introduce the LOCALAI_SETUP_TEED "
                    "self-tee wrapper - it pipes cmd's stdout through "
                    "PowerShell which switches Windows to block buffering "
                    "and breaks every `set /p` prompt. If you want setup "
                    "transcript capture, design a mechanism that preserves "
                    "interactive console I/O (e.g. on-failure-only copy of "
                    "scrollback, or a non-piping wrapper).",
                )
                self.assertNotIn(
                    "Tee-Object",
                    text,
                    f"{rel} must NOT use PowerShell Tee-Object - same "
                    "reason as above. The output piping breaks `set /p` "
                    "prompts.",
                )

    def test_run_bat_writes_session_markers_to_localai_log(self):
        run = read("run.bat")
        self.assertIn(
            "session start",
            run.lower(),
            "run.bat must write a 'session start' marker to localai.log so "
            "the file always exists when the error message references it.",
        )
        self.assertIn(
            "session end",
            run.lower(),
            "run.bat must write a 'session end' marker so log scanners can "
            "tell sessions apart and see the python exit code.",
        )
        self.assertIn(
            "localai.log",
            run,
            "run.bat must reference localai.log as the session-marker destination.",
        )
        # Concurrency guard: run.bat MUST NOT tee the python invocation
        # through Tee-Object — that opens the file with FileShare.Read and
        # would block the app's own logger from writing.
        self.assertNotIn(
            "Tee-Object",
            run,
            "run.bat must NOT use Tee-Object on python main.py — Tee-Object "
            "opens localai.log with FileShare.Read which blocks the app "
            "logger's concurrent writes. Use session markers instead.",
        )

    def test_run_bat_error_message_references_both_logs(self):
        run = read("run.bat")
        # The crash-path block (errorlevel != 0) must mention BOTH log files
        # so users know where to look depending on which phase failed.
        self.assertIn(
            "localai.log",
            run,
            "run.bat crash message must point users at localai.log.",
        )
        self.assertIn(
            "setup.log",
            run,
            "run.bat crash message must point users at setup.log so they "
            "can see what setup put on the machine when the failure is a "
            "broken Python package rather than an app-level bug.",
        )

    def test_gitignore_protects_localai_log(self):
        gitignore = read(".gitignore")
        self.assertIn(
            "localai.log",
            gitignore,
            ".gitignore must exclude localai.log — runtime-only artifact.",
        )
        # setup.log stays gitignored even though the v.4 tee mechanism was
        # reverted in v.5 — future on-failure capture may still want to write
        # there and we never want it in source control.
        self.assertIn(
            "setup.log",
            gitignore,
            ".gitignore must exclude setup.log — reserved name for future "
            "setup transcript capture; never belongs in source control.",
        )


class OnnxRuntimeVariantContractTests(unittest.TestCase):
    """v2026.06.01.7 (Ron, 2026-06-01): setup.bat must install exactly ONE
    onnxruntime runtime variant (onnxruntime / onnxruntime-gpu /
    onnxruntime-directml) and exactly ONE genai variant chosen by detected
    GPU vendor. The three runtime packages share the onnxruntime/ namespace
    and are mutually exclusive — installing more than one leaves
    onnxruntime.__file__ == None and InferenceSession gone (workstation-class
    Py3.13 regression that broke Toolbox Speak in v.6).
    """

    def test_setup_defines_onnx_variant_vars_from_gpu_vendor(self):
        setup = read("setup1.bat")
        self.assertIn(":detect_setup_hardware", setup)
        block_start = setup.index(":detect_setup_hardware")
        block_end = setup.index("goto :eof", block_start)
        block = setup[block_start:block_end]
        # The variant selector must live inside :detect_setup_hardware so
        # SETUP_ONNX_PKG is populated before any pip install runs.
        self.assertIn("SETUP_HAS_NVIDIA", block)
        self.assertIn("SETUP_HAS_DML_GPU", block)
        self.assertIn('set "SETUP_ONNX_PKG=onnxruntime-gpu"', block)
        self.assertIn('set "SETUP_ONNX_GENAI_PKG=onnxruntime-genai-cuda"', block)
        self.assertIn('set "SETUP_ONNX_EP=CUDAExecutionProvider"', block)
        self.assertIn('set "SETUP_ONNX_PKG=onnxruntime-directml"', block)
        self.assertIn('set "SETUP_ONNX_GENAI_PKG=onnxruntime-genai-directml"', block)
        self.assertIn('set "SETUP_ONNX_EP=DmlExecutionProvider"', block)
        self.assertIn('set "SETUP_ONNX_PKG=onnxruntime"', block)
        self.assertIn('set "SETUP_ONNX_GENAI_PKG=onnxruntime-genai"', block)
        self.assertIn('set "SETUP_ONNX_EP=CPUExecutionProvider"', block)

    def test_setup_onnx_install_purges_all_variants_before_install(self):
        """All six mutually-exclusive packages (3 runtimes + 3 genai)
        must be uninstalled BEFORE installing the chosen variant so a
        half-installed state from any prior run cannot shadow the chosen
        wheel and break the onnxruntime namespace."""
        setup = read("setup1.bat")
        # The ONNX install block runs between [*] Installing ONNX runtime
        # packages and the OpenVINO section.
        onnx_start = setup.index("[*] Installing ONNX runtime packages")
        onnx_end = setup.index(":done_onnx", onnx_start)
        onnx_block = setup[onnx_start:onnx_end]
        for pkg in (
            "onnxruntime", "onnxruntime-gpu", "onnxruntime-directml",
            "onnxruntime-genai", "onnxruntime-genai-cuda", "onnxruntime-genai-directml",
        ):
            self.assertIn(
                pkg, onnx_block,
                f"setup.bat ONNX install block must reference {pkg} in the "
                "pre-install purge so prior-run half-installed state cannot "
                "shadow the chosen wheel (v2026.06.01.7).",
            )
        # The install line must use the variables, not hard-coded -directml.
        self.assertIn("!SETUP_ONNX_PKG!", onnx_block)
        self.assertIn("!SETUP_ONNX_GENAI_PKG!", onnx_block)
        # The smoke gate must check both InferenceSession AND
        # ort.__file__ is not None (the broken-namespace signature).
        self.assertIn("from onnxruntime import InferenceSession", onnx_block)
        self.assertIn("ort.__file__ is not None", onnx_block)
        self.assertIn("!SETUP_ONNX_EP!", onnx_block)

    def test_setup_utility_reconcile_uses_chosen_variant_not_hardcoded_directml(self):
        """The post-utility reconcile step used to unconditionally reinstall
        onnxruntime-directml regardless of GPU vendor. On NVIDIA boxes this
        installed BOTH -gpu (left by an earlier dep) AND -directml, leaving
        onnxruntime.__file__ == None. v2026.06.01.7 must use !SETUP_ONNX_PKG!
        so the chosen variant wins.
        """
        setup = read("setup1.bat")
        # The reconcile block runs after :done_utility's predecessor.
        reconcile_start = setup.index("Reconciling")
        reconcile_end = setup.index(":done_utility", reconcile_start)
        reconcile_block = setup[reconcile_start:reconcile_end]
        # Must purge ALL six packages first.
        for pkg in (
            "onnxruntime", "onnxruntime-gpu", "onnxruntime-directml",
            "onnxruntime-genai", "onnxruntime-genai-cuda", "onnxruntime-genai-directml",
        ):
            self.assertIn(pkg, reconcile_block)
        # Must install the chosen variant via variable expansion.
        self.assertIn("!SETUP_ONNX_PKG!", reconcile_block)
        self.assertIn("!SETUP_ONNX_GENAI_PKG!", reconcile_block)
        # The pre-v.7 bug: hard-coded "onnxruntime-directml onnxruntime-genai-directml"
        # on the install line. Make sure that exact pair no longer appears as
        # an install argument (it may still appear in the purge list).
        self.assertNotIn(
            "pip install --upgrade --no-warn-conflicts onnxruntime-directml onnxruntime-genai-directml",
            reconcile_block,
            "setup.bat reconcile step must use !SETUP_ONNX_PKG! "
            "!SETUP_ONNX_GENAI_PKG!, not hard-code the DirectML pair. "
            "Hard-coding DirectML on NVIDIA boxes is what caused the v.6 "
            "Toolbox Speak regression (mutually-exclusive onnxruntime "
            "variants installed side by side, namespace broken).",
        )
        # cwd guard: the v.6 setup log printed "The system cannot find the
        # drive specified." twice from this block. Pin cwd to script dir.
        self.assertIn('cd /d "%~dp0"', reconcile_block)
        # Smoke gate must check the broken-namespace signature.
        self.assertIn("from onnxruntime import InferenceSession", reconcile_block)
        self.assertIn("ort.__file__ is not None", reconcile_block)
        self.assertIn("!SETUP_ONNX_EP!", reconcile_block)

    def test_setup_does_not_install_two_onnxruntime_runtimes_on_same_line(self):
        """Defense in depth: no single pip install line in setup.bat may
        name two of the mutually-exclusive runtime packages together, ever.
        """
        setup = read("setup1.bat")
        runtime_pkgs = ("onnxruntime-gpu", "onnxruntime-directml")
        for line in setup.splitlines():
            # Skip purge lines (uninstall is FINE to list all of them).
            if "pip uninstall" in line:
                continue
            # Skip comment lines.
            if line.strip().startswith("::"):
                continue
            # Skip echo lines (error guidance shown to user, not a pip cmd).
            if line.strip().lower().startswith("echo"):
                continue
            present = [p for p in runtime_pkgs if p in line]
            if len(present) > 1:
                self.fail(
                    f"setup.bat installs mutually-exclusive onnxruntime "
                    f"runtimes on the same line: {present!r}. Pick exactly "
                    f"one via !SETUP_ONNX_PKG!.\n  Offending line: {line.strip()}"
                )


if __name__ == "__main__":
    unittest.main()

