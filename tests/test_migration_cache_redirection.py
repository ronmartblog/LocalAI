# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""v5.3.10 cache-redirection contract tests.

Asserts that every shipped Windows .bat helper that runs pip / python
prepends an explicit cache-redirection block pointing PIP / HF / torch /
TMP / Ollama caches at ``%~dp0.cache\\...`` so installs land on the
install drive (not on a profile-capped %USERPROFILE%).

The macOS / Linux companions (setup.sh / run.sh) get a smaller `export`
block tested at the bottom of the file.

Failures here mean a future agent removed the redirection block and reopened
the door to today's profile-drive ENOSPC disaster.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

APP_ROOT = Path(__file__).parent.parent


REQUIRED_BAT_ENV_LINES = (
    'set "PIP_CACHE_DIR=%~dp0.cache\\pip"',
    'set "TMP=%~dp0.cache\\tmp"',
    'set "TEMP=%~dp0.cache\\tmp"',
    'set "HF_HOME=%~dp0.cache\\huggingface"',
    'set "HUGGINGFACE_HUB_CACHE=%~dp0.cache\\huggingface\\hub"',
    'set "TORCH_HOME=%~dp0.cache\\torch"',
    'set "XDG_CACHE_HOME=%~dp0.cache"',
    'set "OLLAMA_MODELS=%~dp0Ollama"',
)


REQUIRED_BAT_FILES = (
    "setup1.bat",
    "run.bat",
    "fix_nvidia_pytorch.bat",
    "fix_directml_pytorch.bat",
    "verify_models.bat",
    "set_ollama_models_dir.bat",
)


REQUIRED_SH_EXPORTS = (
    'export PIP_CACHE_DIR="$APP_DIR/.cache/pip"',
    'export TMPDIR="$APP_DIR/.cache/tmp"',
    'export HF_HOME="$APP_DIR/.cache/huggingface"',
    'export HUGGINGFACE_HUB_CACHE="$APP_DIR/.cache/huggingface/hub"',
    'export TORCH_HOME="$APP_DIR/.cache/torch"',
    'export XDG_CACHE_HOME="$APP_DIR/.cache"',
    'export OLLAMA_MODELS="$APP_DIR/Ollama"',
)


class CacheRedirectionContractTests(unittest.TestCase):

    # ── per-bat-file checks ──────────────────────────────────────────────

    def _read(self, name: str) -> str:
        return (APP_ROOT / name).read_text(encoding="utf-8", errors="replace")

    def _assert_env_block_present_in_order(self, batch_name: str):
        text = self._read(batch_name)
        positions = []
        for line in REQUIRED_BAT_ENV_LINES:
            idx = text.find(line)
            self.assertNotEqual(
                idx, -1,
                f"{batch_name}: missing required env line {line!r}",
            )
            positions.append(idx)
        # In-order check: each subsequent env line must come at-or-after the
        # previous one.  We tolerate gaps for the `mkdir` guard lines.
        for i in range(1, len(positions)):
            self.assertGreaterEqual(
                positions[i], positions[i - 1],
                f"{batch_name}: env lines must appear in the documented order",
            )

    def test_setup_bat_has_cache_block(self):
        self._assert_env_block_present_in_order("setup1.bat")

    def test_run_bat_has_cache_block(self):
        self._assert_env_block_present_in_order("run.bat")

    def test_fix_nvidia_pytorch_bat_has_cache_block(self):
        self._assert_env_block_present_in_order("fix_nvidia_pytorch.bat")

    def test_fix_directml_pytorch_bat_has_cache_block(self):
        self._assert_env_block_present_in_order("fix_directml_pytorch.bat")

    def test_verify_models_bat_has_cache_block(self):
        self._assert_env_block_present_in_order("verify_models.bat")

    def test_set_ollama_models_dir_bat_has_cache_block(self):
        self._assert_env_block_present_in_order("set_ollama_models_dir.bat")

    # ── ordering: block must precede the first pip / python invocation ──

    def test_setup_bat_env_block_precedes_first_pip_install(self):
        text = self._read("setup1.bat")
        first_env = text.find('set "PIP_CACHE_DIR=%~dp0.cache')
        first_pip = text.find("pip install")
        self.assertGreater(first_env, 0)
        self.assertGreater(first_pip, 0)
        self.assertLess(
            first_env, first_pip,
            "setup1.bat: PIP_CACHE_DIR redirect must appear before the first "
            "pip install call",
        )

    def test_run_bat_env_block_precedes_first_python(self):
        text = self._read("run.bat")
        first_env = text.find('set "PIP_CACHE_DIR=%~dp0.cache')
        # run.bat does not invoke pip but does invoke python; the env block
        # must come before the first python *invocation* so caches inherit.
        # We look for an actual call (`%PYTHON_EXE% ...`, `python.exe ...`,
        # `start "" python` etc.) — not the comment line that mentions python.
        first_python = None
        for m_ in re.finditer(r"(?im)^\s*(?:call\s+|start\s+\"\"\s+|@\s*)?[^:].*python(?:\.exe)?\b", text):
            line = text[m_.start():m_.end()]
            if line.lstrip().startswith(":") or line.lstrip().startswith("rem "):
                continue
            first_python = m_
            break
        self.assertGreater(first_env, 0)
        self.assertIsNotNone(first_python, "run.bat: no python invocation found")
        self.assertLess(first_env, first_python.start(),
                        "run.bat: env block must precede first python call")

    def test_fix_nvidia_pytorch_bat_env_block_precedes_first_pip_install(self):
        text = self._read("fix_nvidia_pytorch.bat")
        first_env = text.find('set "PIP_CACHE_DIR=%~dp0.cache')
        first_pip = text.find("pip install")
        self.assertGreater(first_env, 0)
        self.assertGreater(first_pip, 0)
        self.assertLess(first_env, first_pip)

    def test_fix_directml_pytorch_bat_env_block_precedes_first_pip_install(self):
        text = self._read("fix_directml_pytorch.bat")
        first_env = text.find('set "PIP_CACHE_DIR=%~dp0.cache')
        first_pip = text.find("pip install")
        self.assertGreater(first_env, 0)
        self.assertGreater(first_pip, 0)
        self.assertLess(first_env, first_pip)

    # ── mkdir guards: each cache dir is created before use ──────────────

    def test_setup_bat_creates_cache_dirs_before_use(self):
        text = self._read("setup1.bat")
        for sub in ("pip", "tmp", "huggingface", "torch"):
            pat = re.compile(rf"mkdir\s+\"%~dp0\.cache\\{sub}", re.IGNORECASE)
            self.assertIsNotNone(
                pat.search(text),
                f"setup1.bat: missing mkdir guard for .cache\\{sub}",
            )

    # ── shell scripts (macOS / Linux) ───────────────────────────────────
    # NOTE: setup.sh / run.sh are excluded from the public Windows-only
    # repo (see manifest.txt + scrub_grep.ps1). These tests skip on
    # checkouts where the file is absent, so the same suite stays green
    # both maintainer-side (files present) and repo-side (files absent).

    def test_setup_sh_exports_cache_redirection(self):
        path = APP_ROOT / "setup.sh"
        if not path.exists():
            self.skipTest("setup.sh is maintainer-only; not shipped with the public repo")
        text = path.read_text(encoding="utf-8")
        for line in REQUIRED_SH_EXPORTS:
            self.assertIn(line, text,
                          f"setup.sh: missing required export {line!r}")

    def test_run_sh_exports_cache_redirection(self):
        path = APP_ROOT / "run.sh"
        if not path.exists():
            self.skipTest("run.sh is maintainer-only; not shipped with the public repo")
        text = path.read_text(encoding="utf-8")
        for line in REQUIRED_SH_EXPORTS:
            self.assertIn(line, text,
                          f"run.sh: missing required export {line!r}")

    def test_setup_sh_makes_cache_dirs_with_mkdir_p(self):
        path = APP_ROOT / "setup.sh"
        if not path.exists():
            self.skipTest("setup.sh is maintainer-only; not shipped with the public repo")
        text = path.read_text(encoding="utf-8")
        # We look for a single mkdir -p line that touches the cache dirs.
        self.assertRegex(
            text,
            r"mkdir\s+-p\s+\"\$PIP_CACHE_DIR\".*\$HUGGINGFACE_HUB_CACHE.*\$TORCH_HOME",
        )


if __name__ == "__main__":
    unittest.main()
