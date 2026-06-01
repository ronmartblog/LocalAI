# LocalAI Studio — Architecture

This document is a high-level orientation to the LocalAI Studio codebase
for anyone reading the source. For setup and usage, see the top-level
[README.md](../README.md).

> **Reminder:** This is a personal hobby project. See
> [DISCLAIMER.md](../DISCLAIMER.md) for the full legal context. Nothing
> here is an official statement about any vendor's products, support, or
> roadmap.

## 1. What this is

LocalAI Studio is a **single-process desktop GUI** that orchestrates
several local AI runtimes side-by-side on one machine, without sending
data to a cloud service. It is intentionally a monolithic desktop app
(no server, no multi-user, no daemon).

The UI is built on [CustomTkinter](https://customtkinter.tomschimansky.com/).
All long-running work happens on background threads; UI updates marshal
back to the main thread via `self.after(0, ...)`.

## 2. The four backends

| Backend | Used for | Where it runs | Protocol |
|---|---|---|---|
| [Ollama](https://ollama.com/) | Text and vision LLMs | External `ollama serve` daemon, auto-started by the app | REST + streaming JSON |
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | Image generation (SD / SDXL / FLUX) | Bundled `ComfyUI/` subfolder, auto-started by the app | REST + WebSocket (queue / progress) |
| [ONNX Runtime](https://onnxruntime.ai/) | In-process inference on Windows DirectML | In-process via `onnxruntime` Python package | Direct Python API |
| [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai) | In-process inference on Intel NPU / GPU / CPU (Windows) | In-process via `openvino_genai` Python package | Direct Python API |

Each backend has its own subsystem in the codebase:

- `src/ollama_*.py` — Ollama client, model management, error helpers.
- `src/comfyui_*.py` — ComfyUI workflow construction, REST + WS client.
- `src/onnx_*.py` — ONNX session lifecycle, DirectML provider selection.
- `src/openvino_*.py` — OpenVINO GenAI pipeline orchestration.

The app aggressively gates backend availability on platform detection
(`src/system_info.py`) and missing-dependency probes — a missing backend
is reported in the UI rather than crashing the app.

## 3. File map

```
localai/
├── main.py                  # Entry point — instantiates and runs App.
├── setup.bat                # First-run hardware detection + install.
├── run.bat                  # Launcher for daily use.
├── requirements.txt         # Pinned runtime dependencies.
│
├── src/
│   ├── app.py               # CustomTkinter UI + page controllers.
│   ├── system_info.py       # Platform / CPU / GPU / RAM detection.
│   ├── gpu_detect.py        # NVIDIA / AMD / Intel GPU probing,
│   │                        #   vGPU and MIG-capable profile detection,
│   │                        #   ComfyUI launch-flag selection.
│   ├── catalog.py           # BUILTIN_MODELS catalog (curated default set).
│   ├── models_catalog.json  # Same catalog as data (loader source).
│   ├── hf_compat.py         # Hugging Face slug resolution / shape checks.
│   ├── model_guide.py       # Markdown / HTML guide generation.
│   ├── batch_runner.py      # Benchmark + bulk evaluation orchestration.
│   ├── constrained_env.py   # Detection of constrained-VM environments
│   │                        #   (low VRAM, ephemeral profile, etc.) and
│   │                        #   the resulting policy hints.
│   └── (other helpers)
│
├── tests/
│   ├── test_skus.py
│   ├── test_persistence_and_catalog.py
│   ├── test_app_static_contracts.py
│   ├── test_gpu_detect.py
│   ├── test_batch_runner_*.py
│   └── (others — pytest discovers everything)
│
├── docs/
│   ├── index.html           # Home page / feature tour.
│   ├── image-gen-guide.html # Image-generation tab usage guide.
│   ├── Model-Guide.html     # Per-model details, generated from catalog.
│   └── architecture.md      # ← this file.
```

The maintainer's local working copy is **not** this repository. Live
edits happen in a separate working directory that contains additional
machine-local data (logs, caches, the maintainer's private `skus.json`,
internal AI-agent guidance, third-party runtime checkouts). A small
maintainer-only publish script set produces this repository's contents
as a clean subset; those scripts are not part of the public package.

## 4. The catalog

The model catalog is the single source of truth for what the app
knows how to download and run. Each entry describes:

- A stable `id` (the key everything else references).
- The runtime (`ollama`, `comfyui`, `onnx`, `openvino`).
- The Hugging Face slug, Ollama tag, or local download URL.
- VRAM / RAM requirements.
- Whether it is text, vision, embedding, image-gen, etc.
- Optional `tags` for filter chips.

Two representations are kept in lockstep:

| File | Used by |
|---|---|
| `src/catalog.py` (`BUILTIN_MODELS`) | Python imports, type-checked references in app code. |
| `src/models_catalog.json` | Loader-time data file, also the source for `docs/Model-Guide.html`. |

A test asserts that both representations are equivalent at startup —
keep them in sync.

The catalog **does not** know about hardware SKUs. Per-SKU model
recommendations live in `skus.json` (see next section), keeping the
catalog reusable across users with different hardware.

## 5. SKU profiles (optional)

The app supports an **optional** `skus.json` file at the repository root
that describes one or more hardware profiles. When present, it drives:

- A SKU filter chip strip on the Models tab.
- Per-SKU recommendation badges on model cards.
- Per-SKU presets in the Benchmark tab (quick vs extended).
- A SKU column in the Model Guide.

When absent, all SKU-aware UI surfaces hide and the app falls back to a
single synthetic "This Device" entry derived from local hardware
probing. The repo intentionally ships **zero** hardcoded SKU display
names — there is no built-in vendor or GPU-class default profile.
Users who want a richer benchmark catalog (multiple comparison SKUs,
per-SKU recommended models, vm_size auto-detection) drop in their own
`skus.json`.

**`skus.json` is `.gitignore`'d and never shipped with the repo.** Every
user maintains their own. The expected schema is documented inline at
the top of `src/system_info.py` as part of the loader, and an example is
in `tests/fixtures/`.

## 6. UI layering

```
+-------------------------------------------------------------+
|                  CustomTkinter main window                  |
+--------------+----------------------------------------------+
|  Nav rail    |                Page area                     |
| (left side)  |  (only one page mounted at a time —          |
|              |   lazily constructed on first visit)         |
|              |                                              |
|  Home        |  +----------------------------------------+  |
|  Models      |  | Page-specific controls                 |  |
|  Chat        |  | (cards, tables, form fields, etc.)     |  |
|  Image Gen   |  +----------------------------------------+  |
|  Benchmark   |  +----------------------------------------+  |
|  Toolbox     |  | Status / log strip                     |  |
|  Settings    |  +----------------------------------------+  |
|  About       |                                              |
+--------------+----------------------------------------------+
```

Pages are constructed lazily on first visit (`_switch_page(name)`).
That keeps cold-start cheap and lets the smoke tests verify each page
independently:

```bash
python -c "from src.app import App; app=App(); app.update(); app._switch_page('benchmark'); app.update(); print('BENCHMARK_PAGE_SMOKE_OK'); app.destroy()"
```

## 7. Backgrounding rules

- Anything that touches disk, network, or a subprocess runs on a
  background thread. Never block the UI thread for more than ~16 ms.
- All UI updates from background threads marshal back via
  `self.after(0, lambda: ...)`.
- Cancel-able operations use a `threading.Event` stop flag passed into
  the worker.

## 8. Build and release

```bash
git clone https://github.com/<owner>/localai.git
cd localai
setup.bat       # installs Python deps + ComfyUI
run.bat
python -m pytest tests/   # run the test suite
```

The publish workflow that produces this repository from the maintainer's
working copy lives outside the public package. Contributors interact
with the repo via normal `git` commands; no special tooling is required.

## 9. Testing

`pytest tests/` runs the full suite (a few hundred tests). The suite
covers loader contracts (catalog, SKU profiles), UI-static contracts
(model cards, filter chips), GPU detection regex tables, the benchmark
runner, and the constrained-environment policy hints.

Tests do not require network access or any GPU hardware. They run on a
plain CI runner with Python 3.12+.

## 10. Things that have intentionally been kept out of scope

- A server mode, daemon, or multi-user backend.
- Telemetry or analytics of any kind.
- Bundling model weights with the repo (they always download on demand).
- Replacing CustomTkinter with another UI framework.
- Restructuring the catalog to be per-user / dynamically authored.
- Anything that creates a contractual or support implication with any
  vendor. See [DISCLAIMER.md](../DISCLAIMER.md).