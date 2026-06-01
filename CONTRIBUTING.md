# Contributing to LocalAI Studio

Thank you for your interest in this project. A few things to know up
front.

## This is a personal hobby project

**LocalAI Studio is maintained as a personal hobby in the author's free
time.** It is not a commercial product, has no engineering team behind
it, and the maintainer's response time on issues and pull requests will
vary widely — sometimes within a day, sometimes within months, sometimes
not at all. Please be patient.

If you need a reliable, supported local-AI orchestration tool for
production or commercial use, this is not it. See
[DISCLAIMER.md](DISCLAIMER.md) for full context.

## How to report a bug

Open a [GitHub issue](../../issues) with:

1. **What you tried** (exact command or click path).
2. **What you expected** to happen.
3. **What actually happened**, including any error text from the
   on-screen log panel and from `localai.log` in the repository root.
4. **Your environment**: OS, CPU, GPU (model and VRAM), Python version,
   and which backend you were exercising (Ollama, ComfyUI, ONNX Runtime,
   or OpenVINO GenAI).

Please redact any personal information from logs and screenshots before
pasting them.

## How to request a feature

Open an issue describing the use case **first**. Don't open a pull
request for a new feature before there is general agreement that it
fits the project's scope. Drive-by feature PRs are unlikely to be
merged.

The project's scope is intentionally narrow: a single-process desktop
GUI that orchestrates a handful of local AI runtimes on one machine.
Network services, multi-user features, cloud integrations, and anything
that requires running a server are out of scope.

## How to submit a pull request

1. Fork the repository.
2. Create a branch with a short, descriptive name.
3. Make focused, surgical changes. One concern per PR.
4. Keep the existing code style. Don't reformat the world.
5. Run the test suite locally:
   ```bash
   python -m pytest tests/
   ```
   It must pass before you push.
6. If your change is visible in the UI, smoke-test the app end-to-end:
   ```bash
   python -c "from src.app import App; app=App(); app.update(); print('STARTUP_SMOKE_OK'); app.destroy()"
   ```
7. Open the PR. Describe what changed, why, and how you tested it.

## Code style

- Python 3.12+. No exotic dependencies — see `requirements.txt`.
- No reformatting passes unrelated to your change.
- Comments explain *why*, not *what*. Don't add boilerplate docstrings.
- Keep `src/app.py` cohesive; this is intentionally a monolithic UI
  layer. Pull pure-logic helpers out into small focused modules
  (`src/system_info.py`, `src/catalog.py`, etc.) when the size warrants
  it.
- All UI threading marshaled back to the main thread via
  `self.after(0, ...)`.

## Things that won't get merged

- Renames or reformatting of the catalog or the SKU loader for cosmetic
  reasons.
- Replacing the existing UI framework (CustomTkinter) with another one.
- Adding telemetry, analytics, or "phone home" behavior of any kind.
- Anything that requires shipping model weights with the repo (they live
  in user storage, downloaded on demand).
- Anything that breaks the "single-process desktop app" assumption.
- Documentation that walks back the disclaimers in
  [DISCLAIMER.md](DISCLAIMER.md).

## Code of Conduct

By participating in this project you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License of contributions

By submitting a contribution you agree to license it under the
[Apache License, Version 2.0](LICENSE), the same license that covers the
rest of the project.
