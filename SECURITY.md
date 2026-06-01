# Security Policy

## Reminder: this is a hobby project

LocalAI Studio is a personal hobby project provided **AS-IS** with **no
warranty** of any kind. See [DISCLAIMER.md](DISCLAIMER.md). The maintainer
is one person responding in his spare time. Please calibrate your
expectations accordingly.

## What counts as a security issue

This project runs entirely on your local machine and orchestrates
third-party AI runtimes (Ollama, ComfyUI, ONNX Runtime, OpenVINO GenAI).
A "security issue" in the scope of this project means something like:

- A code path that could lead to arbitrary code execution from a
  malicious model definition in the catalog.
- A path-traversal bug in setup/download code that lets a remote source
  write outside the project's working directories.
- Credentials, tokens, or personal data being written to disk or logs
  in cleartext when they should not be.
- A bug in the catalog loader that bypasses URL/path validation when
  fetching models.

Issues in the **third-party runtimes themselves** (Ollama, ComfyUI,
ONNX Runtime, OpenVINO GenAI, PyTorch, the model weights, NVIDIA
drivers, etc.) are out of scope and should be reported directly to
those upstream projects.

## How to report

For low-severity or already-public issues, open a
[GitHub issue](../../issues) with the `security` label.

For higher-severity issues that have not yet been disclosed, please use
[GitHub Security Advisories](../../security/advisories/new) to file a
private report. Do **not** post sensitive details in public issues.

Please do **not** send the maintainer security reports via email,
LinkedIn, or other private channels outside GitHub. Reports filed
outside the normal channels may be missed.

## Response expectations

There are no SLAs on response time. The maintainer will try to:

1. Acknowledge a private report within a couple of weeks.
2. Decide whether the report is in scope and reproducible.
3. If in scope, work on a fix or mitigation as time allows.
4. Coordinate a disclosure timeline with the reporter.

If you need guaranteed response times, this hobby project is not the
right place to deploy from. Use commercially supported alternatives.