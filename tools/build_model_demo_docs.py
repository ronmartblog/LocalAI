# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""Regenerate the consolidated Model Guide (docs/Model-Guide.html) from the active catalog.

Replaces the previous ``ModelDemoPrompts.html`` / ``ImageGenPrompts.html`` /
``ChatPromptIdeas.html`` / ``model-value-props.html`` generators — all four were
consolidated into a single Model Guide in v5.3.4 (see
``Archive/doc-consolidation-2026-05/``). The data source is unchanged:
``models_catalog.json`` enriched by ``get_model_demo()``, which merges
``src/sample_prompts.MODEL_DEMO_SAMPLE_OVERRIDES`` on top of the fallback
prompts in ``src/model_demos.py``.

The legacy redirect shims were removed in post-v5.3.4 docs cleanup: every live
app entry point now opens ``Model-Guide.html`` directly with the right deep
link, and old bookmarks to ``ChatPromptIdeas.html`` / ``ImageGenPrompts.html``
/ ``ModelDemoPrompts.html`` / ``model-value-props.html`` will 404 by design.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import catalog
from src.model_guide import write_model_guide


def main() -> None:
    docs_dir = ROOT / "docs"
    write_model_guide(
        catalog.load_catalog(ROOT / "models_catalog.json"),
        docs_dir / "Model-Guide.html",
    )


if __name__ == "__main__":
    main()
