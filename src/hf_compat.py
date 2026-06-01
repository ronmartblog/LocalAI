# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""
hf_compat — inspect a Hugging Face repo and decide whether LocalAI can run
it, producing a fully-populated catalog entry the Add-from-Hugging-Face
dialog can drop straight into models_catalog.json.

This module is pure orchestration over a thin client adapter (_HFClient).
``huggingface_hub`` itself is imported lazily *inside* ``inspect()`` so the
~300-800 ms cold-import cost is only paid the first time the user clicks
**Check compatibility** — never at app startup.

Design contract (see docs/architecture.md §7 Backgrounding rules):

- Every HfApi call passes ``timeout=8`` so a slow / VPN'd connection
  doesn't hang the dialog.
- ``hf_revision`` on the returned entry is always a resolved 40-char commit
  SHA, never ``main`` / ``master`` / a branch name.  Extends the catalog
  rule that today only covers ``trust_remote_code`` paths.
- Detection cascade order matters and mirrors src/comfyui_client.py: GGUF
  (Z-Image → Chroma → Flux — Chroma and Z-Image filenames also contain
  "flux"), ONNX-GenAI before plain ONNX, Diffusers via
  ``model_index.json``, single-file safetensors last among image-gen
  families.
- ``recommended_settings`` and ``perf_profile`` are mandatory for image-gen
  entries (v5.1/v5.3 validators); this module always emits both, populated
  from a family-template table.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

from src import logger as _log
from src.hf_model_resolver import ParsedTarget, slug_to_model_id


__all__ = [
    "CompatResult",
    "FamilyTemplate",
    "HfNotFoundError",
    "HfGatedError",
    "HfNetworkError",
    "HfSchemaError",
    "_HFClient",
    "inspect",
    "SUPPORTED_PHASE1_PIPELINES",
]


# 40-char hexadecimal commit SHA — the only revision shape we trust on disk.
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# Network timeout (seconds) for every HfApi round-trip.  Surfaced as an
# argument so tests can override; kept short so a dead network doesn't
# silently lock the dialog for half a minute.
DEFAULT_HF_TIMEOUT = 8.0

# Pipeline tags we have a Phase 1 / Toolbox adapter for today.  Driven by
# src/phase1_adapters.py + src/workflows.py.  Keep in sync when adding a
# new adapter; the cascade falls through to "unsupported, no adapter yet"
# for any tag not in this map.
SUPPORTED_PHASE1_PIPELINES = {
    "automatic-speech-recognition": "phase1",
    "text-to-speech": "phase1",
    "feature-extraction": "phase1",
    "sentence-similarity": "phase1",
    "image-to-text": "phase1",
    "object-detection": "phase1",
}


# ── Custom exceptions ────────────────────────────────────────────────────────
# These translate huggingface_hub-specific exception classes into LocalAI's
# own taxonomy so callers (and tests) never need to import the hub's
# exception types.


class HfNotFoundError(LookupError):
    """The repo does not exist, or is private and we can't see it."""


class HfGatedError(PermissionError):
    """The repo is gated; user must accept terms on huggingface.co."""


class HfNetworkError(ConnectionError):
    """Could not reach huggingface.co within the timeout window."""


class HfSchemaError(ValueError):
    """HfApi returned a response we couldn't make sense of (missing SHA, etc.)."""


# ── Result shapes ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FamilyTemplate:
    """Defaults for one image-gen family.  Populates ``recommended_settings``
    and ``perf_profile`` on the proposed catalog entry so the v5.1/v5.3
    validators don't warn the first time the user reloads the catalog."""

    family_label: str
    width: int
    height: int
    aspect: str
    sampler: str
    scheduler: str
    steps: int
    cfg: float
    cfg_locked: bool
    speed_tier: str
    quality_tier: str
    category_bucket: str
    recommendation: str
    speed_label: str
    notes: str


# Family templates.  Cfg-locked entries (flux/chroma/z-image/sd-turbo/
# sdxl-lightning) mirror the regression-critical rule that CFG=1.0 is
# hard-forced in the ComfyUI workflow JSON for those families.
_FAMILY_TEMPLATES: dict[str, FamilyTemplate] = {
    "sd15": FamilyTemplate(
        family_label="SD 1.5", width=512, height=512, aspect="1:1",
        sampler="euler_a", scheduler="normal", steps=25, cfg=7.0, cfg_locked=False,
        speed_tier="balanced", quality_tier="good", category_bucket="general",
        recommendation="alternative", speed_label="~10s",
        notes="Imported SD 1.5-shaped checkpoint. 512x512 base resolution; CFG ~7.",
    ),
    "sdxl": FamilyTemplate(
        family_label="SDXL", width=1024, height=1024, aspect="1:1",
        sampler="dpmpp_2m", scheduler="karras", steps=30, cfg=7.0, cfg_locked=False,
        speed_tier="balanced", quality_tier="great", category_bucket="general",
        recommendation="alternative", speed_label="~25s",
        notes="Imported SDXL-shaped checkpoint. 1024x1024 base; CFG 5-8 typical.",
    ),
    "flux": FamilyTemplate(
        family_label="Flux", width=1024, height=1024, aspect="1:1",
        sampler="euler", scheduler="simple", steps=4, cfg=1.0, cfg_locked=True,
        speed_tier="fast", quality_tier="great", category_bucket="speed",
        recommendation="alternative", speed_label="~6s",
        notes="Imported Flux-family checkpoint. CFG locked at 1.0 (negatives ignored).",
    ),
    "chroma": FamilyTemplate(
        family_label="Chroma", width=1024, height=1024, aspect="1:1",
        sampler="euler", scheduler="simple", steps=28, cfg=1.0, cfg_locked=True,
        speed_tier="balanced", quality_tier="great", category_bucket="quality",
        recommendation="alternative", speed_label="~18s",
        notes="Imported Chroma checkpoint. CFG locked at 1.0 (negatives ignored).",
    ),
    "z-image": FamilyTemplate(
        family_label="Z-Image", width=1024, height=1024, aspect="1:1",
        sampler="euler", scheduler="simple", steps=8, cfg=1.0, cfg_locked=True,
        speed_tier="fast", quality_tier="great", category_bucket="speed",
        recommendation="alternative", speed_label="~7s",
        notes="Imported Z-Image checkpoint. CFG locked at 1.0 (negatives ignored).",
    ),
}


@dataclass
class CompatResult:
    """Outcome of inspecting a single HF or Ollama target.

    The dialog renders this directly: ``verdict`` drives the pill color,
    ``reasons`` populate the bullet list, and ``proposed_entry`` is what
    ``_commit_user_added_model`` writes into models_catalog.json.
    """

    verdict: str  # supported | warn | unsupported | needs_access
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    proposed_entry: dict[str, Any] = field(default_factory=dict)
    backend: str = "unknown"
    family: Optional[str] = None
    pipeline_tag: Optional[str] = None
    size_bytes_total: int = 0
    requires_trust_remote_code: bool = False
    needs_hf_token_gated: bool = False
    estimated_min_vram_gb: float = 0.0
    estimated_min_ram_gb: float = 0.0
    candidate_files: list[dict[str, Any]] = field(default_factory=list)
    resolved_sha: Optional[str] = None

    @property
    def is_install_blocked(self) -> bool:
        """True when one-click install MUST be refused even if the user
        clicked past the warning — the DO NOT REGRESS gate."""
        return self.verdict in {"unsupported", "needs_access"}


# ── Thin client adapter ──────────────────────────────────────────────────────
# Tests patch this class, not the real HfApi.  Keeps the test surface tiny
# (4 methods, all returning plain dicts) and avoids importing the hub's
# exception classes into every test file.


class _HFClient:
    """Minimal Hugging Face API adapter.  Lazy-imports huggingface_hub so
    app startup never pays for it."""

    def __init__(self, timeout: float = DEFAULT_HF_TIMEOUT, token: Optional[str] = None) -> None:
        self._timeout = timeout
        self._token = token
        self._api = None  # lazily constructed on first use

    def _ensure_api(self) -> Any:
        if self._api is None:
            # Lazy import: never load huggingface_hub at module top of
            # app.py — see docs/architecture.md for hot-path discipline.
            try:
                from huggingface_hub import HfApi  # type: ignore
            except ImportError as exc:
                raise HfNetworkError(
                    "huggingface_hub is not installed. Run: "
                    'pip install "huggingface-hub>=0.34.0,<1.0"'
                ) from exc
            self._api = HfApi(token=self._token)
        return self._api

    def model_info(self, repo_id: str, revision: Optional[str] = None) -> dict[str, Any]:
        """Return ``{sha, siblings:[{rfilename,size}], pipeline_tag, author,
        tags, card_data}``.  Translates hub exceptions into ours."""
        api = self._ensure_api()
        try:
            info = api.model_info(
                repo_id=repo_id,
                revision=revision,
                files_metadata=True,
                timeout=self._timeout,
            )
        except Exception as exc:  # pragma: no cover - exercised via tests with fakes
            self._raise_translated(exc)

        siblings_out: list[dict[str, Any]] = []
        for sib in getattr(info, "siblings", None) or []:
            siblings_out.append({
                "rfilename": getattr(sib, "rfilename", "") or "",
                "size": int(getattr(sib, "size", 0) or 0),
                "lfs": bool(getattr(sib, "lfs", None)),
            })

        card_data = getattr(info, "card_data", None)
        card_dict: dict[str, Any] = {}
        if card_data is not None and hasattr(card_data, "to_dict"):
            try:
                card_dict = card_data.to_dict() or {}
            except Exception:
                card_dict = {}
        elif isinstance(card_data, dict):
            card_dict = dict(card_data)

        return {
            "sha": getattr(info, "sha", None),
            "siblings": siblings_out,
            "pipeline_tag": getattr(info, "pipeline_tag", None),
            "library_name": getattr(info, "library_name", None),
            "tags": list(getattr(info, "tags", None) or []),
            "author": getattr(info, "author", None) or repo_id.split("/", 1)[0],
            "card_data": card_dict,
            "gated": getattr(info, "gated", None),
            "private": bool(getattr(info, "private", False)),
        }

    def fetch_text_file(self, repo_id: str, file_path: str, revision: str) -> Optional[str]:
        """Download a small text file (model_index.json, config.json, etc.)
        from the repo at the resolved SHA.  Returns ``None`` on any error
        so the caller can fall through to defaults."""
        api = self._ensure_api()
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
        except ImportError:
            return None
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                revision=revision,
                etag_timeout=self._timeout,
            )
        except Exception:
            return None
        try:
            with open(local_path, "r", encoding="utf-8") as fh:
                return fh.read()
        except OSError:
            return None

    @staticmethod
    def _raise_translated(exc: Exception) -> None:
        """Translate huggingface_hub exception types into LocalAI's taxonomy."""
        name = type(exc).__name__
        msg = str(exc) or name
        # Prefer name-based matching so we don't import hub exception classes.
        if "GatedRepo" in name or "Gated" in name:
            raise HfGatedError(msg) from exc
        if "RepositoryNotFound" in name or "EntryNotFound" in name or "NotFound" in name:
            raise HfNotFoundError(msg) from exc
        if "Timeout" in name or "Connection" in name or "Network" in name:
            raise HfNetworkError(msg) from exc
        if "RevisionNotFound" in name:
            raise HfNotFoundError(msg) from exc
        # Anything else: keep the original message but normalise the type.
        raise HfNetworkError(f"{name}: {msg}") from exc


# ── Inspect (the cascade) ────────────────────────────────────────────────────


def inspect(target: ParsedTarget, *, client: Optional[_HFClient] = None,
            platform: Optional[str] = None) -> CompatResult:
    """Inspect ``target`` and return a populated :class:`CompatResult`.

    Pass a fake ``client`` from tests; production callers leave it ``None``
    and a real :class:`_HFClient` is constructed.  ``platform`` defaults to
    ``sys.platform`` and is exposed for tests of the Windows-only OpenVINO
    branch.
    """

    if target.route == "ollama":
        return _inspect_ollama(target)

    if client is None:
        client = _HFClient()
    if platform is None:
        platform = sys.platform

    return _inspect_hf(target, client=client, platform=platform)


# ── Ollama branch ────────────────────────────────────────────────────────────


def _resolve_ollama_tags(base: str, *, timeout: float = 6.0,
                         fetcher: Optional[Any] = None) -> list[dict]:
    """Fetch and parse https://ollama.com/library/<base>/tags.

    Returns a list of dicts ``{"tag": "<base>:<size>", "size_label": "28GB"
    or "", "context_label": "128K" or ""}`` in the order they appear on the
    page (Ollama lists the canonical default first).  Returns ``[]`` if the
    page can't be fetched, is empty, or no tags can be extracted — callers
    should treat that as "tag resolution unavailable, fall back to the
    user's pasted tag and let Ollama decide".

    The page is a stable Next.js render with hrefs like
    ``/library/nemotron3:33b`` and a sibling block containing the size and
    context window.  Parsing is regex-based (no BeautifulSoup dep); a
    schema drift just yields ``[]`` which is the correct degraded mode.

    ``fetcher`` is injectable so tests don't need network.
    """
    base = (base or "").strip().lower()
    if not base or "/" in base or ":" in base:
        return []
    url = f"https://ollama.com/library/{base}/tags"
    if fetcher is None:
        from urllib.request import Request, urlopen
        def _default_fetch(u: str) -> str:
            req = Request(u, headers={"User-Agent": "LocalAI-Studio/5.3 (Add-from-HF)"})
            with urlopen(req, timeout=timeout) as resp:
                raw = resp.read(512_000)  # 512 KB cap — page is ~40 KB today
            return raw.decode("utf-8", errors="replace")
        fetcher = _default_fetch
    try:
        html = fetcher(url)
    except Exception as exc:  # noqa: BLE001 — network is best-effort
        _log.info(f"Ollama tags fetch failed for {base!r}: {exc}")
        return []
    if not html:
        return []

    # Tag hrefs look like href="/library/<base>:<size>" (the canonical one),
    # sometimes also href="/library/<base>:<size>-q4_K_M" for quants.  We
    # only want the top-level size tags the user can pull verbatim.
    href_re = re.compile(
        r'href="/library/' + re.escape(base) + r':([A-Za-z0-9_.\-]+)"'
    )
    seen: set[str] = set()
    tags: list[dict] = []
    for m in href_re.finditer(html):
        suffix = m.group(1)
        full = f"{base}:{suffix}"
        if full in seen:
            continue
        seen.add(full)
        # Look ahead a few hundred chars for the size/context block that
        # follows this href in the rendered card.
        snippet = html[m.end(): m.end() + 600]
        size_match = re.search(r"(\d+(?:\.\d+)?\s*[KMGT]B)\b", snippet)
        ctx_match = re.search(r"(\d+(?:\.\d+)?[KMG])\s*context", snippet, re.IGNORECASE)
        tags.append({
            "tag": full,
            "size_label": (size_match.group(1).replace(" ", "") if size_match else ""),
            "context_label": (ctx_match.group(1) if ctx_match else ""),
        })
    return tags


def _inspect_ollama(target: ParsedTarget) -> CompatResult:
    tag = (target.ollama_tag or "").strip()
    if not tag:
        return CompatResult(
            verdict="unsupported",
            reasons=["Could not read an Ollama tag from that URL."],
        )

    base = tag.split(":", 1)[0]
    has_size_suffix = ":" in tag

    # Real-user paste accident (Ron 2026-05-19): a bare /library/<name>
    # URL like https://ollama.com/library/nemotron3 looks like a valid tag
    # but Ollama's pull pipeline only resolves the bare name when the
    # model publisher tagged it as the canonical default — many models
    # (nemotron3, mixtral, etc.) require an explicit :size suffix or pull
    # fails with "manifest does not exist".  When no size was given,
    # consult /library/<base>/tags and either pin the canonical default
    # or warn the user with the list of options.
    extra_reasons: list[str] = []
    tag_options: list[dict] = []
    if not has_size_suffix:
        tag_options = _resolve_ollama_tags(base)
        if tag_options:
            # Use the first listed tag as the canonical default; Ollama
            # renders the recommended pull at the top of its tag list.
            default_tag = tag_options[0]["tag"]
            other_sizes = [t["tag"].split(":", 1)[1] for t in tag_options[1:]]
            tag = default_tag
            label = tag_options[0].get("size_label") or "default"
            if other_sizes:
                extra_reasons.append(
                    f"Pinned to `{default_tag}` ({label}); other sizes on "
                    f"this page: {', '.join(other_sizes[:6])}."
                )
            else:
                extra_reasons.append(
                    f"Pinned to `{default_tag}` ({label})."
                )
        else:
            extra_reasons.append(
                "Heads-up: this URL has no `:size` suffix. If Ollama can't "
                "find a default tag, the download will fail with "
                "\"manifest does not exist\" — re-add with an explicit "
                f"tag like `{base}:7b` from the model's /tags page."
            )

    entry = {
        "id": f"user-ollama-{slug_to_model_id(base)}",
        "name": tag,
        "vendor": "Ollama Library",
        "category": "Small",
        "description": f"Imported from ollama.com/library/{tag}.",
        "parameters": "",
        "size_gb": 0,
        "min_ram_gb": 8,
        "min_vram_gb": 0,
        "context_length": 0,
        "ollama_tag": tag,
        "backend": "ollama",
        "tags": ["user-added", "ollama"],
        "learn_more_url": f"https://ollama.com/library/{base}",
        "source_url": f"https://ollama.com/library/{tag}",
        "user_added": True,
        "added_at": _utcnow_iso(),
    }
    return CompatResult(
        verdict="supported",
        reasons=[
            "Routes through Ollama's pull/run pipeline. "
            "LocalAI's Chat page will use the existing Ollama backend.",
            *extra_reasons,
        ],
        proposed_entry=entry,
        backend="ollama",
    )


# ── HF branch (the main cascade) ─────────────────────────────────────────────


def _inspect_hf(target: ParsedTarget, *, client: _HFClient, platform: str) -> CompatResult:
    # 1. One model_info call — caches files, sha, pipeline tag in a single
    #    network round trip (perf-reviewer's "at most one HfApi call per
    #    inspect" guidance).
    try:
        info = client.model_info(target.repo_id, revision=target.revision)
    except HfGatedError as exc:
        return _gated_result(target.repo_id, str(exc))
    except HfNotFoundError as exc:
        return CompatResult(
            verdict="unsupported",
            reasons=[
                f"Couldn't find `{target.repo_id}` on Hugging Face. "
                "It may have been moved, renamed, or set to private."
            ],
            warnings=[str(exc)] if str(exc) else [],
        )
    except HfNetworkError as exc:
        return CompatResult(
            verdict="unsupported",
            reasons=[
                "Could not reach huggingface.co. Check your network and try again."
            ],
            warnings=[str(exc)] if str(exc) else [],
        )
    except Exception as exc:  # last-line guard so the dialog never sees a stack trace
        _log.warning(f"hf_compat: unexpected error inspecting {target.repo_id!r}: {exc}")
        return CompatResult(
            verdict="unsupported",
            reasons=[f"Couldn't inspect that repo: {exc}"],
        )

    sha = info.get("sha")
    if not isinstance(sha, str) or not _SHA_RE.match(sha.lower()):
        return CompatResult(
            verdict="unsupported",
            reasons=[
                "Hugging Face didn't return a commit SHA for that repo, so "
                "LocalAI can't pin it. Try a different model."
            ],
        )
    sha = sha.lower()

    siblings: list[dict[str, Any]] = info.get("siblings") or []
    filenames: list[str] = [s.get("rfilename") or "" for s in siblings]
    files_lower = {fn.lower() for fn in filenames}

    size_total = sum(int(s.get("size") or 0) for s in siblings)
    pipeline_tag = info.get("pipeline_tag")
    author = info.get("author") or target.repo_id.split("/", 1)[0]

    base_entry: dict[str, Any] = {
        "id": slug_to_model_id(target.repo_id),
        "name": target.repo_id.split("/", 1)[-1],
        "vendor": author,
        "category": "Small",
        "description": "",
        "parameters": "",
        "size_gb": _bytes_to_gb(size_total),
        "min_ram_gb": 8,
        "min_vram_gb": 0,
        "ollama_tag": None,
        "comfyui_model": None,
        "backend": "unknown",  # Each _classify_* MUST override; "unknown" is loud-fail bait.
        "hf_repo": target.repo_id,
        "hf_revision": sha,
        "onnx_repo": None,
        "tags": ["user-added"],
        "learn_more_url": f"https://huggingface.co/{target.repo_id}",
        "source_url": f"https://huggingface.co/{target.repo_id}",
        "user_added": True,
        "added_at": _utcnow_iso(),
        "pipeline_tag": pipeline_tag,
    }

    has_remote_code_files = any(fn.endswith(".py") for fn in filenames)
    if has_remote_code_files:
        base_entry["hf_revision"] = sha  # already pinned, re-affirm intent

    # 2. GGUF branch first — order matters per design contract and mirrors
    #    comfyui_client.py: Z-Image → Chroma → Flux, because both Z-Image
    #    and Chroma filenames can contain the substring "flux".
    gguf_files = [fn for fn in filenames if fn.lower().endswith(".gguf")]
    if gguf_files:
        return _classify_gguf(
            target=target, base_entry=base_entry, gguf_files=gguf_files,
            siblings=siblings, files_lower=files_lower, size_total=size_total,
            has_remote_code=has_remote_code_files,
        )

    # 3. ONNX-GenAI (genai_config.json present) before plain ONNX.
    if "genai_config.json" in files_lower:
        return _classify_onnx_genai(
            target=target, base_entry=base_entry, size_total=size_total,
            has_remote_code=has_remote_code_files,
        )

    # 4. Plain ONNX.
    if any(fn.endswith(".onnx") for fn in files_lower) and "config.json" in files_lower:
        return _classify_onnx(
            target=target, base_entry=base_entry, size_total=size_total,
            has_remote_code=has_remote_code_files,
        )

    # 5. OpenVINO IR.
    if "openvino_model.xml" in files_lower or (
        any(fn.endswith(".xml") for fn in files_lower)
        and any(fn.endswith(".bin") for fn in files_lower)
    ):
        return _classify_openvino(
            target=target, base_entry=base_entry, size_total=size_total,
            platform=platform, has_remote_code=has_remote_code_files,
        )

    # 6. Diffusers model_index.json → image-gen via a known pipeline class.
    if "model_index.json" in files_lower:
        return _classify_diffusers(
            target=target, base_entry=base_entry, client=client,
            sha=sha, size_total=size_total, has_remote_code=has_remote_code_files,
        )

    # 7. Single-file safetensors at root (sd15- or sdxl-shaped).
    root_safetensors = [
        s for s in siblings
        if (s.get("rfilename") or "").endswith(".safetensors")
        and "/" not in (s.get("rfilename") or "")
    ]
    if root_safetensors:
        return _classify_root_safetensors(
            target=target, base_entry=base_entry, root_safetensors=root_safetensors,
            size_total=size_total, has_remote_code=has_remote_code_files,
        )

    # 8. Phase 1 pipeline tag match (Whisper, SpeechT5, Florence, TrOCR, ...).
    if pipeline_tag and pipeline_tag in SUPPORTED_PHASE1_PIPELINES:
        return _classify_phase1(
            target=target, base_entry=base_entry, pipeline_tag=pipeline_tag,
            size_total=size_total, has_remote_code=has_remote_code_files,
        )

    # 9. Transformers text-generation fallback → warn (not unsupported)
    #    per product-designer review #6.  Users can still "Add to catalog
    #    (download later)" if they plan to wire it up themselves.
    if pipeline_tag == "text-generation":
        return _textgen_warn(target=target, base_entry=base_entry, size_total=size_total)

    return CompatResult(
        verdict="unsupported",
        reasons=[
            "LocalAI couldn't tell what kind of model this is. "
            "Open the Files tab on Hugging Face and confirm it contains "
            "GGUF, ONNX, OpenVINO, Diffusers, or single-file SD/SDXL artifacts."
        ],
        pipeline_tag=pipeline_tag,
        size_bytes_total=size_total,
        resolved_sha=sha,
    )


# ── Cascade branches ─────────────────────────────────────────────────────────


def _classify_gguf(*, target: ParsedTarget, base_entry: dict, gguf_files: list[str],
                   siblings: list[dict], files_lower: set, size_total: int,
                   has_remote_code: bool) -> CompatResult:
    """GGUF branch: only Flux / Chroma / Z-Image GGUFs are supported today.
    Generic LLM GGUFs route the user to Ollama."""

    chosen = _pick_preferred_file(target, gguf_files, siblings)
    lower = chosen.lower()
    family: Optional[str] = None
    # Order MUST mirror src/comfyui_client.py (Z-Image → Chroma → Flux). Z-Image
    # comes first because chroma_z_image-style hybrid filenames exist; Chroma
    # then beats Flux because Chroma filenames frequently also contain "flux".
    if "z_image" in lower or "z-image" in lower:
        family = "z-image"
    elif "chroma" in lower:
        family = "chroma"
    elif "flux" in lower:
        family = "flux"

    if family is None:
        return CompatResult(
            verdict="unsupported",
            reasons=[
                "This is a GGUF text model. LocalAI runs chat models through "
                "Ollama — try searching ollama.com for the same model."
            ],
            family=None,
            backend="unknown",
            size_bytes_total=size_total,
            candidate_files=_candidate_file_records(gguf_files, siblings),
            resolved_sha=base_entry["hf_revision"],
        )

    chosen_size = _file_size(chosen, siblings)
    template = _FAMILY_TEMPLATES[family]
    entry = _finalize_image_entry(
        base_entry, family=family, template=template,
        comfyui_model=chosen.rsplit("/", 1)[-1],
        comfyui_model_dest="diffusion_models",
        chosen_size_bytes=chosen_size,
    )
    return CompatResult(
        verdict="supported",
        reasons=[_image_family_reason(family, chosen)],
        warnings=_image_warnings(entry, has_remote_code),
        proposed_entry=entry,
        backend="comfyui",
        family=family,
        size_bytes_total=chosen_size,
        requires_trust_remote_code=has_remote_code,
        estimated_min_vram_gb=entry["min_vram_gb"],
        estimated_min_ram_gb=entry["min_ram_gb"],
        candidate_files=_candidate_file_records(gguf_files, siblings),
        resolved_sha=base_entry["hf_revision"],
    )


def _classify_onnx_genai(*, target: ParsedTarget, base_entry: dict,
                         size_total: int, has_remote_code: bool) -> CompatResult:
    entry = dict(base_entry)
    entry.update({
        "category": _category_for_size(_bytes_to_gb(size_total)),
        "backend": "onnx-genai",
        "ollama_tag": None,
        "comfyui_model": None,
        "onnx_repo": target.repo_id,
        "min_ram_gb": max(8, int(_bytes_to_gb(size_total) * 1.2 + 4)),
        "min_vram_gb": max(4, int(_bytes_to_gb(size_total) * 1.1)),
        "tags": sorted(set(entry.get("tags", []) + ["onnx-genai", "phi-style"])),
        "description": "Imported ONNX-GenAI chat model (Phi-4-style bundle). "
                       "Runs on the ONNX-GenAI backend with DirectML acceleration.",
    })
    return CompatResult(
        verdict="warn",
        reasons=[
            "This looks like an ONNX-GenAI chat bundle (contains a "
            "genai_config.json). LocalAI can run it once the ONNX-GenAI "
            "DirectML backend is installed."
        ],
        warnings=[
            "Before chatting with this model, install the optional "
            "ONNX-GenAI DirectML package: open a terminal and run "
            "`pip install onnxruntime-genai-directml`. LocalAI will surface "
            "a clearer error if it isn't found at load time."
        ],
        proposed_entry=entry,
        backend="onnx-genai",
        size_bytes_total=size_total,
        requires_trust_remote_code=has_remote_code,
        estimated_min_vram_gb=entry["min_vram_gb"],
        estimated_min_ram_gb=entry["min_ram_gb"],
        resolved_sha=base_entry["hf_revision"],
    )


def _classify_onnx(*, target: ParsedTarget, base_entry: dict,
                   size_total: int, has_remote_code: bool) -> CompatResult:
    entry = dict(base_entry)
    entry.update({
        "category": _category_for_size(_bytes_to_gb(size_total)),
        "backend": "onnx",
        "ollama_tag": None,
        "comfyui_model": None,
        "onnx_repo": target.repo_id,
        "min_ram_gb": max(8, int(_bytes_to_gb(size_total) * 1.2 + 2)),
        "min_vram_gb": max(2, int(_bytes_to_gb(size_total))),
        "tags": sorted(set(entry.get("tags", []) + ["onnx"])),
        "description": "Imported ONNX model. Runs on LocalAI's ONNX backend.",
    })
    return CompatResult(
        verdict="supported",
        reasons=["LocalAI can load this on the existing ONNX backend — no extra setup needed."],
        warnings=_remote_code_warning(has_remote_code),
        proposed_entry=entry,
        backend="onnx",
        size_bytes_total=size_total,
        requires_trust_remote_code=has_remote_code,
        estimated_min_vram_gb=entry["min_vram_gb"],
        estimated_min_ram_gb=entry["min_ram_gb"],
        resolved_sha=base_entry["hf_revision"],
    )


def _classify_openvino(*, target: ParsedTarget, base_entry: dict, size_total: int,
                       platform: str, has_remote_code: bool) -> CompatResult:
    is_windows = platform.startswith("win")
    entry = dict(base_entry)
    entry.update({
        "category": _category_for_size(_bytes_to_gb(size_total)),
        "backend": "openvino",
        "ollama_tag": None,
        "comfyui_model": None,
        "ov_repo": target.repo_id,
        "min_ram_gb": max(8, int(_bytes_to_gb(size_total) * 1.2 + 2)),
        "min_vram_gb": 0,
        "tags": sorted(set(entry.get("tags", []) + ["openvino"])),
        "description": "Imported OpenVINO IR model. Routes through openvino-genai.",
    })
    if not is_windows:
        return CompatResult(
            verdict="unsupported",
            reasons=["OpenVINO models only run on Windows in LocalAI today."],
            backend="openvino",
            family=None,
            size_bytes_total=size_total,
            requires_trust_remote_code=has_remote_code,
            resolved_sha=base_entry["hf_revision"],
        )
    return CompatResult(
        verdict="supported",
        reasons=["LocalAI runs OpenVINO IR models via openvino-genai on Windows."],
        warnings=_remote_code_warning(has_remote_code),
        proposed_entry=entry,
        backend="openvino",
        size_bytes_total=size_total,
        requires_trust_remote_code=has_remote_code,
        estimated_min_vram_gb=entry["min_vram_gb"],
        estimated_min_ram_gb=entry["min_ram_gb"],
        resolved_sha=base_entry["hf_revision"],
    )


# Diffusers `_class_name` → LocalAI image family.
_DIFFUSERS_CLASS_TO_FAMILY: dict[str, str] = {
    "StableDiffusionPipeline": "sd15",
    "StableDiffusionXLPipeline": "sdxl",
    "FluxPipeline": "flux",
    "ChromaPipeline": "chroma",
    "ZImagePipeline": "z-image",
    "ZImageTurboPipeline": "z-image",
}


def _classify_diffusers(*, target: ParsedTarget, base_entry: dict, client: _HFClient,
                        sha: str, size_total: int, has_remote_code: bool) -> CompatResult:
    raw = client.fetch_text_file(target.repo_id, "model_index.json", sha)
    class_name = None
    if raw:
        try:
            class_name = (json.loads(raw) or {}).get("_class_name")
        except json.JSONDecodeError:
            class_name = None

    if not class_name:
        return CompatResult(
            verdict="unsupported",
            reasons=[
                "Couldn't read the Diffusers `model_index.json` for that repo. "
                "Open the Files tab and confirm it's a Diffusers checkpoint."
            ],
            size_bytes_total=size_total,
            resolved_sha=sha,
        )

    family = _DIFFUSERS_CLASS_TO_FAMILY.get(class_name)
    if family is None:
        return CompatResult(
            verdict="unsupported",
            reasons=[
                f"LocalAI doesn't have an image-gen workflow for the "
                f"`{class_name}` family yet (covers things like SD3, "
                "PixArt, Hunyuan-DiT, AuraFlow, Lumina, Wan, CogView). "
                "Try one of the families LocalAI already supports instead: "
                "FLUX.1-schnell (`black-forest-labs/FLUX.1-schnell`), "
                "SDXL (`stabilityai/stable-diffusion-xl-base-1.0`), "
                "or Z-Image-Turbo (`Tongyi-Mai/Z-Image-Turbo`)."
            ],
            size_bytes_total=size_total,
            requires_trust_remote_code=has_remote_code,
            resolved_sha=sha,
        )

    template = _FAMILY_TEMPLATES[family]
    entry = _finalize_image_entry(
        base_entry, family=family, template=template,
        comfyui_model="",  # Diffusers pipelines load via repo, no single file
        comfyui_model_dest="checkpoints",
        chosen_size_bytes=size_total,
        hf_repo=target.repo_id,
    )
    return CompatResult(
        verdict="warn",
        reasons=[
            f"Diffusers `{class_name}` matches LocalAI's {template.family_label} "
            "image-gen path, but LocalAI ships ComfyUI checkpoints, not "
            "Diffusers folders. You'll need to convert or download a "
            "single-file equivalent before generating images."
        ],
        warnings=[
            "ComfyUI expects a single .safetensors or .gguf checkpoint, "
            "not a Diffusers folder layout. Look for a community single-file "
            "release of this model on Hugging Face."
        ],
        proposed_entry=entry,
        backend="comfyui",
        family=family,
        size_bytes_total=size_total,
        requires_trust_remote_code=has_remote_code,
        estimated_min_vram_gb=entry["min_vram_gb"],
        estimated_min_ram_gb=entry["min_ram_gb"],
        resolved_sha=sha,
    )


def _classify_root_safetensors(*, target: ParsedTarget, base_entry: dict,
                               root_safetensors: list[dict], size_total: int,
                               has_remote_code: bool) -> CompatResult:
    chosen = max(root_safetensors, key=lambda s: int(s.get("size") or 0))
    chosen_name = chosen.get("rfilename") or ""
    chosen_size = int(chosen.get("size") or 0)
    size_gb = _bytes_to_gb(chosen_size)

    # Rough shape heuristics: SD 1.5 ~2-5GB single-file, SDXL ~6-7GB.
    if size_gb >= 5.5:
        family = "sdxl"
    elif 1.8 <= size_gb <= 5.0:
        family = "sd15"
    else:
        return CompatResult(
            verdict="unsupported",
            reasons=[
                "This looks like an image-gen checkpoint, but the shape "
                "doesn't match SD 1.5 or SDXL. LocalAI may need a new "
                "ComfyUI loader before it can use it."
            ],
            size_bytes_total=chosen_size,
            requires_trust_remote_code=has_remote_code,
            candidate_files=_candidate_file_records(
                [s.get("rfilename") or "" for s in root_safetensors],
                root_safetensors,
            ),
            resolved_sha=base_entry["hf_revision"],
        )

    template = _FAMILY_TEMPLATES[family]
    entry = _finalize_image_entry(
        base_entry, family=family, template=template,
        comfyui_model=chosen_name,
        comfyui_model_dest="checkpoints",
        chosen_size_bytes=chosen_size,
        hf_repo=target.repo_id,
    )
    return CompatResult(
        verdict="supported",
        reasons=[
            f"Looks like a single-file {template.family_label} checkpoint "
            f"(`{chosen_name}`, {size_gb:.1f} GB). Routes through the "
            "existing ComfyUI workflow."
        ],
        warnings=_image_warnings(entry, has_remote_code),
        proposed_entry=entry,
        backend="comfyui",
        family=family,
        size_bytes_total=chosen_size,
        requires_trust_remote_code=has_remote_code,
        estimated_min_vram_gb=entry["min_vram_gb"],
        estimated_min_ram_gb=entry["min_ram_gb"],
        candidate_files=_candidate_file_records(
            [s.get("rfilename") or "" for s in root_safetensors],
            root_safetensors,
        ),
        resolved_sha=base_entry["hf_revision"],
    )


def _classify_phase1(*, target: ParsedTarget, base_entry: dict, pipeline_tag: str,
                     size_total: int, has_remote_code: bool) -> CompatResult:
    category_for_tag = {
        "automatic-speech-recognition": "Speech",
        "text-to-speech": "Speech",
        "feature-extraction": "Embeddings",
        "sentence-similarity": "Embeddings",
        "image-to-text": "Document AI",
        "object-detection": "Document AI",
    }
    entry = dict(base_entry)
    entry.update({
        "category": category_for_tag.get(pipeline_tag, "Small"),
        "backend": "transformers",
        "phase1_adapter": True,
        "ollama_tag": "",
        "comfyui_model": None,
        "tags": sorted(set(entry.get("tags", []) + ["phase1", pipeline_tag])),
        "description": f"Imported Hugging Face {pipeline_tag} model. "
                       "Routes through the Toolbox Phase 1 adapter.",
    })
    warnings = _remote_code_warning(has_remote_code)
    if has_remote_code:
        # Remote-code utility loaders MUST pin hf_revision — security contract.
        warnings.append(
            "This repo ships Python code that runs during model load. "
            "LocalAI has pinned the exact commit hash so the code can't "
            "change underneath you between sessions."
        )
    return CompatResult(
        verdict="warn",
        reasons=[
            f"This is a `{pipeline_tag}` model, which fits one of LocalAI's "
            "Toolbox utility slots. The model should load and run for "
            "benchmarking; a dedicated Toolbox card for this specific "
            "model may not exist yet."
        ],
        warnings=warnings,
        proposed_entry=entry,
        backend="phase1",
        pipeline_tag=pipeline_tag,
        size_bytes_total=size_total,
        requires_trust_remote_code=has_remote_code,
        estimated_min_vram_gb=entry["min_vram_gb"],
        estimated_min_ram_gb=entry["min_ram_gb"],
        resolved_sha=base_entry["hf_revision"],
    )


def _textgen_warn(*, target: ParsedTarget, base_entry: dict,
                  size_total: int) -> CompatResult:
    entry = dict(base_entry)
    entry.update({
        "category": _category_for_size(_bytes_to_gb(size_total)),
        "backend": "transformers",
        "ollama_tag": "",
        "comfyui_model": None,
        "tags": sorted(set(entry.get("tags", []) + ["transformers", "text-generation"])),
        "description": "Imported PyTorch chat model. "
                       "LocalAI's chat backends are Ollama, ONNX, and OpenVINO — "
                       "try the same model on ollama.com if available.",
        "requires_review": True,
    })
    return CompatResult(
        verdict="warn",
        reasons=[
            "This is a PyTorch chat model. LocalAI runs chat through "
            "Ollama, ONNX, or OpenVINO — search ollama.com for the "
            "same model name; that's usually the fastest one-click path."
        ],
        warnings=[
            "We'll add this to your list, but LocalAI can't kick off a "
            "download for it. To actually chat with this model you'll "
            "need to install an Ollama or ONNX build of it separately."
        ],
        proposed_entry=entry,
        backend="transformers",
        pipeline_tag="text-generation",
        size_bytes_total=size_total,
        estimated_min_vram_gb=entry["min_vram_gb"],
        estimated_min_ram_gb=entry["min_ram_gb"],
        resolved_sha=base_entry["hf_revision"],
    )


def _gated_result(repo_id: str, msg: str) -> CompatResult:
    return CompatResult(
        verdict="needs_access",
        reasons=[
            f"`{repo_id}` is gated on Hugging Face. Two steps to unlock it: "
            "(1) Open the model page below and click 'Agree and access'. "
            "(2) If you haven't already, create a Hugging Face access token "
            "(any read-only token works) and sign in on this PC by running "
            "`huggingface-cli login` in a terminal. Then come back here and "
            "click Check & Preview again."
        ],
        warnings=[msg] if msg else [],
        needs_hf_token_gated=True,
        backend="unknown",
    )


# ── Entry-finalisation helpers ───────────────────────────────────────────────


def _finalize_image_entry(base: dict, *, family: str, template: FamilyTemplate,
                          comfyui_model: str, comfyui_model_dest: str,
                          chosen_size_bytes: int,
                          hf_repo: Optional[str] = None) -> dict:
    """Build a catalog entry suitable for an image-gen row.

    Always emits ``recommended_settings`` AND ``perf_profile`` — both are
    required by the v5.1/v5.3 validators (catalog schema contract), and missing
    either will warn at app startup.
    """
    size_gb = _bytes_to_gb(chosen_size_bytes)
    entry = dict(base)
    entry.update({
        "id": f"user-img-{slug_to_model_id(base.get('id', 'imported'))}",
        "category": "Image Generation",
        "backend": "comfyui",
        "ollama_tag": None,
        "comfyui_model": comfyui_model,
        "comfyui_model_dest": comfyui_model_dest,
        "comfyui_manual_url": entry.get("source_url"),
        "hf_repo": hf_repo or entry.get("hf_repo"),
        "size_gb": size_gb,
        "min_ram_gb": max(8, int(size_gb * 1.5 + 4)),
        "min_vram_gb": _vram_floor_for_image_family(family, size_gb),
        "tags": sorted(set((entry.get("tags") or []) + [
            "user-added", "image-gen", family
        ])),
        "description": template.notes,
        "recommended_settings": {
            "width": template.width,
            "height": template.height,
            "aspect": template.aspect,
            "sampler": template.sampler,
            "scheduler": template.scheduler,
            "steps": template.steps,
            "cfg": template.cfg,
            "cfg_locked": template.cfg_locked,
            "family_label": template.family_label,
        },
        "perf_profile": {
            "speed_tier": template.speed_tier,
            "quality_tier": template.quality_tier,
            "category_bucket": template.category_bucket,
            "recommendation": template.recommendation,
            "speed_label": template.speed_label,
            "notes": template.notes,
        },
    })
    return entry


def _vram_floor_for_image_family(family: str, size_gb: float) -> int:
    # Conservative VRAM floors per family, based on Models-page perf badges.
    floor = {
        "sd15": 4,
        "sdxl": 8,
        "flux": 8,
        "chroma": 10,
        "z-image": 8,
    }.get(family, 8)
    # Larger checkpoints raise the floor.
    return max(floor, int(size_gb))


def _category_for_size(size_gb: float) -> str:
    # v5.5.1: tightened buckets so Large isn't squeezed out of existence.
    # Old: <1 / <4 / <10 / <25 / >=25 — under that scheme the only
    # 12–25 GB candidate in the curated catalog (deepseek-r1:32b @
    # 19.9 GB) was hand-set to "Extra Large", leaving the benchmark
    # checklist with zero Large rows. The hand-classification is fixed
    # separately (see ``models_catalog.json``); these boundaries are
    # widened on the Medium side so 7B/8B-Q4 (~4.7 GB) and 13B-Q4
    # (~8 GB) both stay Medium, while 13B-Q8 (~13 GB) and 32B-Q4
    # (~19 GB) move cleanly into Large. Extra Large stays at the
    # 70B+ floor.
    if size_gb < 1.5:
        return "Ultra Small"
    if size_gb < 4.0:
        return "Small"
    if size_gb < 12.0:
        return "Medium"
    if size_gb < 25.0:
        return "Large"
    return "Extra Large"


def _bytes_to_gb(n: int) -> float:
    if n <= 0:
        return 0.0
    return round(n / (1024 ** 3), 2)


def _image_family_reason(family: str, chosen: str) -> str:
    family_label = _FAMILY_TEMPLATES[family].family_label
    return (
        f"Detected {family_label} checkpoint (`{chosen}`). "
        "Routes through LocalAI's existing ComfyUI workflow."
    )


def _image_warnings(entry: dict, has_remote_code: bool) -> list[str]:
    warns = []
    family_label = (entry.get("recommended_settings") or {}).get("family_label", "")
    if family_label in {"Flux", "Chroma", "Z-Image"}:
        warns.append(
            "CFG is locked at 1.0 for this family — negative prompts are ignored."
        )
    warns.extend(_remote_code_warning(has_remote_code))
    return warns


def _remote_code_warning(has_remote_code: bool) -> list[str]:
    if not has_remote_code:
        return []
    return [
        "This repo includes Python code. LocalAI pinned the commit SHA "
        "so future updates can't change the code under you."
    ]


def _pick_preferred_file(target: ParsedTarget, gguf_files: list[str],
                         siblings: list[dict]) -> str:
    """If the user pasted a /blob/ URL targeting a specific GGUF, prefer
    that. Otherwise return the smallest GGUF (quickest download)."""
    if target.file_path and target.file_path in gguf_files:
        return target.file_path
    sized = [(fn, _file_size(fn, siblings)) for fn in gguf_files]
    sized.sort(key=lambda pair: (pair[1] or 0))
    return sized[0][0] if sized else gguf_files[0]


def _file_size(name: str, siblings: list[dict]) -> int:
    for s in siblings:
        if (s.get("rfilename") or "") == name:
            return int(s.get("size") or 0)
    return 0


def _candidate_file_records(file_names: list[str], siblings: list[dict]) -> list[dict]:
    out = []
    for fn in file_names:
        out.append({"rfilename": fn, "size": _file_size(fn, siblings)})
    out.sort(key=lambda r: (r.get("size") or 0))
    return out


def _utcnow_iso() -> str:
    """Stamp ISO-8601 (UTC, second precision) for ``added_at``."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
