# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""
hf_model_resolver — parse Hugging Face and Ollama URLs into a structured
target the Add-from-Hugging-Face flow can act on.

This module is **pure**: no network I/O, no Tk, no huggingface_hub import.
Everything is a function over strings so tests can cover every URL shape
without any heavy dependency.

The Add-from-HF dialog calls parse_url() with whatever the user pasted; the
result tells the next layer (src/hf_compat.py) whether to hit the HF API,
the Ollama branch, or to reject the input outright.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from urllib.parse import unquote, urlparse


__all__ = [
    "InvalidHFUrl",
    "ParsedTarget",
    "parse_url",
    "slug_to_model_id",
]


# Hosts we treat as the Hugging Face hub.  hf.co is a published short host;
# the rest are typos we choose not to silently rewrite.
_HF_HOSTS = {"huggingface.co", "www.huggingface.co", "hf.co"}
_OLLAMA_HOSTS = {"ollama.com", "www.ollama.com"}

# Repo-namespace segments that are NOT model repos.  Pasting one of these
# is almost always a mistake; reject early with a clear message rather than
# letting HfApi return a confusing error later.
_HF_NON_MODEL_PREFIXES = {"spaces", "datasets", "collections", "papers", "organizations", "settings"}

# Path segments we accept on a model URL after <org>/<name>.  Anything else
# (e.g. /discussions, /commits) is non-model UI and gets rejected.
_HF_MODEL_SUBPATHS = {"tree", "blob", "resolve", "raw"}


class InvalidHFUrl(ValueError):
    """The pasted URL or slug is not a usable Hugging Face / Ollama target."""


@dataclass(frozen=True)
class ParsedTarget:
    """Result of parse_url().

    Attributes
    ----------
    route:
        ``"hf"`` for a Hugging Face repo (model_info call required) or
        ``"ollama"`` for an Ollama Library tag (no HF API call).
    repo_id:
        ``"<org>/<name>"`` for HF, empty for Ollama.
    revision:
        Branch / tag / commit SHA from the URL, or ``None`` when the user
        pasted a bare repo URL.  Caller is responsible for resolving this
        to a 40-char commit SHA via the HF API before writing to the
        catalog (the DO NOT REGRESS rule on pinned ``hf_revision``).
    file_path:
        Specific file inside the repo when the URL was a /blob/ or
        /resolve/ link, else ``None``.
    ollama_tag:
        Ollama tag string (``"llama3.2:1b"``) when ``route == "ollama"``,
        else ``None``.
    """

    route: str
    repo_id: str = ""
    revision: Optional[str] = None
    file_path: Optional[str] = None
    ollama_tag: Optional[str] = None


def parse_url(raw: object) -> ParsedTarget:
    """Parse a pasted URL or slug.  Raises :class:`InvalidHFUrl` on rejection.

    Accepts (HF):

    - ``https://huggingface.co/<org>/<name>``  (trailing slash optional)
    - ``https://huggingface.co/<org>/<name>/tree/<rev>``
    - ``https://huggingface.co/<org>/<name>/blob/<rev>/<file...>``
    - ``https://huggingface.co/<org>/<name>/resolve/<rev>/<file...>``
    - ``https://hf.co/<org>/<name>``  (short host rewritten to huggingface.co)
    - ``<org>/<name>``  (bare slug, no scheme)
    - URLs with query strings (``?utm_source=...``) — all params are stripped

    Accepts (Ollama):

    - ``https://ollama.com/library/<tag>``
    - ``ollama:<tag>``  (bare scheme-style slug)

    Rejects (with :class:`InvalidHFUrl`):

    - Empty / whitespace / None
    - ``http://`` (force-upgrade by rejecting; user can re-paste as https)
    - HF spaces / datasets / collections / papers / org-only URLs
    - HF non-model subpaths (/discussions, /commits, /community)
    - Path-traversal characters in the repo id (``..``)
    - Civitai or other non-supported hosts
    """
    if raw is None:
        raise InvalidHFUrl("Paste a Hugging Face URL or `org/name` to add a model.")
    text = str(raw).strip()
    if not text:
        raise InvalidHFUrl("Paste a Hugging Face URL or `org/name` to add a model.")

    # Bare "ollama:<tag>" scheme — quickest path, no URL parsing needed.
    lowered = text.lower()
    if lowered.startswith("ollama:"):
        tag = text.split(":", 1)[1].strip()
        if not tag or "/" in tag:
            raise InvalidHFUrl(f"`{text}` is not a valid Ollama tag.")
        return ParsedTarget(route="ollama", ollama_tag=tag)

    # Bare slug "org/name" (no scheme).  Has to come before urlparse, which
    # would happily parse "org/name" as scheme-less.
    if "://" not in text and not text.startswith("//"):
        return _parse_bare_slug(text)

    # Anything with a scheme.  Lock to https — http URLs get rejected so we
    # never silently let credentials traverse plaintext.
    parsed = urlparse(text)
    scheme = (parsed.scheme or "").lower()
    if scheme == "http":
        raise InvalidHFUrl("Use the https:// version of this URL.")
    if scheme != "https":
        raise InvalidHFUrl(f"Unsupported URL scheme `{scheme}://`.")

    # Paste-accident guard: a second scheme marker in the path means the user
    # pasted on top of a selection that already had a URL, so we're looking
    # at "https://ollama.com/library/https://ollama.com/library/nemotron3".
    # Letting that flow through to Ollama yields a confusing "manifest does
    # not exist" because we'd ship a malformed tag.  Reject loudly here with
    # a copy-friendly message.  (Query strings are not in parsed.path, so a
    # legitimate ?ref=https://... URL is unaffected.)
    path_str = parsed.path or ""
    if "://" in path_str or "https:/" in path_str or "http:/" in path_str:
        raise InvalidHFUrl(
            "That looks like two URLs pasted together — happens when paste "
            "lands on top of an existing URL selection. Clear the field "
            "and paste the URL once."
        )

    host = (parsed.hostname or "").lower()
    if host in _OLLAMA_HOSTS:
        return _parse_ollama_url(parsed.path)
    if host in _HF_HOSTS:
        return _parse_hf_url(parsed.path)

    raise InvalidHFUrl(
        f"`{host}` isn't supported — paste a huggingface.co or ollama.com URL."
    )


def _parse_bare_slug(text: str) -> ParsedTarget:
    if text.count("/") != 1:
        raise InvalidHFUrl(
            "Paste a Hugging Face URL or `org/name` (for example "
            "`black-forest-labs/FLUX.1-schnell`)."
        )
    org, name = (s.strip() for s in text.split("/", 1))
    _validate_repo_id(org, name)
    return ParsedTarget(route="hf", repo_id=f"{org}/{name}")


def _parse_ollama_url(path: str) -> ParsedTarget:
    parts = [unquote(p) for p in (path or "").split("/") if p]
    # Accept /library/<tag>, optionally with a :<size> suffix.  Ollama tags
    # are a single path segment (no slashes); anything deeper is either a
    # paste accident or a non-model URL we don't support.
    if len(parts) < 2 or parts[0] != "library":
        raise InvalidHFUrl("Ollama URL must point to /library/<tag>.")
    if len(parts) > 2:
        raise InvalidHFUrl(
            "Ollama URL has extra path segments after the tag. Use a clean "
            "https://ollama.com/library/<tag> URL — for example "
            "https://ollama.com/library/llama3.2:1b."
        )
    tag = parts[1]
    # Tag validation: no slashes anywhere; at most one ':' (separating
    # name from size, e.g. "llama3.2:1b"); name half cannot be empty.
    if not tag or "/" in tag:
        raise InvalidHFUrl(f"`{tag}` is not a valid Ollama tag.")
    name_half, _, size_half = tag.partition(":")
    if not name_half or ":" in size_half:
        raise InvalidHFUrl(f"`{tag}` is not a valid Ollama tag.")
    return ParsedTarget(route="ollama", ollama_tag=tag)


def _parse_hf_url(path: str) -> ParsedTarget:
    parts = [unquote(p) for p in (path or "").split("/") if p]
    if not parts:
        raise InvalidHFUrl("Paste a model URL like huggingface.co/<org>/<name>.")
    first = parts[0]
    if first in _HF_NON_MODEL_PREFIXES:
        raise InvalidHFUrl(
            f"That looks like a Hugging Face {first} URL, not a model repo. "
            "Open the model page and copy that URL instead."
        )
    if len(parts) < 2:
        raise InvalidHFUrl(
            f"`huggingface.co/{first}` is just an organization. "
            "Open a specific model and paste its URL."
        )
    org, name = parts[0], parts[1]
    _validate_repo_id(org, name)

    if len(parts) == 2:
        return ParsedTarget(route="hf", repo_id=f"{org}/{name}")

    sub = parts[2]
    if sub not in _HF_MODEL_SUBPATHS:
        raise InvalidHFUrl(
            f"`/{sub}` isn't a model URL — open the model's Files tab and "
            "copy a /tree/, /blob/, or /resolve/ URL instead."
        )

    if len(parts) < 4:
        raise InvalidHFUrl(f"`/{sub}` URL is missing the revision name.")
    revision = parts[3]
    if revision in {"..", "."}:
        raise InvalidHFUrl("Revision name is not valid.")

    file_path: Optional[str] = None
    if sub in {"blob", "resolve", "raw"}:
        if len(parts) < 5:
            raise InvalidHFUrl(f"`/{sub}/{revision}` URL is missing the file path.")
        file_path = "/".join(parts[4:])
        if ".." in file_path.split("/"):
            raise InvalidHFUrl("File path is not valid.")

    return ParsedTarget(
        route="hf",
        repo_id=f"{org}/{name}",
        revision=revision,
        file_path=file_path,
    )


def _validate_repo_id(org: str, name: str) -> None:
    for piece, label in ((org, "organization"), (name, "model name")):
        if not piece:
            raise InvalidHFUrl(f"Repo {label} cannot be empty.")
        if piece in {".", ".."} or "/" in piece:
            raise InvalidHFUrl(f"`{piece}` is not a valid repo {label}.")
        if any(ch in piece for ch in (" ", "\t", "\n", "\\")):
            raise InvalidHFUrl(f"`{piece}` contains characters that aren't allowed in a repo {label}.")


def slug_to_model_id(repo_id: str, *, suffix: str = "") -> str:
    """Convert ``"Black-Forest-Labs/FLUX.1-schnell"`` to ``"black-forest-labs-flux-1-schnell"``.

    Used as the proposed catalog id for a user-added model.  Lowercase,
    alnum-or-dash only, collapses runs of dashes, strips edge dashes.
    Optional ``suffix`` is joined with a single dash for collision-avoidance
    callers (``-2``, ``-userN``, etc.).
    """
    base = repo_id.replace("/", "-")
    cleaned = []
    for ch in base.lower():
        if ch.isalnum():
            cleaned.append(ch)
        else:
            cleaned.append("-")
    out = "".join(cleaned)
    while "--" in out:
        out = out.replace("--", "-")
    out = out.strip("-")
    if suffix:
        suffix_clean = "".join(c if c.isalnum() else "-" for c in suffix.lower()).strip("-")
        if suffix_clean:
            out = f"{out}-{suffix_clean}"
    return out or "user-added-model"
