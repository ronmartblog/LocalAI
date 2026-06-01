# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""
Prompt content filter for image generation.

Blocks prompts containing NSFW, violent, or otherwise inappropriate terms
for enterprise / workplace use.  The blocklist is intentionally broad —
it uses substring matching on normalised text so common evasion tricks
(extra spaces, mixed case, l33t-speak basics) are caught.

The blocklist file (content_blocklist.txt) is auto-created on first run
and can be edited by admins.  One term per line, # comments allowed.
"""

import re
from pathlib import Path
from typing import Optional

# ── Default blocklist ─────────────────────────────────────────────────────────
# Covers explicit sexual content, graphic violence, slurs, and common
# prompt-engineering tricks used to bypass safety filters.
# Keep terms lowercase; matching is case-insensitive.

_DEFAULT_TERMS = """\
# Sexual / nudity
nude
naked
topless
bottomless
nsfw
pornograph
hentai
erotic
genitalia
genital
breast exposed
nipple
sexual
intercourse
orgasm
masturbat
fellatio
cunnilingus
bondage sex
bdsm
fetish sex
strip tease
striptease
upskirt
lingerie model
provocative pose

# Violence / gore
dismember
decapitat
mutilat
gore
gory
disembowel
torture scene
graphic violence
blood splatter
bloodbath

# Minors — zero tolerance
child nude
underage
loli
shota
minor nude
pedo
"""

_BLOCKLIST_FILE = "content_blocklist.txt"

# ── Runtime state ─────────────────────────────────────────────────────────────

_cached_terms: Optional[list[str]] = None


def _blocklist_path() -> Path:
    return Path(__file__).parent.parent / _BLOCKLIST_FILE


def _ensure_blocklist() -> Path:
    """Create the default blocklist file if it doesn't exist."""
    p = _blocklist_path()
    if not p.exists():
        p.write_text(_DEFAULT_TERMS, encoding="utf-8")
    return p


def _load_terms() -> list[str]:
    """Load and cache blocklist terms from file."""
    global _cached_terms
    p = _ensure_blocklist()
    terms = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        terms.append(line.lower())
    _cached_terms = terms
    return terms


def reload():
    """Force reload of the blocklist from disk."""
    global _cached_terms
    _cached_terms = None


def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace, strip basic l33t substitutions."""
    t = text.lower()
    # Common l33t substitutions
    t = t.replace("0", "o").replace("1", "i").replace("3", "e")
    t = t.replace("4", "a").replace("5", "s").replace("7", "t")
    t = t.replace("@", "a").replace("$", "s")
    # Collapse whitespace and remove repeated punctuation used as separators
    t = re.sub(r"[\s_\-.*]+", " ", t)
    return t


def check_prompt(prompt: str) -> Optional[str]:
    """Check a prompt against the blocklist.

    Returns None if the prompt is clean, or the matched term if blocked.
    """
    terms = _cached_terms if _cached_terms is not None else _load_terms()
    normalised = _normalise(prompt)

    for term in terms:
        if term in normalised:
            return term

    return None
