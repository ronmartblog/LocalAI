# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""
Thin wrapper around the Ollama REST API.
Ollama must be running on localhost:11434 (default).
"""

import json
import re
import threading
import time
from typing import Callable, Generator, Optional

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False


class OllamaError(Exception):
    pass


def _normalize_ollama_tag_for_match(tag: object) -> str:
    """Normalize Ollama tags so bare names and ``:latest`` variants match."""
    raw = str(tag or "").strip().lower()
    if not raw:
        return ""
    base, sep, suffix = raw.partition(":")
    base = base.strip()
    suffix = suffix.strip()
    if not base:
        return ""
    if not sep or not suffix:
        suffix = "latest"
    return f"{base}:{suffix}"


def ollama_tag_is_local(tag: str, local_names: set[str]) -> bool:
    """Return True when *tag* is represented in Ollama's local model list.

    Ollama canonicalizes a pull of ``mistral-nemo`` or ``phi4-mini`` to
    ``mistral-nemo:latest`` / ``phi4-mini:latest`` in ``/api/tags``. Keep that
    equivalence without treating sibling tags such as ``llama3.2:1b`` and
    ``llama3.2:3b`` as interchangeable.
    """
    raw_tag = str(tag or "").strip().lower()
    if not raw_tag:
        return False
    raw_local = {str(name or "").strip().lower() for name in (local_names or set()) if str(name or "").strip()}
    if raw_tag in raw_local:
        return True
    normalized_tag = _normalize_ollama_tag_for_match(raw_tag)
    normalized_local = {_normalize_ollama_tag_for_match(name) for name in raw_local}
    if normalized_tag and normalized_tag in normalized_local:
        return True
    base = raw_tag.split(":", 1)[0] if ":" in raw_tag else raw_tag
    return bool(base and base in raw_local)


def think_option_for_model(tag: str) -> Optional[bool]:
    """Return an Ollama think option when a model needs a UI-safe override.

    Thinking/reasoning models like Qwen 3 and DeepSeek-R1 emit a `<think>…</think>`
    block before the visible answer. With a finite `num_predict` cap (or a small
    user max_tokens), the entire budget can be consumed inside the thinking
    block, the model is force-stopped, and the visible response (after stripping)
    is empty — the UI then shows a blank bubble. Disabling thinking via Ollama's
    `think: false` option keeps the visible response intact at all budgets. This
    matches the GPU-Super bug-report finding §1.5.
    """
    base = str(tag or "").strip().lower().split(":", 1)[0]
    if base in {"qwen3", "qwen3-coder", "deepseek-r1", "magistral", "phi4-reasoning", "phi4-mini-reasoning", "phi-4-mini-reasoning", "nemotron-3-nano", "nemotron3"}:
        return False
    return None


def _apply_think_option(payload: dict, tag: str) -> None:
    think = think_option_for_model(tag)
    if think is not None:
        payload["think"] = think


def strip_think_blocks(text: str) -> str:
    """Remove model-emitted hidden-thinking blocks from completed text."""
    cleaned = re.sub(r"(?is)<think\b[^>]*>.*?</think>\s*", "", str(text or ""))
    cleaned = re.sub(r"(?is)<think\b[^>]*>.*$", "", cleaned)
    orphan_close = cleaned.lower().rfind("</think>")
    if orphan_close >= 0 and cleaned.lower().rfind("<think", 0, orphan_close) < 0:
        cleaned = cleaned[orphan_close + len("</think>"):].lstrip()
    return cleaned.strip()


class _ThinkBlockStreamFilter:
    """Suppress streamed <think>...</think> content while preserving visible tokens."""

    _OPEN = "<think"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._pending = ""
        self._in_think = False

    @staticmethod
    def _suffix_prefix_len(text: str, marker: str) -> int:
        lower = text.lower()
        marker = marker.lower()
        for length in range(min(len(marker) - 1, len(lower)), 0, -1):
            if lower.endswith(marker[:length]):
                return length
        return 0

    def feed(self, chunk: str) -> str:
        data = self._pending + str(chunk or "")
        self._pending = ""
        visible: list[str] = []
        while data:
            lower = data.lower()
            if self._in_think:
                close_index = lower.find(self._CLOSE)
                if close_index < 0:
                    keep = self._suffix_prefix_len(data, self._CLOSE)
                    self._pending = data[-keep:] if keep else ""
                    return "".join(visible)
                data = data[close_index + len(self._CLOSE):]
                self._in_think = False
                continue

            open_index = lower.find(self._OPEN)
            if open_index < 0:
                keep = self._suffix_prefix_len(data, self._OPEN)
                if keep:
                    visible.append(data[:-keep])
                    self._pending = data[-keep:]
                else:
                    visible.append(data)
                return "".join(visible)

            visible.append(data[:open_index])
            tag_end = data.find(">", open_index)
            if tag_end < 0:
                self._pending = data[open_index:]
                return "".join(visible)
            data = data[tag_end + 1:]
            self._in_think = True
        return "".join(visible)


class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host.rstrip("/")

    # ── Connectivity ──────────────────────────────────────────────────────────

    def is_running(self) -> bool:
        if not _REQUESTS_OK:
            return False
        try:
            r = requests.get(f"{self.host}/", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def version(self) -> str:
        try:
            r = requests.get(f"{self.host}/api/version", timeout=3)
            return r.json().get("version", "unknown")
        except Exception:
            return "unknown"

    # ── Model management ──────────────────────────────────────────────────────

    def list_local_models(self) -> list[dict]:
        """Return models that are already downloaded."""
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=10)
            r.raise_for_status()
            return r.json().get("models", [])
        except Exception as e:
            raise OllamaError(f"Could not list models: {e}") from e

    def local_model_names(self) -> set[str]:
        """Return a set of local model name strings."""
        try:
            models = self.list_local_models()
            return {m["name"] for m in models if m.get("name")}
        except OllamaError:
            return set()

    def is_model_local(self, tag: str) -> bool:
        return ollama_tag_is_local(tag, self.local_model_names())

    def pull_model(
        self,
        tag: str,
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
        stop_event: Optional[threading.Event] = None,
        max_retries: int = 3,
    ) -> None:
        """
        Download a model. Calls progress_cb(status, completed_bytes, total_bytes).
        Raises OllamaError on failure.

        v5: Adds exponential-backoff retry for transient network errors. The pull
        is automatically retried up to ``max_retries`` times on:
          * ``requests.ConnectionError`` (socket reset, DNS hiccup)
          * ``requests.Timeout`` (slow start)
          * Any non-error HTTP read interruption
        Errors that are clearly fatal (HTTP 4xx, ``"error"`` payload from
        the daemon) are surfaced immediately without retry. The stop_event is
        respected between retries to keep cancellation responsive.
        """
        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            if stop_event and stop_event.is_set():
                return
            try:
                with requests.post(
                    f"{self.host}/api/pull",
                    json={"name": tag, "stream": True},
                    stream=True,
                    timeout=(10, 3600),
                ) as resp:
                    if resp.status_code == 404:
                        raise OllamaError(
                            f"Model '{tag}' not found in the Ollama registry."
                        )
                    if 400 <= resp.status_code < 500:
                        raise OllamaError(
                            f"Ollama refused pull (HTTP {resp.status_code}): "
                            f"{resp.text[:200]}"
                        )
                    resp.raise_for_status()
                    for raw_line in resp.iter_lines():
                        if stop_event and stop_event.is_set():
                            return
                        if not raw_line:
                            continue
                        try:
                            data = json.loads(raw_line)
                        except json.JSONDecodeError:
                            continue
                        status = data.get("status", "")
                        completed = data.get("completed", 0)
                        total = data.get("total", 0)
                        if progress_cb:
                            progress_cb(status, completed, total)
                        if data.get("error"):
                            # Daemon-level error: don't retry, surface immediately
                            raise OllamaError(data["error"])
                return  # success
            except OllamaError:
                # Permanent error from the daemon — don't retry.
                raise
            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = e
                if attempt < max_retries - 1:
                    backoff = min(2 ** attempt, 8)  # 1, 2, 4, 8 s cap
                    if progress_cb:
                        progress_cb(
                            f"Network hiccup ({type(e).__name__}); retry "
                            f"{attempt + 1}/{max_retries - 1} in {backoff}s …",
                            0, 0,
                        )
                    # Sleep in 0.5s slices so a stop_event aborts quickly
                    waited = 0.0
                    while waited < backoff:
                        if stop_event and stop_event.is_set():
                            return
                        time.sleep(0.5)
                        waited += 0.5
                    continue
                # exhausted retries — fall through to raise
            except requests.RequestException as e:
                raise OllamaError(f"Download failed: {e}") from e
        raise OllamaError(
            f"Download failed after {max_retries} attempts: {last_err}"
        ) from last_err

    def delete_model(self, tag: str) -> None:
        try:
            errors: list[str] = []
            for key in ("model", "name"):
                r = requests.delete(f"{self.host}/api/delete", json={key: tag}, timeout=15)
                if r.status_code in (200, 204):
                    return
                errors.append(f"{key}= HTTP {r.status_code}: {r.text}")
            raise OllamaError("Delete returned " + " | ".join(errors))
        except requests.RequestException as e:
            raise OllamaError(f"Delete failed: {e}") from e

    # ── Inference ─────────────────────────────────────────────────────────────

    def chat_stream(
        self,
        tag: str,
        messages: list[dict],
        num_gpu: int = -1,
        temperature: float = 0.7,
        stop_event: Optional[threading.Event] = None,
        num_predict: int = -1,
        num_ctx: Optional[int] = None,
        keep_alive: Optional[str] = None,
        repeat_penalty: Optional[float] = None,
        repeat_last_n: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """
        Yield token strings from a streaming chat call.
        num_gpu=-1  → Ollama picks automatically (GPU if available)
        num_gpu=0   → force CPU
        num_predict    → max tokens to generate (-2 = fill context, -1 = model default)
        num_ctx        → optional context cap for the active chat call
        keep_alive     → optional model residency hint (for example "30m")
        repeat_penalty → optional float >0; sent only when provided. Useful as
                         a guard against degenerate same-paragraph loops in
                         benchmarking; the UI never sets it so default chat
                         behavior is unchanged.
        repeat_last_n  → optional int >0; sent only when provided. Window
                         (in tokens) the repeat_penalty considers.
        """
        payload = {
            "model": tag,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_gpu": num_gpu,
                "num_predict": num_predict,
            },
        }
        if isinstance(num_ctx, int) and num_ctx > 0:
            payload["options"]["num_ctx"] = num_ctx
        if isinstance(repeat_penalty, (int, float)) and not isinstance(repeat_penalty, bool):
            if float(repeat_penalty) > 0:
                payload["options"]["repeat_penalty"] = float(repeat_penalty)
        if isinstance(repeat_last_n, int) and not isinstance(repeat_last_n, bool):
            if repeat_last_n > 0:
                payload["options"]["repeat_last_n"] = int(repeat_last_n)
        if isinstance(keep_alive, str) and keep_alive.strip():
            payload["keep_alive"] = keep_alive.strip()
        _apply_think_option(payload, tag)
        try:
            with requests.post(
                f"{self.host}/api/chat",
                json=payload,
                stream=True,
                timeout=(15, 600),
            ) as resp:
                resp.raise_for_status()
                think_filter = _ThinkBlockStreamFilter()
                for raw_line in resp.iter_lines():
                    if stop_event and stop_event.is_set():
                        break
                    if not raw_line:
                        continue
                    try:
                        data = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    if data.get("error"):
                        raise OllamaError(data["error"])
                    token = data.get("message", {}).get("content", "")
                    if token:
                        visible_token = think_filter.feed(token)
                        if visible_token:
                            yield visible_token
                    if data.get("done"):
                        break
        except requests.RequestException as e:
            raise OllamaError(f"Chat failed: {e}") from e

    def chat_stream_with_stats(
        self,
        tag: str,
        messages: list[dict],
        num_gpu: int = -1,
        temperature: float = 0.7,
        stop_event: Optional[threading.Event] = None,
        num_predict: int = -1,
        num_ctx: Optional[int] = None,
        keep_alive: Optional[str] = None,
        read_timeout: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        repeat_last_n: Optional[int] = None,
    ) -> Generator[tuple[str, dict | None], None, None]:
        """
        Yield (token, stats) tuples from a streaming chat call.
        *stats* is None for every intermediate chunk and a dict on the
        final ``done=true`` chunk containing Ollama's native timing:
            total_duration, load_duration, prompt_eval_duration,
            eval_duration, eval_count  (all in nanoseconds).

        ``repeat_penalty`` / ``repeat_last_n`` mirror :py:meth:`chat_stream` and
        are only forwarded to Ollama when provided as positive values.
        """
        payload = {
            "model": tag,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_gpu": num_gpu,
                "num_predict": num_predict,
            },
        }
        if isinstance(num_ctx, int) and num_ctx > 0:
            payload["options"]["num_ctx"] = num_ctx
        if isinstance(repeat_penalty, (int, float)) and not isinstance(repeat_penalty, bool):
            if float(repeat_penalty) > 0:
                payload["options"]["repeat_penalty"] = float(repeat_penalty)
        if isinstance(repeat_last_n, int) and not isinstance(repeat_last_n, bool):
            if repeat_last_n > 0:
                payload["options"]["repeat_last_n"] = int(repeat_last_n)
        if isinstance(keep_alive, str) and keep_alive.strip():
            payload["keep_alive"] = keep_alive.strip()
        _apply_think_option(payload, tag)
        try:
            timeout = (15, read_timeout if read_timeout is not None else 600)
            with requests.post(
                f"{self.host}/api/chat",
                json=payload,
                stream=True,
                timeout=timeout,
            ) as resp:
                resp.raise_for_status()
                think_filter = _ThinkBlockStreamFilter()
                for raw_line in resp.iter_lines():
                    if stop_event and stop_event.is_set():
                        break
                    if not raw_line:
                        continue
                    try:
                        data = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    if data.get("error"):
                        raise OllamaError(data["error"])
                    token = data.get("message", {}).get("content", "")
                    if data.get("done"):
                        visible_token = think_filter.feed(token)
                        stats = {
                            "total_duration": data.get("total_duration", 0),
                            "load_duration": data.get("load_duration", 0),
                            "prompt_eval_duration": data.get("prompt_eval_duration", 0),
                            "prompt_eval_count": data.get("prompt_eval_count", 0),
                            "eval_duration": data.get("eval_duration", 0),
                            "eval_count": data.get("eval_count", 0),
                            "done_reason": data.get("done_reason", ""),
                        }
                        yield visible_token, stats
                        break
                    if token:
                        visible_token = think_filter.feed(token)
                        if visible_token:
                            yield visible_token, None
        except requests.RequestException as e:
            raise OllamaError(f"Chat failed: {e}") from e

    def unload_model(self, tag: str) -> bool:
        """
        Ask Ollama to immediately evict *tag* from VRAM (keep_alive=0).
        Returns True on success, False if the request fails.
        No-op (and returns True) if the model is not currently loaded.
        """
        try:
            r = requests.post(
                f"{self.host}/api/generate",
                json={"model": tag, "keep_alive": 0},
                timeout=10,
            )
            return r.status_code == 200
        except Exception:
            return False

    def running_models(self) -> list[dict]:
        """Return currently loaded models (Ollama /api/ps)."""
        try:
            r = requests.get(f"{self.host}/api/ps", timeout=5)
            return r.json().get("models", [])
        except Exception:
            return []

    def model_info(self, tag: str) -> dict:
        try:
            r = requests.post(f"{self.host}/api/show", json={"name": tag}, timeout=10)
            return r.json()
        except Exception:
            return {}
