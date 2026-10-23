"""Thin wrapper over the Groq chat API (OpenAI-compatible), so the rest of the code
depends on a small interface that tests can fake."""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

log = logging.getLogger("groundon.llm")


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ChatResult:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


class LLMClient(Protocol):
    def chat(self, messages: list[dict], *, tools: list[dict] | None = None,
             model: str | None = None, json_mode: bool = False) -> ChatResult: ...


class LLMError(RuntimeError):
    pass


class KeyPool:
    """Thread-safe round-robin over one API client per key."""

    def __init__(self, clients: list):
        if not clients:
            raise ValueError("KeyPool needs at least one client")
        self._clients = clients
        self._next = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._clients)

    def take(self) -> tuple[int, object]:
        with self._lock:
            slot = self._next
            self._next = (slot + 1) % len(self._clients)
        return slot, self._clients[slot]


class GroqLLM:
    """Groq chat client that spreads requests over several API keys.

    Every request uses the next key (round-robin). If a key is rate-limited (429) or the
    request fails transiently, it fails over to the next key immediately; only when every
    key has failed in a row does it sleep (honouring Groq's retry-after header).

    Note: Groq enforces rate limits per *organization*. Keys from the same organization
    share one quota, so rotation only adds capacity across organizations.
    """

    def __init__(self, api_keys: list[str] | str, model: str, temperature: float = 0.0, timeout: float = 30.0,
                 *, max_rounds: int = 4, client_factory: Callable[[str], object] | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        keys = [api_keys] if isinstance(api_keys, str) else list(api_keys)
        keys = [k for k in keys if k]
        if not keys:
            raise LLMError("No Groq API key set. Add GROQ_API_KEY (or GROQ_API_KEY_1, _2, ...) to .env "
                           "(get one at https://console.groq.com/keys).")
        if client_factory is None:
            from groq import Groq

            # max_retries=0: we handle retries ourselves so a 429 rotates keys instead of waiting.
            def client_factory(key: str):
                return Groq(api_key=key, timeout=timeout, max_retries=0)

        self._pool = KeyPool([client_factory(k) for k in keys])
        self.model = model
        self.temperature = temperature
        self.max_rounds = max_rounds
        self._sleep = sleep

    @property
    def n_keys(self) -> int:
        return len(self._pool)

    def chat(self, messages, *, tools=None, model=None, json_mode=False) -> ChatResult:
        import groq

        kwargs = {"model": model or self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        n = len(self._pool)
        last_error: Exception | None = None
        for attempt in range(n * self.max_rounds):
            slot, client = self._pool.take()
            try:
                response = client.chat.completions.create(**kwargs)
                break
            except (groq.RateLimitError, groq.APIConnectionError, groq.InternalServerError) as e:
                last_error = e
                log.info("Groq key #%d failed (%s); trying next key", slot + 1, type(e).__name__)
            except groq.APIError as e:  # 400/401/404...: retrying with another key won't help
                raise LLMError(f"Groq API error: {e}") from e
            # Every key failed in a row: back off before the next round (but not after the last one).
            if (attempt + 1) % n == 0 and attempt + 1 < n * self.max_rounds:
                self._sleep(_retry_after(last_error, round_number=(attempt + 1) // n))
        else:
            raise LLMError(f"Groq API error: all {n} key(s) failed after {self.max_rounds} rounds: {last_error}")

        message = response.choices[0].message
        calls = []
        for tc in message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        usage = response.usage.model_dump() if response.usage else {}
        usage["key_slot"] = slot + 1  # which key served this call (observability; never the key itself)
        return ChatResult(content=message.content or "", tool_calls=calls, usage=usage)


def _retry_after(error: Exception | None, round_number: int) -> float:
    """Seconds to wait: Groq's retry-after header if present, else exponential backoff. Capped at 30 s."""
    response = getattr(error, "response", None)
    header = response.headers.get("retry-after") if response is not None else None
    try:
        wait = float(header) if header else 2.0 ** round_number
    except ValueError:
        wait = 2.0 ** round_number
    return min(max(wait, 0.5), 30.0)
