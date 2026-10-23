"""Round-robin + failover across API keys, using fake Groq clients (no network)."""

from types import SimpleNamespace

import groq
import httpx
import pytest

from groundon.config import Settings
from groundon.generation.llm import GroqLLM, LLMError


def _error(cls, status, headers=None):
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.groq.com"), headers=headers or {})
    return cls("boom", response=response, body=None)


def _ok(content="hi"):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


class FakeClient:
    def __init__(self, key, script, log):
        self.key, self.script, self.log = key, script, log
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.log.append(self.key)
        outcome = self.script[self.key].pop(0) if self.script[self.key] else _ok()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_llm(script, keys=("k1", "k2"), **kwargs):
    calls, sleeps = [], []
    llm = GroqLLM(list(keys), "model", client_factory=lambda k: FakeClient(k, script, calls),
                  sleep=sleeps.append, **kwargs)
    return llm, calls, sleeps


def test_round_robin_alternates_keys():
    llm, calls, _ = make_llm({"k1": [], "k2": []})
    slots = [llm.chat([{"role": "user", "content": "x"}]).usage["key_slot"] for _ in range(4)]
    assert calls == ["k1", "k2", "k1", "k2"] and slots == [1, 2, 1, 2]


def test_rate_limited_key_fails_over_without_sleeping():
    llm, calls, sleeps = make_llm({"k1": [_error(groq.RateLimitError, 429)], "k2": []})
    result = llm.chat([{"role": "user", "content": "x"}])
    assert calls == ["k1", "k2"] and result.usage["key_slot"] == 2
    assert sleeps == []  # the other key answered immediately


def test_all_keys_limited_backs_off_with_retry_after():
    limited = lambda: _error(groq.RateLimitError, 429, {"retry-after": "3"})  # noqa: E731
    llm, calls, sleeps = make_llm({"k1": [limited()], "k2": [limited()]})
    assert llm.chat([{"role": "user", "content": "x"}]).content == "hi"
    assert calls == ["k1", "k2", "k1"] and sleeps == [3.0]


def test_gives_up_after_max_rounds():
    llm, calls, sleeps = make_llm({"k1": [_error(groq.RateLimitError, 429)] * 9,
                                   "k2": [_error(groq.RateLimitError, 429)] * 9}, max_rounds=2)
    with pytest.raises(LLMError, match="all 2 key"):
        llm.chat([{"role": "user", "content": "x"}])
    assert len(calls) == 4 and len(sleeps) == 1  # backs off between rounds, not after the last


def test_non_retryable_error_is_not_rotated():
    llm, calls, _ = make_llm({"k1": [_error(groq.NotFoundError, 404)], "k2": []})
    with pytest.raises(LLMError):
        llm.chat([{"role": "user", "content": "x"}])
    assert calls == ["k1"]  # a bad model name fails the same on every key


def test_settings_collect_numbered_keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY_2", "second")
    monkeypatch.setenv("GROQ_API_KEY_10", "tenth")
    monkeypatch.setenv("GROQ_API_KEY_1", "first")
    s = Settings(_env_file=None, groq_api_key="first", groq_api_keys="extra, second")
    assert s.api_keys == ["first", "extra", "second", "tenth"]  # ordered, deduplicated


def test_no_keys_is_a_clear_error():
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        GroqLLM([], "model", client_factory=lambda k: None)
