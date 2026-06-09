"""Local LLM client (vLLM, OpenAI-compatible) with structured JSON output.

Design goals:
  - deterministic, schema-constrained outputs (vLLM ``guided_json``)
  - every call cached by a hash of (model, messages, schema) so re-runs are free
  - threaded fan-out for throughput (the corpus is small but we still parallelize)

Launch the server first (see README), e.g. on 4x48GB:

    vllm serve Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 4 \\
        --max-model-len 32768 --gpu-memory-utilization 0.92 --port 8000
"""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Sequence

from openai import BadRequestError, OpenAI

from .cache import JsonCache
from .config import CONFIG, Config

# vLLM raises a 400 like: "maximum context length is 16384 tokens. However, you
# requested 8192 output tokens and your prompt contains at least 8193 input
# tokens ...". Parse those so we can refit max_tokens to the model's window.
_CTX_RE = re.compile(
    r"maximum context length is (\d+).*?contains at least (\d+) input tokens",
    re.IGNORECASE | re.DOTALL,
)


def _refit_max_tokens(err: BadRequestError, margin: int = 128) -> int | None:
    """Given a context-length 400, return an output cap that fits, else None."""
    m = _CTX_RE.search(str(getattr(err, "message", "") or err))
    if not m:
        return None
    ctx, prompt_tokens = int(m.group(1)), int(m.group(2))
    fit = ctx - prompt_tokens - margin
    return fit if fit > 0 else None


class LLMClient:
    def __init__(self, cfg: Config = CONFIG):
        self.cfg = cfg.llm
        self.client = OpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key)
        self._cache = JsonCache(cfg.cache_dir / "llm_calls.sqlite")

    # ------------------------------------------------------------------ #
    def _key(self, messages: list[dict], schema: dict | None, **kw) -> str:
        blob = json.dumps(
            {"m": self.cfg.model, "msgs": messages, "schema": schema, "kw": kw},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def chat(
        self,
        messages: list[dict],
        *,
        schema: dict | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stream: bool = False,
        force: bool = False,
    ) -> str:
        """Single chat completion; returns the raw assistant string (cached).

        Set ``stream=True`` for long generations (e.g. the narrative report):
        tokens arrive incrementally so the client read-timeout applies *between
        tokens* rather than to the whole response, which would otherwise have to
        finish within a single ``timeout_s`` window.
        """
        key = self._key(messages, schema, mt=max_tokens, t=temperature)
        if not force and self._cache.has(key):
            return self._cache.get(key)

        extra_body: dict[str, Any] = {}
        if schema is not None and self.cfg.use_guided_json:
            extra_body["guided_json"] = schema

        kwargs: dict[str, Any] = dict(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature if temperature is None else temperature,
            timeout=self.cfg.timeout_s,
            extra_body=extra_body or None,
        )
        want_tokens = max_tokens or self.cfg.max_tokens

        # One retry: if the requested output + prompt exceeds the served model's
        # context window, refit max_tokens to what fits. This lets the same
        # LLM_MAX_TOKENS work across models with different context lengths
        # (e.g. a 32k Thinking model vs a 16k AWQ model) without manual tuning.
        for attempt in range(2):
            try:
                out = self._create(kwargs, want_tokens, stream)
                break
            except BadRequestError as err:
                fitted = _refit_max_tokens(err)
                if attempt == 1 or fitted is None:
                    raise
                print(f"  [llm] output cap {want_tokens} exceeds context; refitting to {fitted}", flush=True)
                want_tokens = fitted

        self._cache.set(key, out)
        return out

    def _create(self, kwargs: dict, max_tokens: int, stream: bool) -> str:
        """Issue one completion (streamed or not) and return the text."""
        call = dict(kwargs, max_tokens=max_tokens)
        if stream:
            parts: list[str] = []
            for chunk in self.client.chat.completions.create(**call, stream=True):
                if chunk.choices and (delta := chunk.choices[0].delta.content):
                    parts.append(delta)
            return "".join(parts)
        resp = self.client.chat.completions.create(**call)
        return resp.choices[0].message.content or ""

    def json(
        self,
        system: str,
        user: str,
        schema: dict,
        *,
        max_tokens: int | None = None,
        force: bool = False,
    ) -> dict:
        """Chat call that returns parsed JSON validated against ``schema``."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        raw = self.chat(messages, schema=schema, max_tokens=max_tokens, force=force)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Salvage the first {...} block if the model wrapped it in prose.
            start, end = raw.find("{"), raw.rfind("}")
            if 0 <= start < end:
                return json.loads(raw[start : end + 1])
            raise

    # ------------------------------------------------------------------ #
    def map_json(
        self,
        items: Sequence[Any],
        build_prompt: Callable[[Any], tuple[str, str]],
        schema: dict,
        *,
        max_tokens: int | None = None,
        progress: bool = True,
    ) -> list[dict]:
        """Run ``json`` over many items concurrently, preserving order."""

        def _one(idx_item):
            idx, item = idx_item
            system, user = build_prompt(item)
            return idx, self.json(system, user, schema, max_tokens=max_tokens)

        results: list[dict | None] = [None] * len(items)
        with ThreadPoolExecutor(max_workers=self.cfg.max_concurrency) as ex:
            done = 0
            for idx, res in ex.map(_one, enumerate(items)):
                results[idx] = res
                done += 1
                if progress and (done % 10 == 0 or done == len(items)):
                    print(f"  llm {done}/{len(items)}", flush=True)
        return results  # type: ignore[return-value]

    def health(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"LLM endpoint not reachable at {self.cfg.base_url}: {exc}")
            return False
