"""Scopus API client: Search API + Abstract Retrieval API.

- Search API gives lightweight hit metadata (used to enumerate the corpus).
- Abstract Retrieval API (FULL view, the richest view this endpoint accepts;
  COMPLETE is a Search-API-only view and returns 400 here) gives the rich
  record: full abstract, author keywords, index terms, every author with their
  Scopus author-id, affiliations with org-id + country, references, and
  subject-area classifications.

Both are wrapped with retry/backoff and an on-disk cache so re-runs are free.
"""

from __future__ import annotations

import time
from typing import Any, Iterator

import requests

from .cache import JsonCache
from .config import CONFIG, Config


class ScopusClient:
    def __init__(self, cfg: Config = CONFIG):
        self.cfg = cfg
        self.s = cfg.scopus
        self._session = requests.Session()
        self._abstract_cache = JsonCache(cfg.cache_dir / "abstracts.sqlite")

    # ------------------------------------------------------------------ #
    # low-level GET with retry/backoff
    # ------------------------------------------------------------------ #
    def _get(self, url: str, params: dict) -> dict:
        headers = self.s.headers()
        last_exc: Exception | None = None
        for attempt in range(self.s.max_retries):
            try:
                resp = self._session.get(
                    url, params=params, headers=headers, timeout=self.s.timeout_s
                )
            except requests.RequestException as exc:  # network hiccup
                last_exc = exc
                time.sleep(2 ** attempt)
                continue

            if resp.status_code == 200:
                time.sleep(self.s.request_delay_s)
                return resp.json()

            # 429 = rate limited, 5xx = transient -> backoff & retry
            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after = float(resp.headers.get("Retry-After", 2 ** attempt))
                time.sleep(retry_after)
                last_exc = RuntimeError(f"{resp.status_code}: {resp.text[:300]}")
                continue

            # 404 / 401 / 400 etc. are not retryable
            resp.raise_for_status()

        raise RuntimeError(f"Scopus GET failed after retries: {url}\n{last_exc}")

    # ------------------------------------------------------------------ #
    # Search API
    # ------------------------------------------------------------------ #
    def search(self, query: str, *, count: int = 25, max_results: int | None = None) -> list[dict]:
        """Page through the Scopus Search API and return all hit entries."""
        out: list[dict] = []
        start = 0
        total: int | None = None
        while True:
            data = self._get(
                self.s.search_url,
                {"query": query, "count": count, "start": start},
            )
            results = data.get("search-results", {})
            if total is None:
                total = int(results.get("opensearch:totalResults", 0))
            entries = results.get("entry", [])
            if not entries or entries[0].get("error"):
                break
            out.extend(entries)
            start += count
            if start >= (total or 0) or (max_results and len(out) >= max_results):
                break
        if max_results:
            out = out[:max_results]
        return out

    # ------------------------------------------------------------------ #
    # Abstract Retrieval API
    # ------------------------------------------------------------------ #
    @staticmethod
    def _scopus_id_from_eid(eid: str) -> str:
        # EID looks like "2-s2.0-85012345678"; scopus_id is the trailing digits.
        return eid.split("-")[-1] if eid else ""

    def abstract(self, scopus_id: str, *, force: bool = False) -> dict | None:
        """Fetch one FULL abstract record (cached by scopus_id)."""
        key = f"sid:{scopus_id}"
        if not force and self._abstract_cache.has(key):
            return self._abstract_cache.get(key)
        try:
            data = self._get(
                f"{self.s.abstract_url}/{scopus_id}",
                {"view": "FULL"},
            )
        except RuntimeError as exc:
            # Cache the failure marker so we don't hammer a dead id every run.
            self._abstract_cache.set(key, {"_error": str(exc)})
            return None
        self._abstract_cache.set(key, data)
        return data

    def abstracts_for_eids(
        self, eids: list[str], *, progress: bool = True
    ) -> Iterator[tuple[str, dict | None]]:
        """Yield (eid, record) for each EID, using the cache."""
        n = len(eids)
        for i, eid in enumerate(eids, 1):
            sid = self._scopus_id_from_eid(eid)
            rec = self.abstract(sid) if sid else None
            if progress and (i % 10 == 0 or i == n):
                print(f"  abstracts {i}/{n}", flush=True)
            yield eid, rec

    def probe_entitlement(self, scopus_id: str) -> dict[str, Any]:
        """Diagnostic: confirm your API key unlocks the FULL view.

        Returns which rich fields actually came back for one record so you can
        verify access before launching the full enrichment run.
        """
        rec = self.abstract(scopus_id, force=True)
        if not rec or "_error" in (rec or {}):
            return {"ok": False, "detail": (rec or {}).get("_error", "no record")}
        core = rec.get("abstracts-retrieval-response", {})
        coredata = core.get("coredata", {})
        return {
            "ok": True,
            "has_abstract": bool(coredata.get("dc:description")),
            "has_author_keywords": "authkeywords" in core,
            "has_affiliations": "affiliation" in core,
            # References live at tail.bibliography.reference (same path enrich.py
            # parses), not tail.references.
            "has_references": bool(
                (core.get("item", {}).get("bibrecord", {}).get("tail", {}) or {})
                .get("bibliography", {}) or {}
            ),
        }
