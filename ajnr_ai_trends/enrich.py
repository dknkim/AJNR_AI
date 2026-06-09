"""Stage 2 -- enrich each candidate via the Abstract Retrieval API.

Pulls the FULL view for every EID and parses the deeply-nested Scopus JSON
into a flat per-paper record plus child lists (authors, affiliations,
keywords). Everything is cached, so re-runs only fetch new papers.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import CONFIG, Config
from .scopus import ScopusClient


def _as_list(x: Any) -> list:
    """Scopus collapses single-element arrays to dicts; normalize to a list."""
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def _text(x: Any) -> str | None:
    """Pull text out of Scopus's {'$': 'value'} wrappers or plain strings."""
    if isinstance(x, dict):
        return x.get("$")
    return x


def parse_record(eid: str, rec: dict) -> dict | None:
    """Flatten one abstracts-retrieval-response into a structured dict."""
    if not rec or "_error" in rec:
        return None
    core = rec.get("abstracts-retrieval-response", {})
    coredata = core.get("coredata", {})

    # ---- author keywords -------------------------------------------------- #
    author_keywords = [
        _text(k) for k in _as_list((core.get("authkeywords") or {}).get("author-keyword"))
    ]
    author_keywords = [k for k in author_keywords if k]

    # ---- index terms (Scopus controlled vocabulary) ----------------------- #
    idx = (core.get("idxterms") or {}).get("mainterm")
    index_terms = [_text(t) for t in _as_list(idx)]
    index_terms = [t for t in index_terms if t]

    # ---- subject-area classifications ------------------------------------- #
    subjects = [
        {"area": _text(a), "code": a.get("@code"), "abbrev": a.get("@abbrev")}
        for a in _as_list((core.get("subject-areas") or {}).get("subject-area"))
        if isinstance(a, dict)
    ]

    # ---- authors ---------------------------------------------------------- #
    authors = []
    for a in _as_list((core.get("authors") or {}).get("author")):
        if not isinstance(a, dict):
            continue
        pname = a.get("preferred-name", {}) or {}
        authors.append(
            {
                "auid": a.get("@auid"),
                "indexed_name": a.get("ce:indexed-name") or pname.get("ce:indexed-name"),
                "given_name": a.get("ce:given-name") or pname.get("ce:given-name"),
                "surname": a.get("ce:surname") or pname.get("ce:surname"),
                "seq": a.get("@seq"),
                "affil_ids": [
                    aff.get("@id")
                    for aff in _as_list(a.get("affiliation"))
                    if isinstance(aff, dict)
                ],
            }
        )

    # ---- affiliations ----------------------------------------------------- #
    affiliations = []
    for af in _as_list(core.get("affiliation")):
        if not isinstance(af, dict):
            continue
        affiliations.append(
            {
                "afid": af.get("@id"),
                "name": af.get("affilname"),
                "city": af.get("affiliation-city"),
                "country": af.get("affiliation-country"),
            }
        )

    # ---- reference count -------------------------------------------------- #
    tail = (
        core.get("item", {})
        .get("bibrecord", {})
        .get("tail", {})
        or {}
    )
    refs = (tail.get("bibliography") or {})
    n_refs = None
    if refs:
        try:
            n_refs = int(refs.get("@refcount"))
        except (TypeError, ValueError):
            n_refs = len(_as_list(refs.get("reference")))

    return {
        "eid": eid,
        "scopus_id": coredata.get("dc:identifier", "").replace("SCOPUS_ID:", ""),
        "doi": coredata.get("prism:doi"),
        "title": coredata.get("dc:title"),
        "abstract": coredata.get("dc:description"),
        "year": (coredata.get("prism:coverDate") or "")[:4],
        "cover_date": coredata.get("prism:coverDate"),
        "citedby_count": coredata.get("citedby-count"),
        "subtype": coredata.get("subtypeDescription"),
        "aggregation_type": coredata.get("prism:aggregationType"),
        "author_keywords": author_keywords,
        "index_terms": index_terms,
        "subject_areas": subjects,
        "authors": authors,
        "affiliations": affiliations,
        "n_references": n_refs,
        "openaccess": coredata.get("openaccess"),
    }


def enrich_corpus(
    candidates: pd.DataFrame, cfg: Config = CONFIG, *, save: bool = True
) -> list[dict]:
    """Fetch + parse FULL records for all candidate EIDs."""
    client = ScopusClient(cfg)
    eids = candidates["eid"].dropna().tolist()
    records: list[dict] = []
    n_missing = 0
    for eid, rec in client.abstracts_for_eids(eids):
        parsed = parse_record(eid, rec or {})
        if parsed is None:
            n_missing += 1
            continue
        records.append(parsed)
    print(f"Enriched {len(records)} records ({n_missing} missing/failed).")

    if save:
        import json

        out = cfg.tables_dir / "enriched_records.json"
        out.write_text(json.dumps(records, ensure_ascii=False, indent=2))
        print(f"Saved enriched records -> {out}")
    return records
