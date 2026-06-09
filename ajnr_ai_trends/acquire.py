"""Stage 1 -- build the candidate corpus from the Scopus Search API.

We run the AI filter *server-side* with TITLE-ABS-KEY so Scopus searches the
full abstract + indexed keywords (not just the truncated description the old
notebook saw). This yields a candidate set of EIDs; precision is tightened
later by LLM relevance confirmation in the extract stage.
"""

from __future__ import annotations

import datetime as _dt

import pandas as pd

from .config import CONFIG, Config
from .scopus import ScopusClient


def build_query(cfg: Config = CONFIG) -> str:
    c = cfg.corpus
    this_year = _dt.date.today().year
    start_year = this_year - c.years_back + 1
    terms = " OR ".join(f'"{t}"' for t in c.ai_terms)
    return (
        f"ISSN({c.ajnr_issn}) "
        f"AND PUBYEAR > {start_year - 1} AND PUBYEAR < {this_year + 1} "
        f"AND TITLE-ABS-KEY({terms})"
    )


def acquire_corpus(cfg: Config = CONFIG, *, save: bool = True) -> pd.DataFrame:
    """Return a DataFrame of candidate AI-related AJNR hits (one row per paper)."""
    client = ScopusClient(cfg)
    query = build_query(cfg)
    print("Scopus query:\n", query, sep="")
    entries = client.search(query, count=25)
    print(f"Search returned {len(entries)} candidate papers.")

    rows = []
    for p in entries:
        rows.append(
            {
                "eid": p.get("eid"),
                "scopus_id": (p.get("dc:identifier") or "").replace("SCOPUS_ID:", ""),
                "doi": p.get("prism:doi"),
                "title": p.get("dc:title"),
                "year": (p.get("prism:coverDate") or "")[:4],
                "cover_date": p.get("prism:coverDate"),
                "first_author": p.get("dc:creator"),
                "citedby_count": p.get("citedby-count"),
                "subtype": p.get("subtypeDescription"),
                "volume": p.get("prism:volume"),
                "issue": p.get("prism:issueIdentifier"),
                "page_range": p.get("prism:pageRange"),
            }
        )
    df = pd.DataFrame(rows)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["citedby_count"] = (
        pd.to_numeric(df["citedby_count"], errors="coerce").fillna(0).astype(int)
    )
    df = df.dropna(subset=["eid"]).drop_duplicates("eid").reset_index(drop=True)

    if save:
        out = cfg.tables_dir / "corpus_candidates.parquet"
        df.to_parquet(out, index=False)
        df.to_csv(cfg.tables_dir / "corpus_candidates.csv", index=False)
        print(f"Saved {len(df)} candidates -> {out}")
    return df
