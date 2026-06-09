"""Stage 3 -- flatten enriched records into tidy, analysis-ready tables.

Produces:
  papers       : one row per paper (the master table)
  paper_authors: long table linking papers <-> authors (auid)
  paper_affils : long table linking papers <-> affiliations (afid, country)
  paper_keywords: long table of author keywords (one row per paper-keyword)

A "citations_per_year" impact proxy is added (total cites / years since pub).
"""

from __future__ import annotations

import datetime as _dt

import pandas as pd

from .config import CONFIG, Config


def normalize(records: list[dict], cfg: Config = CONFIG, *, save: bool = True) -> dict[str, pd.DataFrame]:
    this_year = _dt.date.today().year

    paper_rows, author_rows, affil_rows, kw_rows = [], [], [], []

    for r in records:
        eid = r["eid"]
        year = pd.to_numeric(r.get("year"), errors="coerce")
        cites = pd.to_numeric(r.get("citedby_count"), errors="coerce")
        cites = 0 if pd.isna(cites) else int(cites)
        countries = sorted(
            {a.get("country") for a in r.get("affiliations", []) if a.get("country")}
        )

        paper_rows.append(
            {
                "eid": eid,
                "scopus_id": r.get("scopus_id"),
                "doi": r.get("doi"),
                "title": r.get("title"),
                "abstract": r.get("abstract"),
                "year": year,
                "cover_date": r.get("cover_date"),
                "citedby_count": cites,
                "subtype": r.get("subtype"),
                "n_authors": len(r.get("authors", [])),
                "n_affiliations": len(r.get("affiliations", [])),
                "countries": countries,
                "n_references": r.get("n_references"),
                "openaccess": r.get("openaccess"),
                "author_keywords": r.get("author_keywords", []),
                "index_terms": r.get("index_terms", []),
                "subject_areas": [s.get("area") for s in r.get("subject_areas", [])],
            }
        )

        for a in r.get("authors", []):
            author_rows.append(
                {
                    "eid": eid,
                    "year": year,
                    "auid": a.get("auid"),
                    "name": a.get("indexed_name"),
                    "seq": pd.to_numeric(a.get("seq"), errors="coerce"),
                    "citedby_count": cites,
                }
            )
        for af in r.get("affiliations", []):
            affil_rows.append(
                {
                    "eid": eid,
                    "year": year,
                    "afid": af.get("afid"),
                    "name": af.get("name"),
                    "country": af.get("country"),
                    "citedby_count": cites,
                }
            )
        for kw in r.get("author_keywords", []):
            kw_rows.append({"eid": eid, "year": year, "keyword": (kw or "").strip().lower()})

    papers = pd.DataFrame(paper_rows)
    papers["year"] = papers["year"].astype("Int64")
    papers["years_since_pub"] = (this_year - papers["year"] + 1).clip(lower=1)
    papers["citations_per_year"] = papers["citedby_count"] / papers["years_since_pub"]

    tables = {
        "papers": papers,
        "paper_authors": pd.DataFrame(author_rows),
        "paper_affils": pd.DataFrame(affil_rows),
        "paper_keywords": pd.DataFrame(kw_rows),
    }

    if save:
        for name, df in tables.items():
            # list/dict columns -> parquet needs object; write csv with json too
            df.to_parquet(cfg.tables_dir / f"{name}.parquet", index=False)
        print("Saved tidy tables:", ", ".join(tables))
    return tables


def load_tables(cfg: Config = CONFIG) -> dict[str, pd.DataFrame]:
    names = ["papers", "paper_authors", "paper_affils", "paper_keywords"]
    return {n: pd.read_parquet(cfg.tables_dir / f"{n}.parquet") for n in names}
