"""Stage 8 -- temporal trend analytics.

Turns the labeled corpus into the time-series that answer the user's questions:

  papers_per_year / citations_per_year      : overall volume & impact
  subfield_year_matrix                       : counts per subfield per year
  subfield_growth                            : CAGR + recent vs early share
  emergence                                  : first-appearance year per subfield/
                                               keyword/method/benchmark
  keyword_trends                             : rising vs fading author keywords
  burst_detection                            : simple burst score per term-year
  top_cited_per_year                         : most-cited paper(s) each year
  hype_vs_impact                             : subfield volume vs mean impact

All return tidy DataFrames; viz.py renders them.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .config import CONFIG, Config


# --------------------------------------------------------------------------- #
def papers_per_year(papers: pd.DataFrame) -> pd.DataFrame:
    g = (
        papers.groupby("year")
        .agg(n_papers=("eid", "count"),
             total_citations=("citedby_count", "sum"),
             mean_citations=("citedby_count", "mean"))
        .reset_index()
        .sort_values("year")
    )
    return g


def subfield_year_matrix(subfields_long: pd.DataFrame) -> pd.DataFrame:
    m = (
        subfields_long.groupby(["year", "subfield"])
        .size()
        .reset_index(name="n_papers")
        .pivot(index="year", columns="subfield", values="n_papers")
        .fillna(0)
        .astype(int)
        .sort_index()
    )
    return m


def subfield_share_matrix(subfields_long: pd.DataFrame, papers: pd.DataFrame) -> pd.DataFrame:
    """Per-year subfield counts normalized by that year's paper count (share)."""
    counts = subfield_year_matrix(subfields_long)
    totals = papers.groupby("year")["eid"].count()
    share = counts.div(totals, axis=0).fillna(0)
    return share


def subfield_growth(subfields_long: pd.DataFrame, papers: pd.DataFrame) -> pd.DataFrame:
    """Growth metrics per subfield: early vs recent share + CAGR of counts."""
    counts = subfield_year_matrix(subfields_long)
    years = counts.index.tolist()
    if not years:
        return pd.DataFrame()
    mid = years[len(years) // 2]
    early = counts.loc[counts.index <= mid].sum()
    recent = counts.loc[counts.index > mid].sum()

    first, last = years[0], years[-1]
    span = max(1, last - first)
    start_counts = counts.loc[first].replace(0, np.nan)
    end_counts = counts.loc[last]
    cagr = (end_counts / start_counts) ** (1 / span) - 1

    out = pd.DataFrame(
        {
            "total": counts.sum(),
            "early_count": early,
            "recent_count": recent,
            "recent_minus_early": recent - early,
            "cagr": cagr,
        }
    ).sort_values("recent_minus_early", ascending=False)
    return out.reset_index().rename(columns={"index": "subfield"})


# --------------------------------------------------------------------------- #
def _explode_list_col(papers: pd.DataFrame, col: str) -> pd.DataFrame:
    sub = papers[["eid", "year", "citedby_count", col]].copy()
    sub = sub.explode(col).dropna(subset=[col])
    sub[col] = sub[col].astype(str).str.strip().str.lower()
    sub = sub[sub[col].str.len() > 1]
    return sub


def emergence(papers: pd.DataFrame, col: str = "author_keywords", min_count: int = 2) -> pd.DataFrame:
    """First-appearance year and trajectory for each term in a list-column."""
    sub = _explode_list_col(papers, col)
    grp = sub.groupby(col)
    out = grp.agg(
        first_year=("year", "min"),
        last_year=("year", "max"),
        total=("eid", "count"),
        total_citations=("citedby_count", "sum"),
    ).reset_index()
    out = out[out["total"] >= min_count]
    return out.sort_values(["first_year", "total"], ascending=[False, False])


def keyword_trends(papers: pd.DataFrame, col: str = "author_keywords", top_n: int = 25) -> pd.DataFrame:
    """Rising vs fading terms: recent-half count minus early-half count."""
    sub = _explode_list_col(papers, col)
    years = sorted(sub["year"].dropna().unique())
    if not years:
        return pd.DataFrame()
    mid = years[len(years) // 2]
    early = Counter(sub.loc[sub["year"] <= mid, col])
    recent = Counter(sub.loc[sub["year"] > mid, col])
    terms = set(early) | set(recent)
    rows = [
        {
            "term": t,
            "early": early.get(t, 0),
            "recent": recent.get(t, 0),
            "delta": recent.get(t, 0) - early.get(t, 0),
            "total": early.get(t, 0) + recent.get(t, 0),
        }
        for t in terms
    ]
    df = pd.DataFrame(rows)
    rising = df.sort_values("delta", ascending=False).head(top_n)
    fading = df.sort_values("delta").head(top_n)
    rising["direction"] = "rising"
    fading["direction"] = "fading"
    return pd.concat([rising, fading], ignore_index=True)


def burst_detection(papers: pd.DataFrame, col: str = "author_keywords", min_total: int = 3) -> pd.DataFrame:
    """Lightweight burst score: max single-year share over a term's baseline.

    For each term, compares each year's count to the term's average yearly
    count; the peak ratio flags terms with a concentrated surge (a "burst").
    A full Kleinberg state-machine is overkill for ~170 papers.
    """
    sub = _explode_list_col(papers, col)
    by = sub.groupby([col, "year"]).size().reset_index(name="n")
    rows = []
    for term, g in by.groupby(col):
        total = g["n"].sum()
        if total < min_total:
            continue
        n_years = g["year"].nunique()
        baseline = total / max(1, n_years)
        peak = g.loc[g["n"].idxmax()]
        rows.append(
            {
                "term": term,
                "total": int(total),
                "peak_year": int(peak["year"]),
                "peak_count": int(peak["n"]),
                "burst_score": peak["n"] / baseline,
            }
        )
    cols = ["term", "total", "peak_year", "peak_count", "burst_score"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("burst_score", ascending=False)


# --------------------------------------------------------------------------- #
def top_cited_per_year(papers: pd.DataFrame, k: int = 1) -> pd.DataFrame:
    cols = ["year", "title", "citedby_count", "citations_per_year",
            "subfields", "first_author"]
    cols = [c for c in cols if c in papers.columns]
    return (
        papers.sort_values("citedby_count", ascending=False)
        .groupby("year", group_keys=False)
        .head(k)[cols]
        .sort_values(["year", "citedby_count"], ascending=[True, False])
    )


def hype_vs_impact(subfields_long: pd.DataFrame) -> pd.DataFrame:
    """Per-subfield volume vs citation impact -> 4-quadrant scatter input."""
    g = (
        subfields_long.groupby("subfield")
        .agg(n_papers=("eid", "count"),
             total_citations=("citedby_count", "sum"),
             mean_citations=("citedby_count", "mean"),
             mean_cpy=("citations_per_year", "mean"))
        .reset_index()
    )
    return g.sort_values("n_papers", ascending=False)


def compute_all(
    papers: pd.DataFrame,
    subfields_long: pd.DataFrame,
    cfg: Config = CONFIG,
    *,
    save: bool = True,
) -> dict[str, pd.DataFrame]:
    out = {
        "papers_per_year": papers_per_year(papers),
        "subfield_year_matrix": subfield_year_matrix(subfields_long).reset_index(),
        "subfield_share_matrix": subfield_share_matrix(subfields_long, papers).reset_index(),
        "subfield_growth": subfield_growth(subfields_long, papers),
        "keyword_emergence": emergence(papers, "author_keywords"),
        "keyword_trends": keyword_trends(papers, "author_keywords"),
        "keyword_bursts": burst_detection(papers, "author_keywords"),
        "method_emergence": emergence(papers, "ai_methods") if "ai_methods" in papers else pd.DataFrame(),
        "benchmark_emergence": emergence(papers, "benchmarks_datasets") if "benchmarks_datasets" in papers else pd.DataFrame(),
        "top_cited_per_year": top_cited_per_year(papers, k=3),
        "hype_vs_impact": hype_vs_impact(subfields_long),
    }
    if save:
        for name, df in out.items():
            if df is not None and not df.empty:
                df.to_csv(cfg.tables_dir / f"trend_{name}.csv", index=False)
        print("Saved trend tables:", ", ".join(out))
    return out
