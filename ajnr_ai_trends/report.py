"""Stage 10 -- assemble an LLM-written narrative report.

Feeds the computed trend tables (per year and overall) to the local LLM and
asks for a grounded, citation-style narrative: what was hot each year, how
themes evolved, what the top-cited papers shared, which subfields rose/fell,
which methods/benchmarks emerged, and who drove it. The model is instructed to
use ONLY the supplied numbers (no outside knowledge) to avoid hallucination.

Outputs a markdown report under data/reports/ and returns its path.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .config import CONFIG, Config
from .llm_client import LLMClient


def thinking_llm(cfg: Config = CONFIG) -> LLMClient | None:
    """Build a *second* LLM client for an alternate (e.g. Thinking) model, used
    to generate a parallel report version. Returns None unless both
    ``THINKING_LLM_BASE_URL`` and ``THINKING_LLM_MODEL`` are set, so the main
    pipeline is unaffected when the alt endpoint isn't running.

    Serve the alt model on its own port (you have the GPUs for two at once),
    e.g. Instruct on 8001 and Thinking on 8002, then export::

        THINKING_LLM_BASE_URL=http://localhost:8002/v1
        THINKING_LLM_MODEL=Qwen/Qwen3-30B-A3B-Thinking-2507
        THINKING_LLM_MAX_TOKENS=8192
    """
    base = os.getenv("THINKING_LLM_BASE_URL")
    model = os.getenv("THINKING_LLM_MODEL")
    if not (base and model):
        return None
    alt = replace(
        cfg.llm,
        base_url=base,
        model=model,
        max_tokens=int(os.getenv("THINKING_LLM_MAX_TOKENS", "8192")),
    )
    return LLMClient(replace(cfg, llm=alt))


def _cited_record(r: pd.Series) -> dict:
    """One paper's citation facts (raw + age-adjusted) for the report."""
    cpy = r.get("citations_per_year")
    return {
        "title": r["title"],
        "year": int(r["year"]) if pd.notna(r.get("year")) else None,
        "citations": int(r["citedby_count"]),
        "citations_per_year": round(float(cpy), 1) if pd.notna(cpy) else 0.0,
        "subfields": r.get("subfields"),
        "summary": r.get("one_sentence_summary"),
    }


def _slim(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Keep only the report-relevant columns (drops noisy IDs) and round means."""
    if df.empty:
        return df
    df = df[[c for c in cols if c in df.columns]].copy()
    if "mean_citations" in df.columns:
        df["mean_citations"] = df["mean_citations"].round(1)
    return df


def _impact_by_term(papers: pd.DataFrame, col: str, *, top_n: int = 12, min_papers: int = 2) -> list[dict]:
    """Rank the values in a list-column by the citations of papers using them.

    Answers "which methods / keywords / benchmarks show up in the most-cited
    work" -- per term: paper count, total citations, mean citations.
    """
    if col not in papers.columns:
        return []
    sub = papers[[col, "citedby_count"]].explode(col).dropna(subset=[col])
    sub = sub[sub[col].astype(str).str.len() > 0]
    if sub.empty:
        return []
    g = (
        sub.groupby(col)
        .agg(
            n_papers=("citedby_count", "size"),
            total_citations=("citedby_count", "sum"),
            mean_citations=("citedby_count", "mean"),
        )
        .reset_index()
    )
    g = g[g["n_papers"] >= min_papers]
    if g.empty:
        return []
    g["mean_citations"] = g["mean_citations"].round(1)
    g = g.sort_values("total_citations", ascending=False).head(top_n)
    return g.rename(columns={col: "term"}).to_dict("records")


def _rigor_trend(papers: pd.DataFrame) -> list[dict]:
    """Per-year share of papers reporting external validation / open code / open
    data / multi-center data -- a methodological-maturity trend."""
    rows = []
    for y, g in papers.groupby("year"):
        if pd.isna(y):
            continue
        row = {"year": int(y), "n_papers": int(len(g))}
        for k in ("external_validation", "code_available", "data_available"):
            if k in g.columns:
                row[f"pct_{k}"] = round(100 * g[k].fillna(False).astype(bool).mean(), 1)
        if "data_regime" in g.columns:
            row["pct_multi_center"] = round(100 * (g["data_regime"] == "multi_center").mean(), 1)
        rows.append(row)
    return rows


def _yearly_brief(papers: pd.DataFrame, subfields_long: pd.DataFrame, year: int) -> dict:
    yp = papers[papers["year"] == year]
    ysf = subfields_long[subfields_long["year"] == year]
    top = yp.sort_values("citedby_count", ascending=False).head(5)
    return {
        "year": int(year),
        "n_papers": int(len(yp)),
        # raw lifetime citations (age-biased: older years accrue more) ...
        "total_citations": int(yp["citedby_count"].sum()),
        # ... and the age-adjusted view (citedby_count / paper age, summed).
        "total_citations_age_adjusted": round(float(yp["citations_per_year"].sum()), 1),
        "top_subfields": ysf["subfield"].value_counts().head(5).to_dict(),
        "top_cited": [_cited_record(r) for _, r in top.iterrows()],
    }


def narrative_report(
    papers: pd.DataFrame,
    subfields_long: pd.DataFrame,
    trend_tables: dict[str, pd.DataFrame],
    influence_tables: dict,
    cfg: Config = CONFIG,
    *,
    save: bool = True,
    llm: LLMClient | None = None,
    out_name: str = "ajnr_ai_trends_report.md",
) -> Path:
    """Write the narrative report. Pass ``llm`` to use a specific model client
    (e.g. a Thinking model via :func:`thinking_llm`) and ``out_name`` to write a
    parallel version without clobbering the default report."""
    llm = llm or LLMClient(cfg)
    years = sorted(int(y) for y in papers["year"].dropna().unique())

    yearly = [_yearly_brief(papers, subfields_long, y) for y in years]
    growth = trend_tables.get("subfield_growth", pd.DataFrame())
    bursts = trend_tables.get("keyword_bursts", pd.DataFrame()).head(15)
    emergence = trend_tables.get("keyword_emergence", pd.DataFrame()).head(20)
    hvi = trend_tables.get("hype_vs_impact", pd.DataFrame())
    authors = _slim(
        influence_tables.get("author_influence", pd.DataFrame()).head(15),
        ["name", "n_papers", "total_citations", "mean_citations", "first_year", "last_year"],
    )
    affils = _slim(
        influence_tables.get("affil_influence", pd.DataFrame()).head(15),
        ["name", "country", "n_papers", "total_citations", "mean_citations"],
    )

    top_overall = papers.sort_values("citedby_count", ascending=False).head(5)
    top_cited_overall = [_cited_record(r) for _, r in top_overall.iterrows()]

    # Most-cited subfields (rank the per-subfield impact table by total citations).
    subfield_impact = (
        hvi.sort_values("total_citations", ascending=False).head(15).to_dict("records")
        if not hvi.empty else []
    )

    facts = {
        "corpus_summary": {
            "total_papers": int(len(papers)),
            "total_citations": int(papers["citedby_count"].sum()),
            "total_citations_age_adjusted": round(float(papers["citations_per_year"].sum()), 1),
            "mean_citations_per_paper": round(float(papers["citedby_count"].mean()), 2),
            "year_range": [years[0], years[-1]] if years else [],
        },
        "top_cited_overall": top_cited_overall,
        "yearly": yearly,
        "subfield_growth": growth.to_dict("records") if not growth.empty else [],
        "subfield_impact": subfield_impact,
        # Which methods / model families / keywords / benchmarks appear in the
        # most-cited work (ranked by total citations of the papers using them).
        "impact_methods": _impact_by_term(papers, "ai_methods"),
        "impact_model_families": _impact_by_term(papers, "model_family"),
        "impact_keywords": _impact_by_term(papers, "author_keywords"),
        "impact_benchmarks": _impact_by_term(papers, "benchmarks_datasets", min_papers=1),
        "rigor_trend": _rigor_trend(papers),
        "keyword_bursts": bursts.to_dict("records") if not bursts.empty else [],
        "emerging_keywords": emergence.to_dict("records") if not emergence.empty else [],
        "hype_vs_impact": hvi.to_dict("records") if not hvi.empty else [],
        "top_authors": authors.to_dict("records") if not authors.empty else [],
        "top_institutions": affils.to_dict("records") if not affils.empty else [],
    }

    return _render_report(facts, cfg=cfg, llm=llm, out_name=out_name, save=save)


_SYSTEM_PROMPT = (
    "You are writing the analysis section of a bibliometric study on "
    "AI in the American Journal of Neuroradiology. Write a clear, structured "
    "markdown report. Ground EVERY claim in the provided JSON facts only; do "
    "not invent papers, numbers, names, or trends. When you cite a number, it "
    "must match the data. If the data is sparse for a year, say so. "
    "For any corpus-wide total (e.g. total papers, total citations) use ONLY "
    "the values in `corpus_summary`; never sum these yourself. In particular, "
    "the `hype_vs_impact` citation counts are PER SUBFIELD and must NOT be "
    "added together -- papers carry multiple subfield labels, so summing them "
    "double-counts citations and inflates the total. "
    "If a provided list or table is empty, state briefly that the field was "
    "not reliably extracted for this corpus and move on -- never invent "
    "entries to fill an empty section."
)

_SECTIONS = (
    "Write a report with these sections:\n"
    "1. Executive summary: a substantial, standalone overview (write 4-6 "
    "paragraphs, or rich bullets) of AI-related research in AJNR over the "
    "study period that a reader could understand without the rest of the "
    "report. It MUST cover: (a) overall scope -- total_papers, year_range, "
    "total_citations and the age-adjusted total (from corpus_summary); (b) "
    "the trend trajectory of AI publishing -- is output accelerating, which "
    "years peaked in volume versus in citation impact, and the overall "
    "story of how AI in neuroradiology evolved across the decade; (c) the "
    "dominant subfields and the fastest-growing vs declining ones (from "
    "subfield_growth); (d) headline emerging methods/keywords/benchmarks; "
    "(e) the single most-cited paper; and (f) the most influential authors "
    "and institutions, naming the leaders with their paper counts and total "
    "citations. Lead with the big-picture trend analysis, not just numbers.\n"
    "2. Year-by-year evolution: for each year report papers count, BOTH the "
    "raw total_citations AND the total_citations_age_adjusted (note that raw "
    "citations favor older years simply because they had more time to accrue, "
    "so the age-adjusted figure is the fairer cross-year comparison), the top "
    "subfields, and that year's top 5 cited papers (from each year's "
    "top_cited)\n"
    "3. Subfield growth and decline (LLMs, multimodal, segmentation, "
    "reconstruction, radiomics, etc.)\n"
    "4. Most-cited subfields: a ranked markdown table from subfield_impact "
    "with columns | Subfield | Papers | Total citations | Mean citations |, "
    "ordered by total citations. Note which subfields are high-volume vs "
    "high-impact-per-paper. IMPORTANT: these are per-subfield counts and a "
    "paper may appear under several subfields, so do NOT sum the totals into "
    "a corpus figure\n"
    "5. Most-cited papers: list and discuss the overall top 5 most-cited "
    "papers (from top_cited_overall) and the characteristics they share\n"
    "6. What drives citations -- methods, model families, keywords, and "
    "benchmarks most associated with highly-cited work: use impact_methods, "
    "impact_model_families, impact_keywords, and impact_benchmarks (each "
    "ranked by total citations of the papers using that term). Present the "
    "leading terms per category with their paper counts and total/mean "
    "citations, and call out what high-impact papers tend to have in common\n"
    "7. Emerging keywords, methods, and benchmarks (from emerging_keywords "
    "and keyword_bursts -- rising vs fading terms over time)\n"
    "8. Methodological rigor and open-science trends over time: from "
    "rigor_trend, describe how the per-year share of papers reporting "
    "external validation, open code, open data, and multi-center data has "
    "changed across the period (is the field maturing methodologically?)\n"
    "9. Influential authors and institutions: present TWO ranked markdown "
    "tables -- (a) top authors with columns | Author | Papers | Total "
    "citations | Mean citations | Active years | (from top_authors: "
    "n_papers, total_citations, mean_citations, first_year-last_year), and "
    "(b) top institutions with columns | Institution | Country | Papers | "
    "Total citations | Mean citations | (from top_institutions). After each "
    "table, briefly contrast high-output vs high-impact players (e.g. many "
    "papers but modest mean citations, or few papers with outsized impact)\n"
    "10. Notable gaps and recommendations for future analysis\n\n"
)


def _render_report(
    facts: dict,
    *,
    cfg: Config = CONFIG,
    llm: LLMClient | None = None,
    out_name: str = "ajnr_ai_trends_report.md",
    save: bool = True,
    force: bool = False,
) -> Path:
    """Turn a computed ``facts`` dict into a markdown report with the given LLM."""
    llm = llm or LLMClient(cfg)
    print(f"  [report] generating with model={llm.cfg.model} @ {llm.cfg.base_url} (force={force})", flush=True)
    user = _SECTIONS + "FACTS (JSON):\n```json\n" + json.dumps(
        facts, ensure_ascii=False, indent=2, default=str
    ) + "\n```"
    md = llm.chat(
        [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": user}],
        # Thinking models spend tokens on <think> before the answer, so allow
        # headroom above the 4k baseline when max_tokens is raised. Read from the
        # client's own config so an injected (e.g. Thinking) model uses its cap.
        max_tokens=max(4096, llm.cfg.max_tokens),
        stream=True,
        force=force,
    )
    if not save:
        return Path()
    path = cfg.report_dir / out_name
    path.write_text(md, encoding="utf-8")
    (cfg.report_dir / f"{path.stem}_facts.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=2, default=str)
    )
    print(f"Saved narrative report ({llm.cfg.model}) -> {path}")
    return path


def report_from_facts(
    cfg: Config = CONFIG,
    *,
    facts_name: str = "ajnr_ai_trends_report_facts.json",
    out_name: str = "ajnr_ai_trends_report_thinking.md",
    llm: LLMClient | None = None,
    save: bool = True,
    force: bool = True,
) -> Path:
    """Regenerate a report from already-saved facts using the currently-served
    model -- no need to recompute the pipeline or reload the tables.

    Workflow when you can only serve one model at a time:
      1. Serve the Instruct model; run the full pipeline (writes
         ``ajnr_ai_trends_report_facts.json``).
      2. Stop it, serve the Thinking model, point LLM_MODEL/LLM_BASE_URL at it,
         then call this to write a second report from the same frozen facts::

             python -c "from ajnr_ai_trends import report; report.report_from_facts()"
    """
    facts = json.loads((cfg.report_dir / facts_name).read_text())
    return _render_report(facts, cfg=cfg, llm=llm, out_name=out_name, save=save, force=force)
