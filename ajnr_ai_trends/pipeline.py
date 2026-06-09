"""One-call orchestration of the full pipeline.

Each stage is cached, so re-running is cheap and you can stop/resume. Import and
call ``run_all()`` from the notebook, or run stages individually.
"""

from __future__ import annotations

from .config import CONFIG, Config
from . import acquire, enrich, normalize, extract, taxonomy, embed, cluster
from . import trends, influence, report


def run_all(cfg: Config = CONFIG, *, use_llm: bool = True, label_topics: bool = True):
    print("=== 1. acquire ===")
    candidates = acquire.acquire_corpus(cfg)

    print("\n=== 2. enrich ===")
    records = enrich.enrich_corpus(candidates, cfg)

    print("\n=== 3. normalize ===")
    tables = normalize.normalize(records, cfg)
    papers = tables["papers"]

    print("\n=== 4. extract (LLM) ===")
    papers = extract.extract_papers(papers, cfg) if use_llm else papers

    print("\n=== 5. taxonomy ===")
    papers = taxonomy.assign_subfields(papers, cfg, use_llm=use_llm)
    # keep relevant AI papers if the LLM relevance flag is present
    if "is_ai_relevant" in papers.columns:
        papers = papers[papers["is_ai_relevant"].fillna(True)].copy()

    print("\n=== 6. embed ===")
    emb = embed.embed_papers(papers, cfg)

    print("\n=== 7. cluster ===")
    topic_summary = cluster.discover_topics(papers, emb, cfg, label_with_llm=label_topics)

    print("\n=== 8. trends ===")
    import pandas as pd

    subfields_long = (
        papers[["eid", "year", "citedby_count", "citations_per_year", "subfields"]]
        .explode("subfields").dropna(subset=["subfields"])
        .rename(columns={"subfields": "subfield"})
    )
    trend_tables = trends.compute_all(papers, subfields_long, cfg)

    print("\n=== 9. influence ===")
    influence_tables = influence.compute_all(tables, subfields_long, cfg)

    print("\n=== 10. report (LLM) ===")
    report_path = (
        report.narrative_report(papers, subfields_long, trend_tables, influence_tables, cfg)
        if use_llm else None
    )

    # Optional parallel report from an alternate (e.g. Thinking) model, if a
    # second endpoint is configured via THINKING_LLM_BASE_URL / THINKING_LLM_MODEL.
    thinking_report_path = None
    t_llm = report.thinking_llm(cfg) if use_llm else None
    if t_llm is not None:
        print("\n=== 10b. report (Thinking model) ===")
        thinking_report_path = report.narrative_report(
            papers, subfields_long, trend_tables, influence_tables, cfg,
            llm=t_llm, out_name="ajnr_ai_trends_report_thinking.md",
        )

    return {
        "papers": papers,
        "tables": tables,
        "subfields_long": subfields_long,
        "topic_summary": topic_summary,
        "trend_tables": trend_tables,
        "influence_tables": influence_tables,
        "report_path": report_path,
        "thinking_report_path": thinking_report_path,
        "embeddings": emb,
    }
