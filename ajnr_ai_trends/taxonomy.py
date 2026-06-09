"""Stage 5 -- assign each paper to AI subfields.

Two complementary passes (a paper can belong to multiple subfields):

  1. rule-based: seed-keyword matching over title+abstract+keywords+extracted
     methods. Fast, transparent, good recall.
  2. LLM zero-shot: multi-label classification against the same taxonomy,
     resolving ambiguous cases the keywords miss. Cached.

The union (configurable) gives the per-paper subfield membership used by the
trend stage to chart growth/decline of LLMs, multimodal AI, RL, agents, etc.
"""

from __future__ import annotations

import json

import pandas as pd

from .config import CONFIG, Config
from .llm_client import LLMClient


def _clean_str(x: object) -> str:
    """Coerce a possibly-NaN/None scalar to a string ('' for missing)."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x)


def _clean_list(x: object) -> list[str]:
    """Coerce a possibly-NaN/None/list field to a list of strings."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return [_clean_str(i) for i in x]
    return [_clean_str(x)]


def _paper_text(row: dict) -> str:
    parts = [
        _clean_str(row.get("title")),
        _clean_str(row.get("abstract")),
        " ".join(_clean_list(row.get("author_keywords"))),
        " ".join(_clean_list(row.get("ai_methods"))),
        " ".join(_clean_list(row.get("novelty_keywords"))),
    ]
    return " ".join(parts).lower()


def rule_labels(row: dict, subfields: dict[str, list[str]]) -> list[str]:
    text = _paper_text(row)
    hits = []
    for label, seeds in subfields.items():
        if any(seed in text for seed in seeds):
            hits.append(label)
    return hits


def _llm_schema(subfields: dict[str, list[str]]) -> dict:
    return {
        "type": "object",
        "properties": {
            "subfields": {
                "type": "array",
                "items": {"type": "string", "enum": list(subfields.keys())},
            }
        },
        "required": ["subfields"],
    }


def _build_prompt(row: dict, labels: list[str]) -> tuple[str, str]:
    system = (
        "You classify a neuroradiology AI paper into one or more AI subfields "
        "from a fixed list. Choose every subfield that genuinely applies; choose "
        "none if the paper is not AI/ML. Return JSON only."
    )
    label_block = "\n".join(f"- {l}" for l in labels)
    kws = ", ".join(_clean_list(row.get("author_keywords"))) or "(none)"
    user = (
        f"SUBFIELDS:\n{label_block}\n\n"
        f"TITLE: {row.get('title')}\n"
        f"ABSTRACT: {row.get('abstract') or '(none)'}\n"
        f"KEYWORDS: {kws}\n\n"
        "Return {\"subfields\": [...]}."
    )
    return system, user


def assign_subfields(
    papers: pd.DataFrame,
    cfg: Config = CONFIG,
    *,
    use_llm: bool = True,
    combine: str = "union",   # "union" | "llm" | "rules"
    save: bool = True,
) -> pd.DataFrame:
    subfields = cfg.subfields
    rows = papers.to_dict("records")

    rule = [rule_labels(r, subfields) for r in rows]

    if use_llm:
        llm = LLMClient(cfg)
        labels = list(subfields.keys())
        schema = _llm_schema(subfields)
        res = llm.map_json(rows, lambda r: _build_prompt(r, labels), schema)
        llm_lab = [r.get("subfields", []) for r in res]
    else:
        llm_lab = [[] for _ in rows]

    final = []
    for rl, ll in zip(rule, llm_lab):
        if combine == "rules":
            final.append(sorted(set(rl)))
        elif combine == "llm":
            final.append(sorted(set(ll)))
        else:
            final.append(sorted(set(rl) | set(ll)))

    out = papers.copy()
    out["subfields_rule"] = rule
    out["subfields_llm"] = llm_lab
    out["subfields"] = final

    if save:
        out.to_parquet(cfg.tables_dir / "papers_labeled.parquet", index=False)
        # explode to a long table for easy groupby
        long = (
            out[["eid", "year", "citedby_count", "citations_per_year", "subfields"]]
            .explode("subfields")
            .dropna(subset=["subfields"])
            .rename(columns={"subfields": "subfield"})
        )
        long.to_parquet(cfg.tables_dir / "paper_subfields_long.parquet", index=False)
        print(f"Assigned subfields ({combine}); long table rows: {len(long)}")
    return out
