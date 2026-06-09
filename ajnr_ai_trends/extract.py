"""Stage 4 -- per-paper structured extraction with the local LLM.

For each paper we extract a controlled, analysis-ready record from the title +
abstract + author keywords. This is where unstructured text becomes the
variables that drive every downstream trend:

  - is_ai_relevant         : LLM relevance confirmation (tightens the corpus)
  - clinical_task          : detection/diagnosis, segmentation, prognosis, ...
  - ai_methods             : free list (e.g. "U-Net", "vision transformer")
  - model_family           : CNN / transformer / LLM / GAN / diffusion / classical-ML / radiomics
  - imaging_modality       : MRI, CT, ... (multi)
  - anatomy                : brain, spine, head&neck, vessels, ...
  - disease                : stroke, glioma, MS, aneurysm, ...
  - data_regime            : single-center / multi-center / public-dataset
  - external_validation    : bool (was the model validated on external data?)
  - benchmarks_datasets    : named datasets/benchmarks (BraTS, ISLES, ...)
  - code_available / data_available : reproducibility signals
  - novelty_keywords       : terms the model thinks are notable/emerging
  - one_sentence_summary   : for reporting

These are exactly the fields needed for "what characteristics did the top-cited
papers share", "which methods/benchmarks gained traction", and subfield trends.
"""

from __future__ import annotations

import pandas as pd

from .config import CONFIG, Config
from .llm_client import LLMClient

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "is_ai_relevant": {"type": "boolean"},
        "clinical_task": {
            "type": "string",
            "enum": [
                "detection", "diagnosis_classification", "segmentation",
                "image_reconstruction", "prognosis_outcome", "image_generation",
                "report_generation_nlp", "workflow_triage", "quality_other",
                "not_applicable",
            ],
        },
        "ai_methods": {"type": "array", "items": {"type": "string"}},
        "model_family": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "cnn", "vision_transformer", "transformer_llm", "rnn",
                    "gan", "diffusion", "graph_nn", "classical_ml",
                    "radiomics", "foundation_model", "other",
                ],
            },
        },
        "imaging_modality": {"type": "array", "items": {"type": "string"}},
        "anatomy": {"type": "array", "items": {"type": "string"}},
        "disease": {"type": "array", "items": {"type": "string"}},
        "data_regime": {
            "type": "string",
            "enum": ["single_center", "multi_center", "public_dataset",
                     "mixed", "unknown"],
        },
        "sample_size": {"type": ["integer", "null"]},
        "external_validation": {"type": "boolean"},
        "benchmarks_datasets": {"type": "array", "items": {"type": "string"}},
        "code_available": {"type": "boolean"},
        "data_available": {"type": "boolean"},
        "novelty_keywords": {"type": "array", "items": {"type": "string"}},
        "one_sentence_summary": {"type": "string"},
    },
    "required": [
        "is_ai_relevant", "clinical_task", "ai_methods", "model_family",
        "imaging_modality", "anatomy", "disease", "data_regime",
        "external_validation", "benchmarks_datasets", "code_available",
        "data_available", "novelty_keywords", "one_sentence_summary",
    ],
}

# Field names grouped by target type (for stable parquet dtypes).
_SCHEMA_FIELDS = list(EXTRACTION_SCHEMA["properties"].keys())


def _types_of(spec: dict) -> set[str]:
    t = spec.get("type")
    return set(t) if isinstance(t, list) else {t}


_ARRAY_FIELDS = {k for k, v in EXTRACTION_SCHEMA["properties"].items() if "array" in _types_of(v)}
_BOOL_FIELDS = {k for k, v in EXTRACTION_SCHEMA["properties"].items() if "boolean" in _types_of(v)}
_INT_FIELDS = {k for k, v in EXTRACTION_SCHEMA["properties"].items() if "integer" in _types_of(v)}


def _unwrap(v: object) -> object:
    """Collapse a stray list onto a scalar: [] -> None, [x, ...] -> x."""
    if isinstance(v, list):
        return v[0] if v else None
    return v


def _as_bool(v: object) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "yes", "1", "y"}:
            return True
        if s in {"false", "no", "0", "n"}:
            return False
    return None


def _coerce_to_schema(rec: object) -> dict:
    """Project one LLM result onto the schema fields with stable types.

    Drops hallucinated extra keys and forces every field to its schema type so
    pyarrow never sees a column that mixes types -- e.g. a reasoning model that
    returns ``[]`` for a boolean field. Array fields always serialize as lists;
    scalar fields unwrap stray lists and cast to bool / int / str (or None).
    """
    rec = rec if isinstance(rec, dict) else {}
    out: dict = {}
    for k in _SCHEMA_FIELDS:
        v = rec.get(k)
        if k in _ARRAY_FIELDS:
            v = [] if v is None else (v if isinstance(v, list) else [v])
        elif k in _BOOL_FIELDS:
            v = _as_bool(_unwrap(v))
        elif k in _INT_FIELDS:
            v = _unwrap(v)
            v = int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
        else:  # string fields
            v = _unwrap(v)
            v = None if v is None else (v if isinstance(v, str) else str(v))
        out[k] = v
    return out

SYSTEM = (
    "You are a meticulous neuroradiology + machine-learning research analyst. "
    "You read a paper's title, abstract, and author keywords and return a "
    "structured JSON record that conforms to the schema.\n"
    "Fill every field as completely as the text allows -- do NOT leave arrays "
    "empty or fields null when the paper gives you enough to infer a value:\n"
    "- clinical_task: assign the single best-fitting task; use 'not_applicable' "
    "only if the paper truly involves no clinical AI task.\n"
    "- ai_methods: list the concrete methods/architectures named or clearly "
    "implied (e.g. 'u-net', 'random forest', 'vision transformer', 'radiomics').\n"
    "- model_family: map those methods onto the allowed categories (e.g. a "
    "U-Net/ResNet -> 'cnn'; a ViT -> 'vision_transformer'; GPT/BERT/any LLM -> "
    "'transformer_llm'; handcrafted-feature pipelines -> 'radiomics' and/or "
    "'classical_ml'). Include every family that applies; use 'other' only if a "
    "model is clearly used but none fit.\n"
    "- data_regime: infer from the described cohort (single_center, "
    "multi_center, public_dataset, mixed); use 'unknown' only when there is no "
    "signal.\n"
    "- benchmarks_datasets: list any named datasets/benchmarks (public ones like "
    "'brats', 'adni', or described in-house/institutional cohorts).\n"
    "- novelty_keywords: 3-6 short lowercase phrases capturing the paper's "
    "specific contribution or novelty.\n"
    "Use lowercase short canonical terms (e.g. 'mri', 'ct', 'brain', 'stroke', "
    "'u-net', 'vision transformer'). "
    "Be conservative ONLY for the verifiable yes/no claims: set "
    "external_validation, code_available, and data_available to true only when "
    "the text explicitly supports them. "
    "If the paper is not about AI/ML/data-driven methods at all, set "
    "is_ai_relevant=false."
)


def _build_prompt(row: dict) -> tuple[str, str]:
    kws = ", ".join(row.get("author_keywords") or []) or "(none)"
    user = (
        f"TITLE: {row.get('title')}\n\n"
        f"ABSTRACT: {row.get('abstract') or '(no abstract available)'}\n\n"
        f"AUTHOR KEYWORDS: {kws}\n\n"
        "Return the JSON record."
    )
    return SYSTEM, user


def extract_papers(
    papers: pd.DataFrame, cfg: Config = CONFIG, *, save: bool = True
) -> pd.DataFrame:
    """Run LLM extraction over every paper; return papers + extracted columns."""
    llm = LLMClient(cfg)
    items = papers.to_dict("records")
    results = llm.map_json(items, _build_prompt, EXTRACTION_SCHEMA)
    results = [_coerce_to_schema(r) for r in results]

    ext = pd.DataFrame(results)
    ext.insert(0, "eid", papers["eid"].values)
    merged = papers.merge(ext, on="eid", how="left", suffixes=("", "_llm"))

    if save:
        ext.to_parquet(cfg.tables_dir / "extractions.parquet", index=False)
        merged.to_parquet(cfg.tables_dir / "papers_extracted.parquet", index=False)
        print(f"Saved extractions for {len(ext)} papers.")
    return merged
