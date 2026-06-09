"""Central configuration for the AJNR AI-trends pipeline.

Secrets are read from environment variables so they never live in the
notebook. For convenience during local development the Scopus API key falls
back to the value already used in ``test.ipynb`` -- replace this by exporting
``SCOPUS_API_KEY`` in your shell or a ``.env`` file.

    export SCOPUS_API_KEY=...

The FULL Abstract Retrieval view (full abstract, affiliations, references) is
unlocked by your institution's IP-based Scopus entitlement, so no institutional
token is required as long as you run from that network. (Note: COMPLETE is a
Search-API view and is rejected by the Abstract Retrieval endpoint.)

Everything path-related is rooted at ``DATA_DIR`` so the whole run is
reproducible and cacheable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent                      # .../AJNR
DATA_DIR = Path(os.getenv("AJNR_DATA_DIR", PROJECT_DIR / "data"))

RAW_DIR = DATA_DIR / "raw"          # raw API JSON payloads
CACHE_DIR = DATA_DIR / "cache"      # sqlite caches (abstracts, llm calls)
TABLES_DIR = DATA_DIR / "tables"    # tidy parquet/csv tables
FIG_DIR = DATA_DIR / "figures"      # generated plots
REPORT_DIR = DATA_DIR / "reports"   # markdown / html reports

for _d in (RAW_DIR, CACHE_DIR, TABLES_DIR, FIG_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Scopus
# --------------------------------------------------------------------------- #
@dataclass
class ScopusConfig:
    api_key: str = field(
        default_factory=lambda: os.getenv(
            "SCOPUS_API_KEY", "57581543dea4d979a748eb7941383971"
        )
    )
    search_url: str = "https://api.elsevier.com/content/search/scopus"
    abstract_url: str = "https://api.elsevier.com/content/abstract/scopus_id"
    abstract_doi_url: str = "https://api.elsevier.com/content/abstract/doi"
    # Politeness: Scopus weekly quota is large but per-second throttling helps.
    request_delay_s: float = 0.15
    max_retries: int = 4
    timeout_s: int = 30

    def headers(self) -> dict[str, str]:
        return {"X-ELS-APIKey": self.api_key, "Accept": "application/json"}


# --------------------------------------------------------------------------- #
# Corpus definition
# --------------------------------------------------------------------------- #
@dataclass
class CorpusConfig:
    # AJNR identifiers (ISSN print / e-ISSN). ISSN() matches both forms.
    ajnr_issn: str = "0195-6108"
    journal_name: str = "American Journal of Neuroradiology"
    years_back: int = 10
    # Server-side AI filter. Kept broad on the API side; precision is improved
    # later by full-abstract keyword + LLM relevance confirmation.
    ai_terms: tuple[str, ...] = (
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "DL",
        "AI",
        "neural network",
        "convolutional neural network",
        "transformer",
        "large language model",
        "foundation model",
        "generative",
        "computer-aided",
        "automated segmentation",
        "self-supervised",
        "natural language processing",
        "chatgpt",
        "gpt-4",
    )


# --------------------------------------------------------------------------- #
# Local LLM (vLLM, OpenAI-compatible endpoint)
# --------------------------------------------------------------------------- #
@dataclass
class LLMConfig:
    """Config for the local LLM served by vLLM.

    Recommended launch on 4x48GB (see README):

        vllm serve Qwen/Qwen2.5-72B-Instruct \\
            --tensor-parallel-size 4 --max-model-len 32768 \\
            --gpu-memory-utilization 0.92 --port 8000

    Qwen2.5-72B-Instruct is more than sufficient for structured extraction,
    cluster labeling, and summarization here -- the task is bounded by prompt
    design and JSON-schema adherence, not raw capability, so the marginal
    benefit of a larger model (Qwen3-235B-A22B-Instruct, etc.) is small. If you
    want stronger reasoning at the same memory budget, Qwen3-32B (dense) is a
    good drop-in. Set the model name below to whatever you serve.
    """

    base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
    )
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "EMPTY"))
    model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")
    )
    temperature: float = 0.0          # deterministic for extraction/labeling
    # Reasoning ("Thinking") models emit a long <think> trace before the answer,
    # which counts against max_tokens; bump LLM_MAX_TOKENS (e.g. 8192) for those
    # or the JSON answer gets truncated. 1536 is plenty for non-thinking models.
    max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "1536")))
    # The narrative report is a single long (~4k-token) generation; on a 72B
    # model over PCIe that can take several minutes, so keep this generous.
    timeout_s: int = field(default_factory=lambda: int(os.getenv("LLM_TIMEOUT_S", "600")))
    max_concurrency: int = 8          # parallel requests to vLLM
    # vLLM supports guided JSON decoding; turn off if your server lacks it.
    use_guided_json: bool = True


# --------------------------------------------------------------------------- #
# Local embeddings
# --------------------------------------------------------------------------- #
@dataclass
class EmbedConfig:
    """Local sentence/document embeddings for clustering.

    Defaults to a strong scientific-document model. Options:
      - "allenai/specter2_base"  : purpose-built for scientific-paper similarity
                                    (needs the proximity adapter; see embed.py)
      - "BAAI/bge-m3"            : strong general multilingual, easy via S-T
      - "Qwen/Qwen3-Embedding-4B": SOTA general embeddings (needs prompts)
    """

    model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "allenai/specter2_base"))
    device: str = field(default_factory=lambda: os.getenv("EMBED_DEVICE", "cuda"))
    batch_size: int = 32
    normalize: bool = True


@dataclass
class ClusterConfig:
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    umap_min_dist: float = 0.0
    umap_metric: str = "cosine"
    hdbscan_min_cluster_size: int = 5     # small corpus -> small clusters
    hdbscan_min_samples: int | None = None
    random_state: int = 42
    top_n_terms: int = 12                 # c-TF-IDF terms per cluster


# --------------------------------------------------------------------------- #
# AI subfield taxonomy (the user's named subfields + neuroradiology-relevant)
# --------------------------------------------------------------------------- #
# Each subfield: a human label + seed keywords for the rule-based pass. The LLM
# pass refines/confirms and can assign multiple labels per paper.
SUBFIELDS: dict[str, list[str]] = {
    "Large language models / NLP": [
        "large language model", "llm", "chatgpt", "gpt-4", "gpt-3", "bert",
        "natural language processing", "nlp", "report generation", "chatbot",
        "text", "prompt",
    ],
    "Generative / diffusion models": [
        "generative adversarial", "gan", "diffusion model", "synthesis",
        "image generation", "denoising diffusion", "variational autoencoder",
    ],
    "Multimodal AI": [
        "multimodal", "multi-modal", "vision-language", "image-text",
        "clinical-imaging fusion", "multimodal fusion",
    ],
    "Image segmentation": [
        "segmentation", "u-net", "unet", "nnu-net", "lesion segmentation",
        "tumor segmentation", "auto-segmentation",
    ],
    "Detection / classification (CADx)": [
        "detection", "classification", "computer-aided", "diagnosis",
        "triage", "screening", "lesion detection",
    ],
    "Image reconstruction / acceleration": [
        "reconstruction", "accelerated", "denoising", "super-resolution",
        "compressed sensing", "deep learning reconstruction", "undersampled",
    ],
    "Quantitative imaging": [
        "texture analysis", "quantitative imaging",
        "handcrafted features", "feature extraction",
    ],
    "Prognosis / outcome prediction": [
        "outcome prediction", "prognosis", "survival", "risk prediction",
        "treatment response", "recurrence",
    ],
    "Reinforcement learning": [
        "reinforcement learning", "q-learning", "policy gradient", "reward",
    ],
    "AI agents / autonomous systems": [
        "ai agent", "autonomous agent", "agentic", "tool use", "multi-agent",
    ],
    "Foundation models / self-supervised": [
        "foundation model", "self-supervised", "pretrained", "pretraining",
        "transfer learning", "contrastive learning",
    ],
    "Federated / privacy-preserving learning": [
        "federated learning", "privacy-preserving", "differential privacy",
    ],
    "Explainability / trustworthy AI": [
        "explainable", "interpretability", "saliency", "grad-cam",
        "uncertainty", "calibration", "trustworthy",
    ],
}


# --------------------------------------------------------------------------- #
# Bundled config object
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    scopus: ScopusConfig = field(default_factory=ScopusConfig)
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    subfields: dict[str, list[str]] = field(default_factory=lambda: SUBFIELDS)

    # path shortcuts
    raw_dir: Path = RAW_DIR
    cache_dir: Path = CACHE_DIR
    tables_dir: Path = TABLES_DIR
    fig_dir: Path = FIG_DIR
    report_dir: Path = REPORT_DIR


CONFIG = Config()
