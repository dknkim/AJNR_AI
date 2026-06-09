"""AJNR AI publication-trends analysis pipeline.

A modular pipeline for analyzing long-term trends in AI-related publications
in the American Journal of Neuroradiology (AJNR) using Scopus metadata and a
locally hosted Hugging Face LLM (served via vLLM) plus local embeddings.

Stage modules (run in this order, each is independently cacheable):

    1. acquire   -- build the candidate corpus from the Scopus Search API
    2. enrich    -- pull full records via the Abstract Retrieval API
    3. normalize -- flatten nested records into tidy tables
    4. extract   -- per-paper structured extraction with the local LLM
    5. taxonomy  -- assign each paper to AI subfields (rules + LLM zero-shot)
    6. embed     -- local scientific-text embeddings
    7. cluster   -- unsupervised topic discovery + LLM cluster labels
    8. trends    -- time-series, emergence/burst, citation-weighted impact
    9. influence -- author / institution / country influence + networks
   10. report    -- LLM-written annual narrative + figures

The driver notebook ``run_pipeline.ipynb`` orchestrates these in order.
"""

from . import config

__all__ = ["config"]
__version__ = "0.1.0"
