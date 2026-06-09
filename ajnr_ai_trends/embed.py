"""Stage 6 -- local scientific-text embeddings.

Embeds title+abstract for every paper. Two backends:

  - SPECTER2 (default, "allenai/specter2_base"): citation-trained embeddings
    purpose-built for scientific-paper similarity/clustering. Needs the
    ``proximity`` adapter via the ``adapters`` library; we fall back gracefully.
  - any sentence-transformers model (e.g. "BAAI/bge-m3",
    "Qwen/Qwen3-Embedding-4B") if SPECTER2 deps are unavailable.

Embeddings are cached to .npy keyed by model + corpus hash.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .config import CONFIG, Config


def _doc_text(row: pd.Series) -> str:
    title = row.get("title") or ""
    abstract = row.get("abstract") or ""
    return f"{title}\n\n{abstract}".strip()


def _corpus_hash(texts: list[str], model: str) -> str:
    h = hashlib.sha256(model.encode())
    for t in texts:
        h.update(t.encode("utf-8", "ignore"))
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
def _embed_specter2(texts: list[str], cfg: Config) -> np.ndarray:
    """SPECTER2 via the adapters library (preferred for science papers)."""
    import torch
    from adapters import AutoAdapterModel
    from transformers import AutoTokenizer

    name = cfg.embed.model
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoAdapterModel.from_pretrained(name)
    model.load_adapter(
        "allenai/specter2", source="hf", load_as="proximity", set_active=True
    )
    model.to(cfg.embed.device).eval()

    vecs = []
    bs = cfg.embed.batch_size
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            inputs = tok(
                batch, padding=True, truncation=True, max_length=512,
                return_tensors="pt",
            ).to(cfg.embed.device)
            out = model(**inputs)
            # CLS token embedding
            vecs.append(out.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(vecs)


def _embed_sentence_transformers(texts: list[str], cfg: Config) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(cfg.embed.model, device=cfg.embed.device)
    return model.encode(
        texts,
        batch_size=cfg.embed.batch_size,
        normalize_embeddings=cfg.embed.normalize,
        show_progress_bar=True,
    )


def embed_papers(
    papers: pd.DataFrame, cfg: Config = CONFIG, *, save: bool = True
) -> np.ndarray:
    texts = [_doc_text(r) for _, r in papers.iterrows()]
    cache = cfg.cache_dir / f"emb_{_corpus_hash(texts, cfg.embed.model)}.npy"
    if cache.exists():
        print(f"Loading cached embeddings -> {cache.name}")
        return np.load(cache)

    name = cfg.embed.model.lower()
    if "specter2" in name:
        try:
            emb = _embed_specter2(texts, cfg)
        except Exception as exc:  # noqa: BLE001 - dependency/adapter fallback
            print(f"SPECTER2 path failed ({exc}); falling back to sentence-transformers.")
            emb = _embed_sentence_transformers(texts, cfg)
    else:
        emb = _embed_sentence_transformers(texts, cfg)

    emb = np.asarray(emb, dtype=np.float32)
    if cfg.embed.normalize:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.clip(norms, 1e-8, None)

    if save:
        np.save(cache, emb)
        print(f"Saved embeddings {emb.shape} -> {cache.name}")
    return emb
