"""Stage 7 -- unsupervised topic discovery + LLM cluster labels.

A BERTopic-style pipeline implemented directly so we control every step:

  embeddings --UMAP--> low-dim --HDBSCAN--> clusters
  per cluster: c-TF-IDF top terms  + representative papers
  LLM: turn (top terms + rep titles) into a human topic label + 1-line summary

This is the *data-driven* complement to the fixed taxonomy: it surfaces themes
we did not pre-specify (the "what new topics emerged" question). Cluster -1 is
HDBSCAN noise/outliers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from .config import CONFIG, Config
from .llm_client import LLMClient


def _reduce(emb: np.ndarray, cfg: Config) -> np.ndarray:
    import umap

    c = cfg.cluster
    n_comp = min(c.umap_n_components, max(2, emb.shape[0] - 2))
    reducer = umap.UMAP(
        n_neighbors=min(c.umap_n_neighbors, max(2, emb.shape[0] - 1)),
        n_components=n_comp,
        min_dist=c.umap_min_dist,
        metric=c.umap_metric,
        random_state=c.random_state,
    )
    return reducer.fit_transform(emb)


def _cluster(low: np.ndarray, cfg: Config) -> np.ndarray:
    import hdbscan

    c = cfg.cluster
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=c.hdbscan_min_cluster_size,
        min_samples=c.hdbscan_min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    return clusterer.fit_predict(low)


def _ctfidf_terms(papers: pd.DataFrame, labels: np.ndarray, cfg: Config) -> dict[int, list[str]]:
    """class-based TF-IDF: top distinctive terms per cluster."""
    docs = (papers["title"].fillna("") + ". " + papers["abstract"].fillna("")).tolist()
    df = pd.DataFrame({"doc": docs, "cluster": labels})
    grouped = df.groupby("cluster")["doc"].apply(" ".join)

    vec = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    counts = vec.fit_transform(grouped.values)            # (n_clusters, n_terms)
    words = np.array(vec.get_feature_names_out())

    tf = counts.toarray().astype(float)
    tf_sum = tf.sum(axis=1, keepdims=True)
    tf_norm = tf / np.clip(tf_sum, 1, None)
    n = tf.shape[0]
    doc_freq = (tf > 0).sum(axis=0)
    idf = np.log(1 + n / np.clip(doc_freq, 1, None))
    ctfidf = tf_norm * idf

    out = {}
    for i, cl in enumerate(grouped.index):
        top = np.argsort(ctfidf[i])[::-1][: cfg.cluster.top_n_terms]
        out[int(cl)] = words[top].tolist()
    return out


_LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["label", "summary"],
}


def _label_clusters(
    terms: dict[int, list[str]], reps: dict[int, list[str]], cfg: Config
) -> dict[int, dict]:
    llm = LLMClient(cfg)
    out = {}
    for cl, words in terms.items():
        if cl == -1:
            out[cl] = {"label": "Outliers / unclustered", "summary": ""}
            continue
        system = (
            "You name a research topic cluster of neuroradiology AI papers. "
            "Given distinctive terms and representative titles, return a short "
            "specific label (<=6 words) and a one-sentence summary. JSON only."
        )
        user = (
            f"DISTINCTIVE TERMS: {', '.join(words)}\n\n"
            f"REPRESENTATIVE TITLES:\n- " + "\n- ".join(reps.get(cl, [])) + "\n\n"
            "Return {\"label\": ..., \"summary\": ...}."
        )
        out[cl] = llm.json(system, user, _LABEL_SCHEMA)
    return out


def discover_topics(
    papers: pd.DataFrame,
    emb: np.ndarray,
    cfg: Config = CONFIG,
    *,
    label_with_llm: bool = True,
    save: bool = True,
) -> pd.DataFrame:
    low = _reduce(emb, cfg)
    labels = _cluster(low, cfg)

    papers = papers.copy()
    papers["topic"] = labels
    papers["umap_x"] = low[:, 0]
    papers["umap_y"] = low[:, 1] if low.shape[1] > 1 else 0.0

    terms = _ctfidf_terms(papers, labels, cfg)

    # representative titles = highest-cited members per cluster
    reps: dict[int, list[str]] = {}
    for cl in terms:
        sub = papers[papers["topic"] == cl].sort_values("citedby_count", ascending=False)
        reps[cl] = sub["title"].head(5).tolist()

    label_map = (
        _label_clusters(terms, reps, cfg)
        if label_with_llm
        else {cl: {"label": f"Topic {cl}", "summary": ", ".join(w[:5])}
              for cl, w in terms.items()}
    )

    papers["topic_label"] = papers["topic"].map(lambda c: label_map.get(c, {}).get("label"))

    topic_summary = pd.DataFrame(
        [
            {
                "topic": cl,
                "label": label_map.get(cl, {}).get("label"),
                "summary": label_map.get(cl, {}).get("summary"),
                "n_papers": int((labels == cl).sum()),
                "total_citations": int(papers.loc[papers["topic"] == cl, "citedby_count"].sum()),
                "top_terms": ", ".join(terms.get(cl, [])),
                "rep_titles": " | ".join(reps.get(cl, [])),
            }
            for cl in sorted(terms)
        ]
    ).sort_values("n_papers", ascending=False)

    if save:
        papers.to_parquet(cfg.tables_dir / "papers_topics.parquet", index=False)
        topic_summary.to_parquet(cfg.tables_dir / "topic_summary.parquet", index=False)
        topic_summary.to_csv(cfg.tables_dir / "topic_summary.csv", index=False)
        n_topics = len([c for c in terms if c != -1])
        print(f"Discovered {n_topics} topics (+outliers). Saved topic tables.")
    return topic_summary
