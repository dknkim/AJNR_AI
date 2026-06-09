"""Visualization helpers. Each returns the saved PNG path."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CONFIG, Config

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 200, "font.size": 10})


def _save(fig, name: str, cfg: Config) -> Path:
    path = cfg.fig_dir / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_volume_impact(ppy: pd.DataFrame, cfg: Config = CONFIG) -> Path:
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(ppy["year"], ppy["n_papers"], color="#4C72B0", alpha=0.8, label="Papers")
    ax1.set_xlabel("Year"); ax1.set_ylabel("AI papers", color="#4C72B0")
    ax2 = ax1.twinx()
    ax2.plot(ppy["year"], ppy["total_citations"], color="#C44E52", marker="o", label="Citations")
    ax2.set_ylabel("Total citations", color="#C44E52")
    ax1.set_title("AJNR AI papers and citations by year")
    return _save(fig, "volume_impact_by_year.png", cfg)


def plot_subfield_stream(matrix: pd.DataFrame, cfg: Config = CONFIG) -> Path:
    m = matrix.set_index("year") if "year" in matrix.columns else matrix
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.stackplot(m.index, m.T.values, labels=m.columns)
    ax.set_xlabel("Year"); ax.set_ylabel("Papers")
    ax.set_title("AI subfield volume over time (stacked)")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    return _save(fig, "subfield_stream.png", cfg)


def plot_subfield_heatmap(share: pd.DataFrame, cfg: Config = CONFIG) -> Path:
    m = share.set_index("year") if "year" in share.columns else share
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(m.T.values, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(m.index))); ax.set_xticklabels(m.index, rotation=45)
    ax.set_yticks(range(len(m.columns))); ax.set_yticklabels(m.columns, fontsize=8)
    ax.set_title("Subfield share of yearly AI output")
    fig.colorbar(im, ax=ax, label="share")
    return _save(fig, "subfield_heatmap.png", cfg)


def plot_growth(growth: pd.DataFrame, cfg: Config = CONFIG) -> Path:
    g = growth.sort_values("recent_minus_early")
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#55A868" if v >= 0 else "#C44E52" for v in g["recent_minus_early"]]
    ax.barh(g["subfield"], g["recent_minus_early"], color=colors)
    ax.set_xlabel("Recent-half minus early-half paper count")
    ax.set_title("Subfield growth (recent vs early)")
    return _save(fig, "subfield_growth.png", cfg)


def plot_hype_vs_impact(hvi: pd.DataFrame, cfg: Config = CONFIG) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(hvi["n_papers"], hvi["mean_citations"], s=60, color="#4C72B0")
    for _, r in hvi.iterrows():
        ax.annotate(r["subfield"], (r["n_papers"], r["mean_citations"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axvline(hvi["n_papers"].median(), ls="--", c="gray", lw=0.8)
    ax.axhline(hvi["mean_citations"].median(), ls="--", c="gray", lw=0.8)
    ax.set_xlabel("Volume (papers)"); ax.set_ylabel("Mean citations / paper")
    ax.set_title("Subfield volume vs impact")
    return _save(fig, "hype_vs_impact.png", cfg)


def plot_topic_scatter(papers_topics: pd.DataFrame, cfg: Config = CONFIG) -> Path:
    fig, ax = plt.subplots(figsize=(10, 8))
    topics = sorted(papers_topics["topic"].unique())
    cmap = plt.cm.get_cmap("tab20", max(len(topics), 1))
    for i, t in enumerate(topics):
        sub = papers_topics[papers_topics["topic"] == t]
        label = sub["topic_label"].iloc[0] if "topic_label" in sub and len(sub) else f"Topic {t}"
        ax.scatter(sub["umap_x"], sub["umap_y"], s=30, color=cmap(i),
                   label=f"{label} ({len(sub)})", alpha=0.7)
    ax.set_title("Discovered topic map (UMAP)")
    ax.legend(fontsize=6, loc="best", ncol=2)
    ax.set_xticks([]); ax.set_yticks([])
    return _save(fig, "topic_map.png", cfg)


def plot_country_trends(country: pd.DataFrame, top_n: int = 8, cfg: Config = CONFIG) -> Path:
    top = country.groupby("country")["n_papers"].sum().nlargest(top_n).index
    sub = country[country["country"].isin(top)]
    piv = sub.pivot_table(index="year", columns="country", values="n_papers", aggfunc="sum").fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    for c in piv.columns:
        ax.plot(piv.index, piv[c], marker="o", label=c)
    ax.set_xlabel("Year"); ax.set_ylabel("Papers"); ax.set_title("AI papers by country")
    ax.legend(fontsize=7)
    return _save(fig, "country_trends.png", cfg)


def plot_coauthor_network(G, cent: pd.DataFrame, top_n: int = 60, cfg: Config = CONFIG) -> Path:
    import networkx as nx

    if G.number_of_nodes() == 0:
        return cfg.fig_dir / "coauthor_network.png"
    keep = set(cent.head(top_n)["auid"]) if not cent.empty else set(G.nodes)
    H = G.subgraph(keep)
    fig, ax = plt.subplots(figsize=(11, 11))
    pos = nx.spring_layout(H, seed=42, k=0.3)
    sizes = [H.nodes[n].get("papers", 1) * 40 for n in H.nodes]
    nx.draw_networkx_nodes(H, pos, node_size=sizes, node_color="#4C72B0", alpha=0.8, ax=ax)
    nx.draw_networkx_edges(H, pos, alpha=0.2, ax=ax)
    labels = {n: H.nodes[n].get("name", "") for n in H.nodes if H.nodes[n].get("papers", 0) >= 2}
    nx.draw_networkx_labels(H, pos, labels, font_size=6, ax=ax)
    ax.set_title("AJNR AI co-authorship network (top authors)")
    ax.axis("off")
    return _save(fig, "coauthor_network.png", cfg)
