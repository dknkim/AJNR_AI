"""Stage 9 -- author / institution / country influence + collaboration network.

Single-journal corpus, so "venue" influence is out; instead we rank the people
and places driving AJNR's AI output and chart their footprint across subfields.

  author_influence   : productivity + citation impact per author (auid)
  affil_influence    : same per institution (afid)
  country_trends     : papers per country per year
  subfield_by_affil  : which institutions lead which subfields
  coauthor_network   : co-authorship graph + centrality (networkx)
"""

from __future__ import annotations

import itertools

import pandas as pd

from .config import CONFIG, Config


def author_influence(paper_authors: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    g = (
        paper_authors.dropna(subset=["auid"])
        .groupby(["auid", "name"])
        .agg(n_papers=("eid", "nunique"),
             total_citations=("citedby_count", "sum"),
             mean_citations=("citedby_count", "mean"),
             first_year=("year", "min"),
             last_year=("year", "max"))
        .reset_index()
    )
    # h-index-like rank by citations, tie-broken by productivity
    return g.sort_values(["total_citations", "n_papers"], ascending=False).head(top_n)


def affil_influence(paper_affils: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    g = (
        paper_affils.dropna(subset=["afid"])
        .groupby(["afid", "name", "country"])
        .agg(n_papers=("eid", "nunique"),
             total_citations=("citedby_count", "sum"),
             mean_citations=("citedby_count", "mean"))
        .reset_index()
    )
    return g.sort_values(["n_papers", "total_citations"], ascending=False).head(top_n)


def country_trends(paper_affils: pd.DataFrame) -> pd.DataFrame:
    return (
        paper_affils.dropna(subset=["country"])
        .drop_duplicates(["eid", "country"])      # count a country once per paper
        .groupby(["year", "country"])
        .size()
        .reset_index(name="n_papers")
    )


def subfield_by_affil(
    paper_affils: pd.DataFrame, subfields_long: pd.DataFrame, top_n: int = 20
) -> pd.DataFrame:
    merged = paper_affils.merge(
        subfields_long[["eid", "subfield"]], on="eid", how="inner"
    )
    g = (
        merged.dropna(subset=["afid"])
        .groupby(["name", "subfield"])
        .size()
        .reset_index(name="n_papers")
    )
    return g.sort_values("n_papers", ascending=False).head(top_n)


def coauthor_network(paper_authors: pd.DataFrame, min_papers: int = 1):
    """Build a co-authorship graph; return (graph, centrality DataFrame)."""
    import networkx as nx

    G = nx.Graph()
    counts = paper_authors.dropna(subset=["auid"]).groupby("auid")["eid"].nunique()
    for eid, grp in paper_authors.dropna(subset=["auid"]).groupby("eid"):
        auids = grp["auid"].unique().tolist()
        names = dict(zip(grp["auid"], grp["name"]))
        for a in auids:
            if counts.get(a, 0) >= min_papers:
                G.add_node(a, name=names.get(a), papers=int(counts.get(a, 0)))
        for a, b in itertools.combinations([a for a in auids if counts.get(a, 0) >= min_papers], 2):
            if G.has_edge(a, b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a, b, weight=1)

    if G.number_of_nodes() == 0:
        return G, pd.DataFrame()

    deg = nx.degree_centrality(G)
    btw = nx.betweenness_centrality(G, weight="weight")
    cent = pd.DataFrame(
        [
            {
                "auid": n,
                "name": G.nodes[n].get("name"),
                "papers": G.nodes[n].get("papers"),
                "degree_centrality": deg.get(n, 0),
                "betweenness": btw.get(n, 0),
            }
            for n in G.nodes
        ]
    ).sort_values("betweenness", ascending=False)
    return G, cent


def compute_all(tables: dict[str, pd.DataFrame], subfields_long: pd.DataFrame,
                cfg: Config = CONFIG, *, save: bool = True) -> dict:
    pa, paf = tables["paper_authors"], tables["paper_affils"]
    out = {
        "author_influence": author_influence(pa),
        "affil_influence": affil_influence(paf),
        "country_trends": country_trends(paf),
        "subfield_by_affil": subfield_by_affil(paf, subfields_long),
    }
    G, cent = coauthor_network(pa)
    out["coauthor_centrality"] = cent
    if save:
        for name, df in out.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(cfg.tables_dir / f"influence_{name}.csv", index=False)
        print("Saved influence tables:", ", ".join(out))
    return {**out, "_graph": G}
