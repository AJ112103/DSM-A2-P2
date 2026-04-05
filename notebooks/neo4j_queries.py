#!/usr/bin/env python3
"""
CS-3510 Assignment 2 Part 2 — Neo4j GDS Queries (Q1–Q5)
Aamer Jalan, Ashoka University
"""

import os, json, time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from neo4j import GraphDatabase
from collections import defaultdict

# ── Setup ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.bbox": "tight",
})
sns.set_palette("colorblind")

FIGURES = os.path.join(os.path.dirname(__file__), "..", "report", "figures")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "Aamer@DSM"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS),
                               max_connection_lifetime=3600,
                               connection_acquisition_timeout=300)


def save_fig(name):
    path = os.path.join(FIGURES, name)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def run_query(query, timeout=600, **params):
    with driver.session() as s:
        result = s.run(query, parameters=params, timeout=timeout)
        return result.data()


def drop_graph_if_exists(name):
    try:
        run_query(f"CALL gds.graph.drop('{name}') YIELD graphName RETURN graphName")
        print(f"  Dropped graph '{name}'")
    except Exception:
        pass


# =============================================================================
# Q1: PageRank on User→Business Review Graph (7 marks)
# =============================================================================
def run_q1():
    print("\n" + "=" * 70)
    print("Q1: PageRank on User→Business Review Graph")
    print("=" * 70)

    # Use Cypher projection to project User→Business edges through Review nodes
    print("  Step 1: Projecting graph via Cypher projection...")
    drop_graph_if_exists("user-biz-pr")

    result = run_query("""
        CALL gds.graph.project.cypher(
            'user-biz-pr',
            'MATCH (n) WHERE n:User OR n:Business RETURN id(n) AS id, labels(n) AS labels',
            'MATCH (u:User)-[:WROTE]->(r:Review)-[:REVIEWS]->(b:Business)
             RETURN id(u) AS source, id(b) AS target, r.stars AS weight',
            {readConcurrency: 4}
        ) YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
    """, timeout=600)
    print(f"  Projected: {result[0]['nodeCount']:,} nodes, {result[0]['relationshipCount']:,} edges")

    # Step 2: Run PageRank
    print("  Step 2: Running PageRank (20 iterations)...")
    pr_results = run_query("""
        CALL gds.pageRank.stream('user-biz-pr', {
            maxIterations: 20,
            relationshipWeightProperty: 'weight'
        })
        YIELD nodeId, score
        WITH gds.util.asNode(nodeId) AS node, score
        WHERE 'Business' IN labels(node)
        RETURN node.business_id AS business_id,
               node.name AS name,
               node.review_count AS review_count,
               node.stars AS avg_stars,
               score AS pageRank
        ORDER BY score DESC
        LIMIT 100
    """, timeout=600)

    df = pd.DataFrame(pr_results)
    top15 = df.head(15).copy()
    print("\n  Top 15 Businesses by PageRank:")
    print(top15[["name", "pageRank", "review_count", "avg_stars"]].to_string(index=False))

    # Spearman correlations
    df["pr_rank"] = range(1, len(df) + 1)
    df["rc_rank"] = df["review_count"].rank(ascending=False, method="min")
    df["star_rank"] = df["avg_stars"].rank(ascending=False, method="min")

    rho_rc, p_rc = stats.spearmanr(df["pr_rank"], df["rc_rank"])
    rho_star, p_star = stats.spearmanr(df["pr_rank"], df["star_rank"])
    print(f"\n  Spearman (PageRank vs Review Count): rho={rho_rc:.4f}, p={p_rc:.4e}")
    print(f"  Spearman (PageRank vs Avg Stars): rho={rho_star:.4f}, p={p_star:.4e}")

    # Identify divergent businesses
    top15["rc_rank_in15"] = top15["review_count"].rank(ascending=False, method="min").astype(int)
    top15["pr_rank_in15"] = range(1, 16)
    top15["rank_diff"] = (top15["rc_rank_in15"] - top15["pr_rank_in15"]).astype(int)
    divergent = top15[top15["rank_diff"].abs() >= 5]
    if len(divergent) > 0:
        print("\n  Divergent businesses (rank diff ≥5):")
        print(divergent[["name", "pageRank", "review_count", "rank_diff"]].to_string(index=False))

    top15.to_csv(os.path.join(DATA, "q1_pagerank_top15.csv"), index=False)
    df.to_csv(os.path.join(DATA, "q1_pagerank_top100.csv"), index=False)

    # Visualizations
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["review_count"], df["pageRank"], alpha=0.6, s=30)
    for _, row in top15.head(5).iterrows():
        ax.annotate(row["name"][:20], (row["review_count"], row["pageRank"]),
                    fontsize=7, alpha=0.8, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Review Count")
    ax.set_ylabel("PageRank Score")
    ax.set_title(f"PageRank vs Review Count (Spearman rho={rho_rc:.3f})")
    ax.grid(alpha=0.3)
    save_fig("q1_pagerank_vs_reviews.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["avg_stars"], df["pageRank"], alpha=0.6, s=30)
    ax.set_xlabel("Average Star Rating")
    ax.set_ylabel("PageRank Score")
    ax.set_title(f"PageRank vs Avg Stars (Spearman rho={rho_star:.3f})")
    ax.grid(alpha=0.3)
    save_fig("q1_pagerank_vs_stars.png")

    fig, ax1 = plt.subplots(figsize=(12, 5))
    x = np.arange(15)
    names = [n[:25] for n in top15["name"]]
    ax1.bar(x, top15["pageRank"], color=sns.color_palette("colorblind")[0], alpha=0.8)
    ax1.set_ylabel("PageRank Score", color=sns.color_palette("colorblind")[0])
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(x, top15["review_count"], "o-", color=sns.color_palette("colorblind")[1], linewidth=2)
    ax2.set_ylabel("Review Count", color=sns.color_palette("colorblind")[1])
    ax1.set_title("Top 15 Businesses by PageRank")
    plt.tight_layout()
    save_fig("q1_pagerank_top15_bar.png")

    # Export user PageRank scores for predictive model
    print("  Exporting user PageRank scores...")
    user_pr = run_query("""
        CALL gds.pageRank.stream('user-biz-pr', {
            maxIterations: 20,
            relationshipWeightProperty: 'weight'
        })
        YIELD nodeId, score
        WITH gds.util.asNode(nodeId) AS node, score
        WHERE 'User' IN labels(node) AND score > 0.15
        RETURN node.user_id AS user_id, score AS pagerank
        ORDER BY score DESC
    """, timeout=600)
    pd.DataFrame(user_pr).to_csv(os.path.join(DATA, "user_pagerank.csv"), index=False)
    print(f"  Exported {len(user_pr)} user PageRank scores")

    drop_graph_if_exists("user-biz-pr")
    return df


# =============================================================================
# Q2: Louvain Community Detection on FRIENDS_WITH (7 marks)
# =============================================================================
def run_q2():
    print("\n" + "=" * 70)
    print("Q2: Louvain Community Detection on FRIENDS_WITH")
    print("=" * 70)

    print("  Step 1: Projecting FRIENDS_WITH graph...")
    drop_graph_if_exists("friends-graph")
    result = run_query("""
        CALL gds.graph.project(
            'friends-graph', 'User',
            {FRIENDS_WITH: {orientation: 'UNDIRECTED'}}
        ) YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
    """, timeout=300)
    print(f"  Projected: {result[0]['nodeCount']:,} nodes, {result[0]['relationshipCount']:,} edges")

    print("  Step 2: Running Louvain...")
    louvain_result = run_query("""
        CALL gds.louvain.write('friends-graph', {
            writeProperty: 'communityId'
        })
        YIELD communityCount, modularity
        RETURN communityCount, modularity
    """, timeout=600)
    print(f"  Communities: {louvain_result[0]['communityCount']}, Modularity: {louvain_result[0]['modularity']:.4f}")

    print("  Step 3: Analyzing communities with ≥25 members...")
    community_sizes = run_query("""
        MATCH (u:User)
        WHERE u.communityId IS NOT NULL
        WITH u.communityId AS cid, count(u) AS size
        WHERE size >= 25
        RETURN cid, size ORDER BY size DESC
    """)
    print(f"  {len(community_sizes)} communities with ≥25 members")

    rows = []
    for i, cd in enumerate(community_sizes[:40]):
        cid, size = cd["cid"], cd["size"]

        state_data = run_query("""
            MATCH (u:User {communityId: $cid})-[:WROTE]->(r:Review)-[:REVIEWS]->(b:Business)
            WITH b.state AS state, count(r) AS rc
            RETURN state, rc ORDER BY rc DESC
        """, timeout=120, cid=cid)

        if not state_data:
            continue

        total = sum(s["rc"] for s in state_data)
        geo_conc = state_data[0]["rc"] / total if total > 0 else 0
        top3_states = "; ".join(f"{s['state']} ({s['rc']})" for s in state_data[:3])

        cat_data = run_query("""
            MATCH (u:User {communityId: $cid})-[:WROTE]->(:Review)-[:REVIEWS]->(b:Business)
                  -[:IN_CATEGORY]->(c:Category)
            WITH c.name AS cat, count(*) AS rc
            RETURN cat, rc ORDER BY rc DESC LIMIT 3
        """, timeout=120, cid=cid)
        top3_cats = "; ".join(f"{c['cat']} ({c['rc']})" for c in cat_data)

        rows.append({
            "community_id": cid, "size": size, "total_reviews": total,
            "top3_states": top3_states, "top3_categories": top3_cats,
            "geo_concentration": round(geo_conc, 4),
            "most_common_state": state_data[0]["state"]
        })
        if (i + 1) % 10 == 0:
            print(f"    {i + 1} communities processed...")

    df = pd.DataFrame(rows).sort_values("geo_concentration", ascending=False)
    print(f"\n  Most concentrated:")
    print(df.head(5)[["community_id", "size", "geo_concentration", "most_common_state"]].to_string(index=False))
    print(f"\n  Least concentrated:")
    print(df.tail(5)[["community_id", "size", "geo_concentration", "most_common_state"]].to_string(index=False))

    df.to_csv(os.path.join(DATA, "q2_louvain_communities.csv"), index=False)

    # Viz 1: Community sizes
    fig, ax = plt.subplots(figsize=(12, 5))
    top20 = df.nlargest(20, "size")
    ax.bar(range(len(top20)), top20["size"], color=sns.color_palette("colorblind")[0])
    ax.set_xlabel("Community Rank")
    ax.set_ylabel("Size")
    ax.set_title("Top 20 Communities by Size")
    plt.tight_layout()
    save_fig("q2_community_sizes.png")

    # Viz 2: Geo concentration
    fig, ax = plt.subplots(figsize=(12, 5))
    show = pd.concat([df.head(10), df.tail(10)])
    colors = ["#2ecc71"] * min(10, len(df.head(10))) + ["#e74c3c"] * min(10, len(df.tail(10)))
    ax.barh(range(len(show)), show["geo_concentration"], color=colors[:len(show)])
    labels = [f"C{r['community_id']} (n={r['size']}, {r['most_common_state']})" for _, r in show.iterrows()]
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Geographic Concentration Index")
    ax.set_title("Most vs Least Geographically Concentrated Communities")
    ax.invert_yaxis()
    plt.tight_layout()
    save_fig("q2_geo_concentration.png")

    # Export community sizes
    comm_sizes_map = {r["community_id"]: r["size"] for _, r in df.iterrows()}
    pd.DataFrame(list(comm_sizes_map.items()), columns=["community_id", "size"]).to_csv(
        os.path.join(DATA, "community_sizes.csv"), index=False)

    return df


# =============================================================================
# Q3: Node Similarity (Jaccard) for Market Saturation (7 marks)
# =============================================================================
def run_q3():
    print("\n" + "=" * 70)
    print("Q3: Node Similarity (Jaccard) for Market Saturation")
    print("=" * 70)

    # Use Cypher projection: Business→User (through Review nodes)
    print("  Step 1: Projecting Business-User graph...")
    drop_graph_if_exists("biz-reviewer")

    result = run_query("""
        CALL gds.graph.project.cypher(
            'biz-reviewer',
            'MATCH (n) WHERE n:Business OR n:User RETURN id(n) AS id, labels(n) AS labels',
            'MATCH (b:Business)<-[:REVIEWS]-(r:Review)<-[:WROTE]-(u:User)
             RETURN id(b) AS source, id(u) AS target',
            {readConcurrency: 4}
        ) YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
    """, timeout=600)
    print(f"  Projected: {result[0]['nodeCount']:,} nodes, {result[0]['relationshipCount']:,} edges")

    # Run Node Similarity
    print("  Step 2: Running Node Similarity...")
    sim_results = run_query("""
        CALL gds.nodeSimilarity.stream('biz-reviewer', {
            degreeCutoff: 5,
            topK: 10,
            similarityCutoff: 0.01,
            concurrency: 4
        })
        YIELD node1, node2, similarity
        WITH gds.util.asNode(node1) AS b1, gds.util.asNode(node2) AS b2, similarity
        WHERE 'Business' IN labels(b1) AND 'Business' IN labels(b2)
        RETURN b1.business_id AS biz1, b1.city AS city1,
               b2.business_id AS biz2, b2.city AS city2,
               similarity
    """, timeout=600)
    print(f"  Got {len(sim_results):,} similarity pairs")

    drop_graph_if_exists("biz-reviewer")

    # Get business categories
    print("  Getting business categories...")
    cat_data = run_query("""
        MATCH (b:Business)-[:IN_CATEGORY]->(c:Category)
        RETURN b.business_id AS bid, collect(c.name) AS cats
    """, timeout=120)
    biz_cats = {r["bid"]: r["cats"] for r in cat_data}

    # Filter to intra-city, intra-category pairs
    print("  Computing intra-category similarity per city...")
    city_cat_sims = defaultdict(list)
    city_cat_biz = defaultdict(set)

    for row in sim_results:
        if row["city1"] != row["city2"]:
            continue
        city = row["city1"]
        cats1 = set(biz_cats.get(row["biz1"], []))
        cats2 = set(biz_cats.get(row["biz2"], []))
        for cat in cats1 & cats2:
            city_cat_sims[(city, cat)].append(row["similarity"])
            city_cat_biz[(city, cat)].update([row["biz1"], row["biz2"]])

    rows = []
    for (city, cat), sims in city_cat_sims.items():
        n_biz = len(city_cat_biz[(city, cat)])
        if n_biz >= 5:
            rows.append({
                "city": city, "category": cat, "n_businesses": n_biz,
                "mean_jaccard": round(np.mean(sims), 4), "n_pairs": len(sims)
            })

    df = pd.DataFrame(rows).sort_values("mean_jaccard", ascending=False)

    top5_sat = df.head(5)
    top5_frag = df.tail(5)
    print("\n  Top 5 Most Saturated:")
    print(top5_sat.to_string(index=False))
    print("\n  Top 5 Most Fragmented:")
    print(top5_frag.to_string(index=False))

    # Get star stats for comparison
    combo_stats = []
    for _, row in pd.concat([top5_sat, top5_frag]).iterrows():
        stat = run_query("""
            MATCH (b:Business {city: $city})-[:IN_CATEGORY]->(c:Category {name: $cat})
            RETURN avg(b.stars) AS ms, stDev(b.stars) AS ss, avg(b.review_count) AS mrc, count(b) AS n
        """, city=row["city"], cat=row["category"])
        if stat and stat[0]["ms"] is not None:
            combo_stats.append({
                "city": row["city"], "category": row["category"],
                "mean_jaccard": row["mean_jaccard"],
                "mean_stars": round(stat[0]["ms"], 3),
                "std_stars": round(stat[0]["ss"] or 0, 3),
                "mean_review_count": round(stat[0]["mrc"], 1),
                "type": "Saturated" if row["mean_jaccard"] >= top5_sat["mean_jaccard"].min() else "Fragmented"
            })
    stats_df = pd.DataFrame(combo_stats)
    print("\n  Rating comparison:")
    print(stats_df.to_string(index=False))

    df.to_csv(os.path.join(DATA, "q3_node_similarity.csv"), index=False)
    stats_df.to_csv(os.path.join(DATA, "q3_saturated_vs_fragmented.csv"), index=False)

    # Viz 1
    fig, ax = plt.subplots(figsize=(10, 6))
    highlight = pd.concat([top5_sat, top5_frag])
    highlight = highlight.copy()
    highlight["label"] = highlight["city"] + "\n" + highlight["category"]
    n_sat = len(top5_sat)
    n_frag = len(top5_frag)
    ax.barh(range(len(highlight)), highlight["mean_jaccard"],
            color=["#2ecc71"] * n_sat + ["#e74c3c"] * n_frag)
    ax.set_yticks(range(len(highlight)))
    ax.set_yticklabels(highlight["label"], fontsize=8)
    ax.set_xlabel("Mean Intra-Category Jaccard Similarity")
    ax.set_title("Most Saturated vs Most Fragmented Markets")
    ax.invert_yaxis()
    plt.tight_layout()
    save_fig("q3_saturation_comparison.png")

    # Viz 2
    if len(stats_df) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, metric, title in zip(axes,
            ["mean_stars", "std_stars", "mean_review_count"],
            ["Mean Stars", "Std Dev Stars", "Mean Review Count"]):
            sat_vals = stats_df[stats_df["type"] == "Saturated"][metric]
            frag_vals = stats_df[stats_df["type"] == "Fragmented"][metric]
            ax.bar([0, 1], [sat_vals.mean() if len(sat_vals) > 0 else 0,
                            frag_vals.mean() if len(frag_vals) > 0 else 0],
                   color=[sns.color_palette("colorblind")[0], sns.color_palette("colorblind")[1]],
                   tick_label=["Saturated", "Fragmented"])
            ax.set_title(title)
        plt.suptitle("Saturated vs Fragmented Market Metrics", fontsize=13)
        plt.tight_layout()
        save_fig("q3_market_comparison.png")

    # Export business Jaccard for predictive model
    biz_jaccard = defaultdict(list)
    for row in sim_results:
        biz_jaccard[row["biz1"]].append(row["similarity"])
        biz_jaccard[row["biz2"]].append(row["similarity"])
    pd.DataFrame([{"business_id": b, "mean_jaccard": np.mean(s)} for b, s in biz_jaccard.items()]).to_csv(
        os.path.join(DATA, "business_jaccard.csv"), index=False)

    return df


# =============================================================================
# Q4: Betweenness Centrality vs Degree Centrality (7 marks)
# =============================================================================
def run_q4():
    print("\n" + "=" * 70)
    print("Q4: Betweenness Centrality vs Degree Centrality")
    print("=" * 70)

    # Check/create friends graph
    graphs = run_query("CALL gds.graph.list() YIELD graphName RETURN graphName")
    if not any(g["graphName"] == "friends-graph" for g in graphs):
        print("  Re-projecting friends graph...")
        run_query("""
            CALL gds.graph.project('friends-graph', 'User',
                {FRIENDS_WITH: {orientation: 'UNDIRECTED'}})
            YIELD graphName RETURN graphName
        """, timeout=300)

    print("  Step 1: Running Betweenness Centrality (sampled)...")
    bc_results = run_query("""
        CALL gds.betweenness.stream('friends-graph', {samplingSize: 500, concurrency: 4})
        YIELD nodeId, score
        WITH gds.util.asNode(nodeId) AS u, score
        RETURN u.user_id AS user_id, u.name AS name,
               u.review_count AS review_count, score AS betweenness
        ORDER BY score DESC LIMIT 100
    """, timeout=600)
    bc_df = pd.DataFrame(bc_results)
    top20_bc = bc_df.head(20)

    print("  Step 2: Running Degree Centrality...")
    deg_results = run_query("""
        CALL gds.degree.stream('friends-graph')
        YIELD nodeId, score
        WITH gds.util.asNode(nodeId) AS u, score
        RETURN u.user_id AS user_id, u.name AS name,
               u.review_count AS review_count, score AS degree
        ORDER BY score DESC LIMIT 100
    """, timeout=300)
    deg_df = pd.DataFrame(deg_results)
    top20_deg = deg_df.head(20)

    bc_set = set(top20_bc["user_id"])
    deg_set = set(top20_deg["user_id"])
    overlap = bc_set & deg_set
    high_bc_low_deg = bc_set - deg_set
    high_deg_low_bc = deg_set - bc_set

    print(f"\n  Overlap: {len(overlap)}")
    print(f"  High-BC/Low-Deg: {len(high_bc_low_deg)}")
    print(f"  High-Deg/Low-BC: {len(high_deg_low_bc)}")

    # Compare groups
    group_stats = {}
    for gname, uids in [("High-BC/Low-Deg", list(high_bc_low_deg)),
                          ("High-Deg/Low-BC", list(high_deg_low_bc))]:
        if not uids:
            continue
        data = run_query("""
            UNWIND $uids AS uid
            MATCH (u:User {user_id: uid})-[:WROTE]->(r:Review)-[:REVIEWS]->(b:Business)
            WITH u, count(DISTINCT b.city) AS cities, count(r) AS reviews
            OPTIONAL MATCH (u)-[:WROTE]->(:Review)-[:REVIEWS]->(:Business)-[:IN_CATEGORY]->(c:Category)
            WITH u, cities, reviews, count(DISTINCT c.name) AS cats
            RETURN avg(reviews) AS mean_reviews, avg(cities) AS mean_cities, avg(cats) AS mean_cats, count(u) AS n
        """, timeout=120, uids=uids)
        if data:
            group_stats[gname] = {
                "n": data[0]["n"],
                "mean_reviews": round(data[0]["mean_reviews"] or 0, 1),
                "mean_cities": round(data[0]["mean_cities"] or 0, 1),
                "mean_categories": round(data[0]["mean_cats"] or 0, 1),
            }

    print("\n  Group Comparison:")
    for g, s in group_stats.items():
        print(f"    {g}: {s}")

    comp_df = pd.DataFrame([{"group": k, **v} for k, v in group_stats.items()])
    comp_df.to_csv(os.path.join(DATA, "q4_group_comparison.csv"), index=False)

    # Merge for scatter
    all_bc = {r["user_id"]: r["betweenness"] for r in bc_results}
    all_deg = {r["user_id"]: r["degree"] for r in deg_results}
    all_uids = set(list(all_bc.keys()) + list(all_deg.keys()))

    scatter = []
    for uid in all_uids:
        bc_val = all_bc.get(uid, 0)
        deg_val = all_deg.get(uid, 0)
        grp = ("Both" if uid in overlap else "Top BC" if uid in bc_set
               else "Top Deg" if uid in deg_set else "Other")
        scatter.append({"user_id": uid, "betweenness": bc_val, "degree": deg_val, "group": grp})
    sdf = pd.DataFrame(scatter)

    # Viz 1: Scatter
    fig, ax = plt.subplots(figsize=(9, 7))
    gc = {"Other": "#cccccc", "Top BC": "#e74c3c", "Top Deg": "#3498db", "Both": "#2ecc71"}
    for grp in ["Other", "Top Deg", "Top BC", "Both"]:
        sub = sdf[sdf["group"] == grp]
        if len(sub) == 0:
            continue
        ax.scatter(sub["degree"], sub["betweenness"], label=grp,
                   color=gc[grp], s=50 if grp != "Other" else 20,
                   alpha=0.8 if grp != "Other" else 0.3, zorder=3 if grp != "Other" else 1)
    ax.set_xlabel("Degree Centrality")
    ax.set_ylabel("Betweenness Centrality")
    ax.set_title("Betweenness vs Degree Centrality")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig("q4_bc_vs_degree_scatter.png")

    # Viz 2: Venn
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, f"Top 20 Betweenness ∩ Top 20 Degree\n= {len(overlap)} users\n\n"
            f"BC only: {len(high_bc_low_deg)} | Deg only: {len(high_deg_low_bc)}",
            ha="center", va="center", fontsize=13, transform=ax.transAxes)
    ax.set_title("Overlap of Top 20 Sets")
    ax.axis("off")
    save_fig("q4_venn_overlap.png")

    # Viz 3: Comparison
    if len(comp_df) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, m, t in zip(axes, ["mean_reviews", "mean_cities", "mean_categories"],
                             ["Mean Reviews", "Mean Distinct Cities", "Mean Distinct Categories"]):
            ax.bar(comp_df["group"], comp_df[m],
                   color=[sns.color_palette("colorblind")[0], sns.color_palette("colorblind")[1]][:len(comp_df)])
            ax.set_title(t)
            ax.tick_params(axis="x", labelsize=8)
        plt.suptitle("High-BC/Low-Deg vs High-Deg/Low-BC Users", fontsize=11)
        plt.tight_layout()
        save_fig("q4_group_comparison.png")

    # Export betweenness for predictive model
    print("  Exporting betweenness scores...")
    bc_all = run_query("""
        CALL gds.betweenness.stream('friends-graph', {samplingSize: 500, concurrency: 4})
        YIELD nodeId, score
        WITH gds.util.asNode(nodeId) AS u, score
        WHERE score > 0
        RETURN u.user_id AS user_id, score AS betweenness
        ORDER BY score DESC LIMIT 50000
    """, timeout=600)
    pd.DataFrame(bc_all).to_csv(os.path.join(DATA, "user_betweenness.csv"), index=False)
    print(f"  Exported {len(bc_all)} betweenness scores")

    drop_graph_if_exists("friends-graph")
    return comp_df


# =============================================================================
# Q5: Link Prediction for Restaurant Recommendation (7 marks)
# =============================================================================
def run_q5():
    print("\n" + "=" * 70)
    print("Q5: Link Prediction for Restaurant Recommendation")
    print("=" * 70)

    # Python-based link prediction with Neo4j-extracted features
    print("  Step 1: Getting restaurant review edges...")
    edges = run_query("""
        MATCH (u:User)-[:WROTE]->(r:Review)-[:REVIEWS]->(b:Business)
              -[:IN_CATEGORY]->(c:Category {name: 'Restaurants'})
        WITH u, b, r
        RETURN u.user_id AS uid, b.business_id AS bid,
               r.stars AS stars, r.date AS date,
               u.review_count AS u_rc, u.average_stars AS u_avg, u.fans AS u_fans,
               b.stars AS b_stars, b.review_count AS b_rc
        ORDER BY r.date
    """, timeout=600)
    print(f"  {len(edges):,} restaurant review edges")

    edf = pd.DataFrame(edges)
    edf["date"] = pd.to_datetime(edf["date"])

    # Chrono split: each user's last review = test
    user_counts = edf["uid"].value_counts()
    multi = user_counts[user_counts >= 2].index
    edf = edf[edf["uid"].isin(multi)].copy()
    last_idx = edf.groupby("uid")["date"].idxmax()
    test_mask = edf.index.isin(last_idx)
    train_df = edf[~test_mask]
    test_df = edf[test_mask]
    print(f"  Train: {len(train_df):,}, Test: {len(test_df):,}")

    # User/biz features from training data
    user_feats = train_df.groupby("uid").agg(
        u_train_rc=("stars", "count"), u_train_avg=("stars", "mean")
    ).fillna(0)
    biz_feats = train_df.groupby("bid").agg(
        b_train_rc=("stars", "count"), b_train_avg=("stars", "mean")
    ).fillna(0)

    # Community IDs
    comm = {r["user_id"]: r["communityId"] for r in run_query(
        "MATCH (u:User) WHERE u.communityId IS NOT NULL RETURN u.user_id AS user_id, u.communityId AS communityId"
    )}

    # Build features
    all_biz = edf["bid"].unique()
    np.random.seed(42)

    def make_features(uid, bid, label, row_data=None):
        uf = user_feats.loc[uid] if uid in user_feats.index else pd.Series({"u_train_rc": 0, "u_train_avg": 0})
        bf = biz_feats.loc[bid] if bid in biz_feats.index else pd.Series({"b_train_rc": 0, "b_train_avg": 0})
        u_rc = row_data["u_rc"] if row_data is not None else 0
        u_avg = row_data["u_avg"] if row_data is not None else 3.5
        u_fans = row_data["u_fans"] if row_data is not None else 0
        b_stars = row_data["b_stars"] if row_data is not None else 3.5
        b_rc = row_data["b_rc"] if row_data is not None else 0
        return {
            "label": label, "u_rc": u_rc, "u_avg": u_avg, "u_fans": u_fans,
            "b_stars": b_stars, "b_rc": b_rc,
            "u_train_rc": uf.get("u_train_rc", 0), "u_train_avg": uf.get("u_train_avg", 0),
            "b_train_rc": bf.get("b_train_rc", 0), "b_train_avg": bf.get("b_train_avg", 0),
            "has_community": 1 if uid in comm else 0,
        }

    # Sample test users
    test_users = test_df["uid"].unique()
    np.random.shuffle(test_users)
    test_users = test_users[:500]

    # Test features: positives + negatives
    feat_rows = []
    uid_col = []
    for _, row in test_df[test_df["uid"].isin(test_users)].iterrows():
        feat_rows.append(make_features(row["uid"], row["bid"], 1, row))
        uid_col.append(row["uid"])
        # Negatives
        visited = set(edf[edf["uid"] == row["uid"]]["bid"])
        negs = [b for b in np.random.choice(all_biz, 5, replace=False) if b not in visited][:3]
        biz_lookup = edf.groupby("bid").first()
        for nb in negs:
            rd = {"u_rc": row["u_rc"], "u_avg": row["u_avg"], "u_fans": row["u_fans"],
                  "b_stars": biz_lookup.loc[nb]["b_stars"] if nb in biz_lookup.index else 3.5,
                  "b_rc": biz_lookup.loc[nb]["b_rc"] if nb in biz_lookup.index else 0}
            feat_rows.append(make_features(row["uid"], nb, 0, rd))
            uid_col.append(row["uid"])

    feat_df = pd.DataFrame(feat_rows)
    feat_df["uid"] = uid_col

    # Train features (sample)
    train_users = train_df["uid"].unique()
    np.random.shuffle(train_users)
    train_sample = train_users[:2000]
    train_rows = []
    biz_lookup = edf.groupby("bid").first()
    for uid in train_sample:
        ur = train_df[train_df["uid"] == uid]
        visited = set(ur["bid"])
        for _, row in ur.sample(min(3, len(ur))).iterrows():
            train_rows.append(make_features(uid, row["bid"], 1, row))
            negs = [b for b in np.random.choice(all_biz, 5, replace=False) if b not in visited][:3]
            for nb in negs:
                rd = {"u_rc": row["u_rc"], "u_avg": row["u_avg"], "u_fans": row["u_fans"],
                      "b_stars": biz_lookup.loc[nb]["b_stars"] if nb in biz_lookup.index else 3.5,
                      "b_rc": biz_lookup.loc[nb]["b_rc"] if nb in biz_lookup.index else 0}
                train_rows.append(make_features(uid, nb, 0, rd))

    train_feat_df = pd.DataFrame(train_rows)

    # Train model
    print("  Step 4: Training model...")
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, roc_curve

    feature_cols = ["u_rc", "u_avg", "u_fans", "b_stars", "b_rc",
                    "u_train_rc", "u_train_avg", "b_train_rc", "b_train_avg", "has_community"]

    X_train = train_feat_df[feature_cols].fillna(0)
    y_train = train_feat_df["label"]
    X_test = feat_df[feature_cols].fillna(0)
    y_test = feat_df["label"]

    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"  AUC-ROC: {auc:.4f}")

    # Precision@10
    feat_df["prob"] = y_prob
    p10_list = []
    for uid in test_users[:100]:
        up = feat_df[feat_df["uid"] == uid].sort_values("prob", ascending=False).head(10)
        if len(up) > 0:
            p10_list.append(up["label"].mean())
    p10 = np.mean(p10_list)
    print(f"  Precision@10: {p10:.4f}")

    # Feature importance
    imp = pd.DataFrame({"feature": feature_cols, "importance": clf.feature_importances_}).sort_values("importance", ascending=False)
    print("\n  Top 5 Features:")
    print(imp.head(5).to_string(index=False))

    # Sample recommendations
    print("\n  Sample Recommendations (5 users):")
    for uid in test_users[:5]:
        up = feat_df[(feat_df["uid"] == uid) & (feat_df["label"] == 0)].sort_values("prob", ascending=False).head(3)
        print(f"    User {uid[:12]}... -> {len(up)} recs")

    imp.to_csv(os.path.join(DATA, "q5_feature_importance.csv"), index=False)

    # Viz 1: ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"GBM (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Restaurant Link Prediction")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig("q5_roc_curve.png")

    # Viz 2: Feature importance
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp["feature"], imp["importance"], color=sns.color_palette("colorblind")[0])
    ax.set_xlabel("Importance")
    ax.set_title("Link Prediction Feature Importance")
    ax.invert_yaxis()
    plt.tight_layout()
    save_fig("q5_feature_importance.png")

    return {"auc": auc, "p10": p10}


# =============================================================================
if __name__ == "__main__":
    try:
        run_q1()
        run_q2()
        run_q3()
        run_q4()
        run_q5()
        print("\n" + "=" * 70)
        print("All Neo4j queries complete!")
        print("=" * 70)
    finally:
        driver.close()
