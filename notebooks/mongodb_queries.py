#!/usr/bin/env python3
"""
CS-3510 Assignment 2 Part 2 — MongoDB Queries (Q1–Q3)
Aamer Jalan, Ashoka University
"""

import pymongo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# ── Setup ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})
sns.set_palette("colorblind")

FIGURES = os.path.join(os.path.dirname(__file__), "..", "report", "figures")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["yelp_subset"]

# ── Helper ───────────────────────────────────────────────────────────────────
def save_fig(name):
    path = os.path.join(FIGURES, name)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")

# =============================================================================
# Q1: Cohort Analysis of User Reviewing Behaviour
# =============================================================================
def run_q1():
    print("\n" + "=" * 70)
    print("Q1: Cohort Analysis of User Reviewing Behaviour")
    print("=" * 70)

    # Strategy: use Python-side join for speed (avoid $lookup on 2.76M docs)
    # Step 1: Build user_id -> cohort year map
    print("  Building user cohort map...")
    user_cohorts = {}
    for u in db.users.find({}, {"user_id": 1, "yelping_since": 1, "_id": 0}):
        year = int(u["yelping_since"][:4])
        user_cohorts[u["user_id"]] = year

    print(f"  {len(user_cohorts)} users mapped to cohorts")

    # Step 2: Aggregate reviews with cohort info
    # We'll stream reviews and accumulate stats per cohort
    print("  Streaming reviews...")
    cohort_stats = {}
    count = 0
    for r in db.reviews.find({}, {
        "user_id": 1, "stars": 1, "useful": 1, "text": 1, "_id": 0
    }):
        uid = r["user_id"]
        if uid not in user_cohorts:
            continue
        cy = user_cohorts[uid]
        stars = r["stars"]
        useful = r["useful"]
        text_len = len(r.get("text", ""))

        if cy not in cohort_stats:
            cohort_stats[cy] = {
                "n": 0, "star_sum": 0, "star_sq_sum": 0,
                "text_len_sum": 0, "useful_sum": 0,
                "star_counts": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            }
        s = cohort_stats[cy]
        s["n"] += 1
        s["star_sum"] += stars
        s["star_sq_sum"] += stars ** 2
        s["text_len_sum"] += text_len
        s["useful_sum"] += useful
        s["star_counts"][int(stars)] += 1
        count += 1
        if count % 500000 == 0:
            print(f"    {count:,} reviews processed...")

    print(f"  Total: {count:,} reviews processed")

    # Build results table
    rows = []
    for year in sorted(cohort_stats.keys()):
        s = cohort_stats[year]
        n = s["n"]
        mean_stars = s["star_sum"] / n
        std_stars = (s["star_sq_sum"] / n - mean_stars ** 2) ** 0.5
        mean_text = s["text_len_sum"] / n
        mean_useful = s["useful_sum"] / n
        star_props = {k: v / n for k, v in s["star_counts"].items()}
        rows.append({
            "cohort_year": year,
            "reviews": n,
            "mean_stars": round(mean_stars, 3),
            "std_stars": round(std_stars, 3),
            "mean_text_len": round(mean_text, 1),
            "mean_useful": round(mean_useful, 3),
            "prop_1": round(star_props[1], 4),
            "prop_2": round(star_props[2], 4),
            "prop_3": round(star_props[3], 4),
            "prop_4": round(star_props[4], 4),
            "prop_5": round(star_props[5], 4),
        })

    df = pd.DataFrame(rows)
    print("\n  Results:")
    print(df.to_string(index=False))

    # Identify extremes
    max_stars_row = df.loc[df["mean_stars"].idxmax()]
    max_useful_row = df.loc[df["mean_useful"].idxmax()]
    print(f"\n  Highest mean star rating: {max_stars_row['cohort_year']} ({max_stars_row['mean_stars']})")
    print(f"  Highest mean useful votes: {max_useful_row['cohort_year']} ({max_useful_row['mean_useful']})")

    # Save data
    df.to_csv(os.path.join(DATA, "q1_cohort_analysis.csv"), index=False)

    # ── Visualization 1: Grouped bar chart — star distribution by cohort ──
    fig, ax = plt.subplots(figsize=(12, 5))
    cohorts = df["cohort_year"].values
    x = np.arange(len(cohorts))
    width = 0.15
    colors = sns.color_palette("colorblind", 5)
    for i, star in enumerate([1, 2, 3, 4, 5]):
        ax.bar(x + (i - 2) * width, df[f"prop_{star}"], width,
               label=f"{star} star", color=colors[i])
    ax.set_xlabel("Cohort Year (Year Joined Yelp)")
    ax.set_ylabel("Proportion of Reviews")
    ax.set_title("Star Rating Distribution by User Cohort")
    ax.set_xticks(x)
    ax.set_xticklabels(cohorts, rotation=45, ha="right")
    ax.legend(title="Stars", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    save_fig("q1_star_distribution_by_cohort.png")

    # ── Visualization 2: Line chart — mean useful votes over cohorts ──
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["cohort_year"], df["mean_useful"], marker="o", linewidth=2, color=sns.color_palette("colorblind")[0])
    ax.set_xlabel("Cohort Year")
    ax.set_ylabel("Mean Useful Votes per Review")
    ax.set_title("Mean Useful Votes per Review by User Cohort")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig("q1_mean_useful_by_cohort.png")

    # ── Visualization 3: Heatmap — star proportions by cohort ──
    heatmap_data = df[["cohort_year", "prop_1", "prop_2", "prop_3", "prop_4", "prop_5"]].set_index("cohort_year")
    heatmap_data.columns = ["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlOrRd",
                ax=ax, linewidths=0.5, cbar_kws={"label": "Proportion"})
    ax.set_xlabel("Star Category")
    ax.set_ylabel("Cohort Year")
    ax.set_title("Star Rating Proportions by User Cohort")
    plt.tight_layout()
    save_fig("q1_star_heatmap.png")

    return df


# =============================================================================
# Q2: Month-over-Month Category Rating Trends
# =============================================================================
def run_q2():
    print("\n" + "=" * 70)
    print("Q2: Month-over-Month Category Rating Trends")
    print("=" * 70)

    # Use aggregation pipeline to group by (category, year-month)
    print("  Running aggregation pipeline...")
    pipeline = [
        {"$addFields": {"yearMonth": {"$substr": ["$date", 0, 7]}}},
        {"$lookup": {
            "from": "businesses",
            "localField": "business_id",
            "foreignField": "business_id",
            "pipeline": [{"$project": {"categories": 1, "_id": 0}}],
            "as": "biz"
        }},
        {"$unwind": "$biz"},
        {"$unwind": "$biz.categories"},
        {"$group": {
            "_id": {"cat": "$biz.categories", "ym": "$yearMonth"},
            "avgStars": {"$avg": "$stars"},
            "count": {"$sum": 1}
        }},
        {"$group": {
            "_id": "$_id.cat",
            "totalReviews": {"$sum": "$count"},
            "monthly": {"$push": {"ym": "$_id.ym", "avg": "$avgStars", "count": "$count"}}
        }},
        {"$match": {"totalReviews": {"$gte": 500}}},
        {"$addFields": {
            "monthly": {"$sortArray": {"input": "$monthly", "sortBy": {"ym": 1}}}
        }}
    ]

    results = list(db.reviews.aggregate(pipeline, allowDiskUse=True))
    print(f"  {len(results)} categories with ≥500 reviews")

    # Compute MoM changes and trend consistency
    rows = []
    for cat_doc in results:
        category = cat_doc["_id"]
        monthly = cat_doc["monthly"]
        total = cat_doc["totalReviews"]

        if len(monthly) < 2:
            continue

        changes = []
        for i in range(len(monthly) - 1):
            changes.append(monthly[i + 1]["avg"] - monthly[i]["avg"])

        n_pairs = len(changes)
        upward_consistency = sum(1 for c in changes if c > 0) / n_pairs
        downward_consistency = sum(1 for c in changes if c < 0) / n_pairs

        rows.append({
            "category": category,
            "total_reviews": total,
            "n_months": len(monthly),
            "n_pairs": n_pairs,
            "upward_consistency": round(upward_consistency, 4),
            "downward_consistency": round(downward_consistency, 4),
            "monthly": monthly
        })

    df = pd.DataFrame(rows)

    # Top 3 upward
    top_up = df.nlargest(3, "upward_consistency")
    print("\n  Top 3 Most Consistent Upward Trends:")
    print(top_up[["category", "total_reviews", "n_months", "upward_consistency"]].to_string(index=False))

    # Top 3 downward
    top_down = df.nlargest(3, "downward_consistency")
    print("\n  Top 3 Most Consistent Downward Trends:")
    print(top_down[["category", "total_reviews", "n_months", "downward_consistency"]].to_string(index=False))

    # Save summary
    summary = pd.concat([
        top_up[["category", "total_reviews", "n_months", "upward_consistency"]].assign(trend="upward"),
        top_down[["category", "total_reviews", "n_months", "downward_consistency"]].assign(trend="downward")
    ])
    summary.to_csv(os.path.join(DATA, "q2_trend_consistency.csv"), index=False)

    # ── Visualization 1: Line charts for top/bottom categories ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    colors = sns.color_palette("colorblind", 3)

    for idx, (_, row) in enumerate(top_up.iterrows()):
        monthly = row["monthly"]
        ym = [m["ym"] for m in monthly]
        avg = [m["avg"] for m in monthly]
        # Subsample x-axis labels if too many
        ax1.plot(range(len(ym)), avg, label=row["category"], color=colors[idx], linewidth=1.5)
    ax1.set_title("Top 3 Upward Trending Categories")
    ax1.set_xlabel("Month Index")
    ax1.set_ylabel("Avg Star Rating")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    for idx, (_, row) in enumerate(top_down.iterrows()):
        monthly = row["monthly"]
        ym = [m["ym"] for m in monthly]
        avg = [m["avg"] for m in monthly]
        ax2.plot(range(len(ym)), avg, label=row["category"], color=colors[idx], linewidth=1.5)
    ax2.set_title("Top 3 Downward Trending Categories")
    ax2.set_xlabel("Month Index")
    ax2.set_ylabel("Avg Star Rating")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_fig("q2_monthly_trends.png")

    # ── Visualization 2: Bar chart of trend consistency ──
    fig, ax = plt.subplots(figsize=(10, 5))
    all_highlight = pd.concat([
        top_up[["category", "upward_consistency"]].rename(columns={"upward_consistency": "consistency"}).assign(direction="Upward"),
        top_down[["category", "downward_consistency"]].rename(columns={"downward_consistency": "consistency"}).assign(direction="Downward")
    ])
    x = np.arange(len(all_highlight))
    colors_bar = ["#2ecc71" if d == "Upward" else "#e74c3c" for d in all_highlight["direction"]]
    ax.barh(x, all_highlight["consistency"], color=colors_bar)
    ax.set_yticks(x)
    ax.set_yticklabels([f"{row['category']} ({row['direction']})" for _, row in all_highlight.iterrows()], fontsize=9)
    ax.set_xlabel("Trend Consistency Score")
    ax.set_title("Category Rating Trend Consistency")
    ax.invert_yaxis()
    plt.tight_layout()
    save_fig("q2_trend_consistency.png")

    return df


# =============================================================================
# Q3: Check-in Frequency × Category Cross-Tabulation
# =============================================================================
def run_q3():
    print("\n" + "=" * 70)
    print("Q3: Check-in Frequency × Category Cross-Tabulation")
    print("=" * 70)

    # Step 1: Get check-in counts per business
    print("  Step 1: Computing check-in counts...")
    checkin_counts = {}
    for c in db.checkins.find({}, {"business_id": 1, "date": 1, "_id": 0}):
        checkin_counts[c["business_id"]] = len(c.get("date", []))

    counts_arr = np.array(list(checkin_counts.values()))
    q1 = np.percentile(counts_arr, 25)
    q3 = np.percentile(counts_arr, 75)
    print(f"  {len(checkin_counts)} businesses with check-ins")
    print(f"  Quartiles: Q1={q1}, Q3={q3}")

    def classify(count):
        if count <= q1:
            return "Low"
        elif count <= q3:
            return "Medium"
        else:
            return "High"

    # Step 2: Top 10 categories by total review count
    print("  Step 2: Finding top 10 categories...")
    pipeline = [
        {"$unwind": "$categories"},
        {"$group": {"_id": "$categories", "totalReviews": {"$sum": "$review_count"}}},
        {"$sort": {"totalReviews": -1}},
        {"$limit": 10}
    ]
    top_cats = [r["_id"] for r in db.businesses.aggregate(pipeline)]
    print(f"  Top 10 categories: {top_cats}")

    # Step 3: Get tip counts per business
    print("  Step 3: Computing tip counts per business...")
    tip_pipeline = [
        {"$group": {"_id": "$business_id", "tipCount": {"$sum": 1}}}
    ]
    tip_counts = {r["_id"]: r["tipCount"] for r in db.tips.aggregate(tip_pipeline)}
    print(f"  {len(tip_counts)} businesses have tips")

    # Step 4: Build cross-tabulation
    print("  Step 4: Building cross-tabulation...")
    # Get all businesses with their categories, stars, review_count
    # Filter to businesses that have check-ins
    cells = {}  # (checkin_class, category) -> list of (stars, review_count, tip_ratio)
    for b in db.businesses.find(
        {"business_id": {"$in": list(checkin_counts.keys())}},
        {"business_id": 1, "categories": 1, "stars": 1, "review_count": 1, "_id": 0}
    ):
        bid = b["business_id"]
        checkin_class = classify(checkin_counts[bid])
        cats = b.get("categories") or []
        stars = b.get("stars", 0)
        review_count = b.get("review_count", 0)
        tips = tip_counts.get(bid, 0)
        tip_ratio = tips / review_count if review_count > 0 else 0

        for cat in cats:
            if cat in top_cats:
                key = (checkin_class, cat)
                if key not in cells:
                    cells[key] = {"stars": [], "review_count": [], "tip_ratio": []}
                cells[key]["stars"].append(stars)
                cells[key]["review_count"].append(review_count)
                cells[key]["tip_ratio"].append(tip_ratio)

    # Build result table
    rows = []
    for (cls, cat), data in cells.items():
        rows.append({
            "checkin_class": cls,
            "category": cat,
            "n_businesses": len(data["stars"]),
            "mean_stars": round(np.mean(data["stars"]), 3),
            "mean_review_count": round(np.mean(data["review_count"]), 1),
            "mean_tip_ratio": round(np.mean(data["tip_ratio"]), 4),
        })

    df = pd.DataFrame(rows)
    # Order classes
    class_order = ["Low", "Medium", "High"]
    df["checkin_class"] = pd.Categorical(df["checkin_class"], categories=class_order, ordered=True)
    df = df.sort_values(["category", "checkin_class"])

    print("\n  Cross-Tabulation:")
    # Pivot for display
    for metric in ["mean_stars", "mean_review_count", "mean_tip_ratio"]:
        pivot = df.pivot(index="category", columns="checkin_class", values=metric)
        pivot = pivot[class_order]
        print(f"\n  {metric}:")
        print(pivot.to_string())

    df.to_csv(os.path.join(DATA, "q3_cross_tabulation.csv"), index=False)

    # ── Visualization 1: Heatmaps ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = [
        ("mean_stars", "Mean Star Rating", "YlOrRd"),
        ("mean_review_count", "Mean Review Count", "Blues"),
        ("mean_tip_ratio", "Tip-to-Review Ratio", "Greens")
    ]
    for ax, (metric, title, cmap) in zip(axes, metrics):
        pivot = df.pivot(index="category", columns="checkin_class", values=metric)
        pivot = pivot[class_order]
        sns.heatmap(pivot, annot=True, fmt=".2f" if metric != "mean_review_count" else ".0f",
                    cmap=cmap, ax=ax, linewidths=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Check-in Class")
        ax.set_ylabel("")
    plt.suptitle("Check-in Frequency × Category Cross-Tabulation", fontsize=13, y=1.02)
    plt.tight_layout()
    save_fig("q3_cross_tab_heatmaps.png")

    # ── Visualization 2: Grouped bar chart ──
    fig, ax = plt.subplots(figsize=(12, 5))
    # Show mean_stars across check-in classes for each category
    pivot_stars = df.pivot(index="category", columns="checkin_class", values="mean_stars")
    pivot_stars = pivot_stars[class_order]
    pivot_stars.plot(kind="bar", ax=ax, width=0.7)
    ax.set_xlabel("Category")
    ax.set_ylabel("Mean Star Rating")
    ax.set_title("Mean Star Rating by Check-in Class and Category")
    ax.legend(title="Check-in Class")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    save_fig("q3_grouped_bar.png")

    return df


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    q1_df = run_q1()
    q2_df = run_q2()
    q3_df = run_q3()
    print("\n" + "=" * 70)
    print("All MongoDB queries complete!")
    print("=" * 70)
