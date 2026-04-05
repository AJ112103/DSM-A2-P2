#!/usr/bin/env python3
"""
CS-3510 Assignment 2 Part 2 — Predictive Modelling
Predict useful votes per review using review, user, and graph features.
Aamer Jalan, Ashoka University
"""

import os
import pymongo
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import shap

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

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["yelp_subset"]


def save_fig(name):
    path = os.path.join(FIGURES, name)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# =============================================================================
# Step 1: Feature Extraction
# =============================================================================
print("=" * 70)
print("Step 1: Feature Extraction")
print("=" * 70)

# Load graph features
print("  Loading graph features...")
user_pr = pd.read_csv(os.path.join(DATA, "user_pagerank.csv"))
user_bc = pd.read_csv(os.path.join(DATA, "user_betweenness.csv"))
biz_jac = pd.read_csv(os.path.join(DATA, "business_jaccard.csv"))
comm_sizes = pd.read_csv(os.path.join(DATA, "community_sizes.csv"))

# Get community assignments from Neo4j (saved during Q2)
from neo4j import GraphDatabase
neo_driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "Aamer@DSM"))
with neo_driver.session() as s:
    comm_data = s.run("""
        MATCH (u:User) WHERE u.communityId IS NOT NULL
        RETURN u.user_id AS user_id, u.communityId AS communityId
    """).data()
neo_driver.close()
user_comm = pd.DataFrame(comm_data)
user_comm = user_comm.merge(comm_sizes, left_on="communityId", right_on="community_id", how="left")
user_comm = user_comm[["user_id", "communityId", "size"]].rename(columns={"size": "community_size"})
print(f"  PageRank: {len(user_pr)}, Betweenness: {len(user_bc)}, Jaccard: {len(biz_jac)}, Communities: {len(user_comm)}")

# Build user lookup
print("  Building user feature map...")
user_map = {}
for u in db.users.find({}, {
    "user_id": 1, "yelping_since": 1, "review_count": 1,
    "average_stars": 1, "fans": 1, "elite": 1, "_id": 0
}):
    tenure = 2022 - int(u["yelping_since"][:4])  # years since joining
    elite = u.get("elite") or []
    user_map[u["user_id"]] = {
        "user_tenure": tenure,
        "user_review_count": u.get("review_count", 0),
        "user_avg_stars": u.get("average_stars", 0),
        "user_fans": u.get("fans", 0),
        "is_elite": 1 if len(elite) > 0 else 0,
        "elite_years": len(elite),
    }
print(f"  {len(user_map)} users loaded")

# Merge graph features into user map
pr_map = dict(zip(user_pr["user_id"], user_pr["pagerank"]))
bc_map = dict(zip(user_bc["user_id"], user_bc["betweenness"]))
comm_map = dict(zip(user_comm["user_id"], user_comm["community_size"].fillna(0)))
jac_map = dict(zip(biz_jac["business_id"], biz_jac["mean_jaccard"]))

# Stream reviews and build feature matrix
print("  Streaming reviews and building feature matrix...")
# Use a sample for efficiency (2.76M reviews is large)
# Stratified sample: ensure we get enough high-useful reviews
SAMPLE_SIZE = 300000

# First, count distribution
useful_dist = list(db.reviews.aggregate([
    {"$group": {"_id": {"$switch": {
        "branches": [
            {"case": {"$eq": ["$useful", 0]}, "then": "0"},
            {"case": {"$lte": ["$useful", 5]}, "then": "1-5"},
            {"case": {"$lte": ["$useful", 20]}, "then": "6-20"},
        ],
        "default": "21+"
    }}, "count": {"$sum": 1}}}
]))
print("  Useful vote distribution:")
for d in useful_dist:
    print(f"    {d['_id']}: {d['count']:,}")

# Sample reviews
rows = []
count = 0
for r in db.reviews.find({}, {
    "review_id": 1, "user_id": 1, "business_id": 1,
    "stars": 1, "useful": 1, "text": 1, "date": 1, "_id": 0
}):
    # Stratified sampling: keep all reviews with useful > 5, sample others
    useful = r.get("useful", 0)
    if useful <= 5:
        if np.random.random() > 0.08:  # ~8% sample of low-useful
            continue

    uid = r["user_id"]
    bid = r["business_id"]
    text = r.get("text", "")

    u = user_map.get(uid, {})
    if not u:
        continue

    # Review features
    text_len = len(text)
    word_count = len(text.split())
    date_str = r.get("date", "2020-01-01")
    try:
        review_year = int(date_str[:4])
        review_month = int(date_str[5:7])
        recency = (2022 - review_year) * 365 + (1 - review_month) * 30
    except:
        recency = 365

    row = {
        "useful": useful,
        "star_rating": r["stars"],
        "text_length": text_len,
        "word_count": word_count,
        "recency_days": recency,
        # User features
        "user_tenure": u.get("user_tenure", 0),
        "user_review_count": u.get("user_review_count", 0),
        "user_avg_stars": u.get("user_avg_stars", 0),
        "user_fans": u.get("user_fans", 0),
        "is_elite": u.get("is_elite", 0),
        "elite_years": u.get("elite_years", 0),
        # Graph features
        "user_pagerank": pr_map.get(uid, 0.15),
        "user_betweenness": bc_map.get(uid, 0),
        "user_community_size": comm_map.get(uid, 0),
        "biz_mean_jaccard": jac_map.get(bid, 0),
    }
    rows.append(row)
    count += 1
    if count % 50000 == 0:
        print(f"    {count:,} reviews sampled...")
    if count >= SAMPLE_SIZE:
        break

df = pd.DataFrame(rows)
print(f"  Final sample: {len(df):,} reviews")
print(f"  Useful distribution in sample:")
print(f"    0: {(df['useful']==0).sum():,}")
print(f"    1-5: {((df['useful']>=1) & (df['useful']<=5)).sum():,}")
print(f"    6-20: {((df['useful']>=6) & (df['useful']<=20)).sum():,}")
print(f"    21+: {(df['useful']>=21).sum():,}")

# =============================================================================
# Step 2: Model Training & Evaluation
# =============================================================================
print("\n" + "=" * 70)
print("Step 2: Model Training & Evaluation")
print("=" * 70)

feature_cols = [
    "star_rating", "text_length", "word_count", "recency_days",
    "user_tenure", "user_review_count", "user_avg_stars", "user_fans",
    "is_elite", "elite_years",
    "user_pagerank", "user_betweenness", "user_community_size", "biz_mean_jaccard"
]

X = df[feature_cols].fillna(0)
y = df["useful"]

# Log-transform target for training (right-skewed)
y_log = np.log1p(y)

# Stratified split by useful bucket
df["useful_bucket"] = pd.cut(df["useful"], bins=[-1, 0, 5, 20, float("inf")],
                              labels=["0", "1-5", "6-20", "21+"])
X_train, X_test, y_train_log, y_test_log, y_train, y_test = train_test_split(
    X, y_log, y, test_size=0.2, random_state=42,
    stratify=df["useful_bucket"]
)

print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")

# Train LightGBM
print("  Training LightGBM...")
model = lgb.LGBMRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)
model.fit(X_train, y_train_log,
          eval_set=[(X_test, y_test_log)],
          callbacks=[lgb.early_stopping(50, verbose=False)])

# Predictions (inverse log transform)
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_pred = np.maximum(y_pred, 0)  # clip negatives

# Overall metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\n  Overall Metrics:")
print(f"    RMSE: {rmse:.4f}")
print(f"    MAE:  {mae:.4f}")
print(f"    R²:   {r2:.4f}")

# Per-bucket metrics
test_df = X_test.copy()
test_df["y_true"] = y_test.values
test_df["y_pred"] = y_pred
test_df["bucket"] = pd.cut(test_df["y_true"], bins=[-1, 0, 5, float("inf")],
                            labels=["0", "1-5", "6+"])

print(f"\n  Per-Bucket Metrics:")
print(f"  {'Bucket':<10} {'N':>6} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
bucket_metrics = []
for bucket in ["0", "1-5", "6+"]:
    sub = test_df[test_df["bucket"] == bucket]
    if len(sub) == 0:
        continue
    b_rmse = np.sqrt(mean_squared_error(sub["y_true"], sub["y_pred"]))
    b_mae = mean_absolute_error(sub["y_true"], sub["y_pred"])
    b_r2 = r2_score(sub["y_true"], sub["y_pred"]) if len(sub) > 1 else 0
    print(f"  {bucket:<10} {len(sub):>6} {b_rmse:>8.4f} {b_mae:>8.4f} {b_r2:>8.4f}")
    bucket_metrics.append({"bucket": bucket, "n": len(sub), "rmse": b_rmse, "mae": b_mae, "r2": b_r2})

pd.DataFrame(bucket_metrics).to_csv(os.path.join(DATA, "pred_bucket_metrics.csv"), index=False)

# =============================================================================
# Step 3: Feature Importance & Discussion
# =============================================================================
print("\n" + "=" * 70)
print("Step 3: Feature Importance")
print("=" * 70)

# SHAP values
print("  Computing SHAP values...")
explainer = shap.TreeExplainer(model)
# Use a sample for SHAP (computing on full test set is slow)
shap_sample = X_test.sample(min(5000, len(X_test)), random_state=42)
shap_values = explainer.shap_values(shap_sample)

# Top 3 features
mean_shap = np.abs(shap_values).mean(axis=0)
feat_imp = pd.DataFrame({
    "feature": feature_cols,
    "mean_shap": mean_shap,
    "lgbm_importance": model.feature_importances_
}).sort_values("mean_shap", ascending=False)

print("\n  Top Features by SHAP:")
print(feat_imp.head(5).to_string(index=False))
feat_imp.to_csv(os.path.join(DATA, "pred_feature_importance.csv"), index=False)

# ── Visualization 1: SHAP Summary Plot ──
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values, shap_sample, feature_names=feature_cols, show=False,
                  max_display=14)
plt.title("SHAP Feature Importance — Useful Vote Prediction")
plt.tight_layout()
save_fig("pred_shap_summary.png")

# ── Visualization 2: Predicted vs Actual ──
fig, ax = plt.subplots(figsize=(7, 7))
# Subsample for clarity
sample_idx = np.random.choice(len(y_test), min(5000, len(y_test)), replace=False)
ax.scatter(y_test.values[sample_idx], y_pred[sample_idx], alpha=0.2, s=10)
max_val = min(y_test.max(), 100)
ax.plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="Perfect prediction")
ax.set_xlabel("Actual Useful Votes")
ax.set_ylabel("Predicted Useful Votes")
ax.set_title(f"Predicted vs Actual (R² = {r2:.3f})")
ax.set_xlim(0, max_val)
ax.set_ylim(0, max_val)
ax.legend()
ax.grid(alpha=0.3)
save_fig("pred_vs_actual.png")

# ── Visualization 3: Residual Distribution by Bucket ──
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, bucket in zip(axes, ["0", "1-5", "6+"]):
    sub = test_df[test_df["bucket"] == bucket]
    residuals = sub["y_true"] - sub["y_pred"]
    ax.hist(residuals, bins=50, color=sns.color_palette("colorblind")[0], alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_title(f"Bucket: {bucket} useful (n={len(sub):,})")
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_ylabel("Count")
plt.suptitle("Residual Distribution by Useful Vote Bucket", fontsize=13)
plt.tight_layout()
save_fig("pred_residuals.png")

# ── Visualization 4: Feature Importance Bar Chart ──
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(feat_imp["feature"], feat_imp["mean_shap"], color=sns.color_palette("colorblind")[0])
ax.set_xlabel("Mean |SHAP Value|")
ax.set_title("Feature Importance (SHAP)")
ax.invert_yaxis()
plt.tight_layout()
save_fig("pred_feature_importance.png")

print("\n" + "=" * 70)
print("Predictive Modelling Complete!")
print("=" * 70)
