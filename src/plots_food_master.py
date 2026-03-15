import os
import pandas as pd
import matplotlib.pyplot as plt
import textwrap
import numpy as np


DIRECT_PATH = "data/processed/predictions_llm_direct_food_master.csv"
PIPELINE_PATH = "data/processed/predictions_llm_pipeline_food_master.csv"
OUT_DIR = "reports/figures"

direct_df = pd.read_csv(DIRECT_PATH)[["meal_title", "pred_calories"]].copy()
pipeline_df = pd.read_csv(PIPELINE_PATH)[["meal_title", "pred_calories"]].copy()

print("DIRECT PREDICTIONS")
print("------------------")
print("count:", len(direct_df))
print("mean:", direct_df["pred_calories"].mean())
print("median:", direct_df["pred_calories"].median())
print("min:", direct_df["pred_calories"].min())
print("max:", direct_df["pred_calories"].max())
print("range:", direct_df["pred_calories"].max() - direct_df["pred_calories"].min())

print("\nPIPELINE PREDICTIONS")
print("--------------------")
print("count:", len(pipeline_df))
print("mean:", pipeline_df["pred_calories"].mean())
print("median:", pipeline_df["pred_calories"].median())
print("min:", pipeline_df["pred_calories"].min())
print("max:", pipeline_df["pred_calories"].max())
print("range:", pipeline_df["pred_calories"].max() - pipeline_df["pred_calories"].min())

diff = pipeline_df["pred_calories"] - direct_df["pred_calories"]

print("\nCOMPARISON")
print("----------")
print("pipeline higher count:", (diff > 0).sum())
print("pipeline higher percent:", (diff > 0).mean() * 100)
print("median difference:", diff.median())
print("mean difference:", diff.mean())

# Histogram: direct
direct_plot = direct_df.loc[direct_df["pred_calories"] <= 3000, "pred_calories"]
plt.figure(figsize=(8, 5))
plt.hist(direct_plot, bins=30)
plt.xlabel("Predicted calories (kcal)")
plt.ylabel("Frequency")
plt.title("Histogram of direct calorie predictions")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "hist_direct_food_master.png"), dpi=300)
plt.close()

# Histogram: pipeline
pipeline_plot = pipeline_df.loc[pipeline_df["pred_calories"] <= 3000, "pred_calories"]
plt.figure(figsize=(8, 5))
plt.hist(pipeline_plot, bins=30)
plt.xlabel("Predicted calories (kcal)")
plt.ylabel("Frequency")
plt.title("Histogram of pipeline calorie predictions")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "hist_pipeline_food_master.png"), dpi=300)
plt.close()

# Top disagreement examples
top_idx = (pipeline_df["pred_calories"] - direct_df["pred_calories"]).sort_values(ascending=False).head(10).index

top_titles = direct_df.loc[top_idx, "meal_title"]
top_direct = direct_df.loc[top_idx, "pred_calories"]
top_pipeline = pipeline_df.loc[top_idx, "pred_calories"]

plot_df = pd.DataFrame({
    "meal_title": top_titles,
    "direct_pred": top_direct,
    "pipeline_pred": top_pipeline}).iloc[::-1]

wrapped_titles = [textwrap.fill(title, width=35) for title in plot_df["meal_title"]]

plt.figure(figsize=(10, 8))

y = np.arange(len(plot_df)) * 1.3
bar_height = 0.35

plt.barh(y - bar_height / 2, plot_df["direct_pred"], height=bar_height, label="Direct")
plt.barh(y + bar_height / 2, plot_df["pipeline_pred"], height=bar_height, label="Pipeline")

plt.yticks(y, wrapped_titles)
plt.xlabel("Predicted calories (kcal)")
plt.ylabel("Meal title")
plt.title("Top positive disagreement examples: direct vs. pipeline predictions")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "top_disagreement_food_master.png"), dpi=300)
plt.close()


print("Saved")