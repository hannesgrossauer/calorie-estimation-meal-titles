import pandas as pd
import matplotlib.pyplot as plt

direct = pd.read_csv("data/processed/predictions_llm_direct_test_set.csv")
pipeline = pd.read_csv("data/processed/predictions_llm_pipeline_test_set.csv")


# Histogram 1: Direct LLM vs reference
plt.figure(figsize=(8, 5))
plt.hist(direct["total_calories"], bins=30, alpha=0.7, label="Reference total calories")
plt.hist(direct["pred_calories"], bins=30, alpha=0.7, label="Direct LLM predictions")
plt.xlim(0, 1800)
plt.ylim(0, 130)
plt.xlabel("Total calories (kcal)")
plt.ylabel("Frequency")
plt.title("Distribution of Reference and Direct LLM Predicted Calories")
plt.legend()
plt.tight_layout()
plt.savefig("reports/figures/hist_direct_vs_reference_nutrition5k.png", dpi=300)
plt.close()

# Histogram 2: Pipeline LLM vs reference
plt.figure(figsize=(8, 5))
plt.hist(pipeline["total_calories"], bins=30, alpha=0.7, label="Reference total calories")
plt.hist(pipeline["pred_calories"], bins=30, alpha=0.7, label="Pipeline LLM predictions")
plt.xlim(0, 1800)
plt.ylim(0, 130)
plt.xlabel("Total calories (kcal)")
plt.ylabel("Frequency")
plt.title("Distribution of Reference and Pipeline LLM Predicted Calories")
plt.legend()
plt.tight_layout()
plt.savefig("reports/figures/hist_pipeline_vs_reference_nutrition5k.png", dpi=300)
plt.close()

print("Saved")