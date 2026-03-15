import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

TRAIN_PATH = "data/processed/train_set.csv"
TEST_PATH = "data/processed/test_set.csv"

OUT_MEDIAN = "data/processed/baseline_median.csv"
OUT_TFIDF = "data/processed/baseline_tfidf.csv"
OUT_RES = "data/processed/results.csv"


# Metrics - Returns MAE, MedAE, Acc@10/20/30
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    abs_err = np.abs(y_pred - y_true)
    mae = float(np.mean(abs_err))
    medae = float(np.median(abs_err))

    acc10 = float(np.mean(abs_err <= 0.10 * y_true) * 100.0)
    acc20 = float(np.mean(abs_err <= 0.20 * y_true) * 100.0)
    acc30 = float(np.mean(abs_err <= 0.30 * y_true) * 100.0)

    return {
        "MAE": mae,
        "MedAE": medae,
        "Acc@10%": acc10,
        "Acc@20%": acc20,
        "Acc@30%": acc30,
    }


# main
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)


# ------------------------
# Baseline A: Median
# ------------------------
train_median = float(train["total_calories"].median())

test_pred_median = test.copy()
test_pred_median["pred_calories"] = train_median
test_pred_median.to_csv(OUT_MEDIAN, index=False)

metrics_median = compute_metrics(
    test_pred_median["total_calories"].values,
    test_pred_median["pred_calories"].values
)
print("Baseline A (Median) saved:", OUT_MEDIAN)
print("Baseline A metrics:", metrics_median)

# ------------------------
# Baseline B: TF-IDF + Ridge
# ------------------------
# TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2))

X_train = vectorizer.fit_transform(train["meal_title"].astype(str).values)
y_train = train["total_calories"].values

# Ridge
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# test-set without new learning
X_test = vectorizer.transform(test["meal_title"].astype(str).values)
y_test = test["total_calories"].values
y_pred = model.predict(X_test)

metrics_tfidf = compute_metrics(y_test, y_pred)

test_pred_tfidf = test.copy()
test_pred_tfidf["pred_calories"] = y_pred
test_pred_tfidf_out = test_pred_tfidf.copy()
test_pred_tfidf_out["pred_calories"] = test_pred_tfidf_out["pred_calories"].round(2)
test_pred_tfidf_out.to_csv(OUT_TFIDF, index=False)

print("Baseline B (TF-IDF+Ridge) saved:", OUT_TFIDF)
print("Baseline B metrics:", metrics_tfidf)

# Results csv
rows = [
    {"Method": "Baseline A (Median)", "Dataset": "Nutrition5k-test", **metrics_median},
    {"Method": "Baseline B (TF-IDF+Ridge)", "Dataset": "Nutrition5k-test", **metrics_tfidf}
]

results = pd.DataFrame(rows)

metric_cols = ["MAE", "MedAE", "Acc@10%", "Acc@20%", "Acc@30%"]
for i in metric_cols:
    if i in results.columns:
        results[i] = pd.to_numeric(results[i], errors="coerce").round(2)
    else:
        results[i] = np.nan

results = results[["Method", "Dataset"] + metric_cols]
results.to_csv(OUT_RES, index=False)
print("Saved results:", OUT_RES)