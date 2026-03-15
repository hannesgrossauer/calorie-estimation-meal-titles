import os
import pandas as pd
import matplotlib.pyplot as plt

NUTRI_PATH = "data/processed/nutrition5k_master.csv"
FOOD_PATH = "data/processed/food_master.csv"

OUT_SUMMARY = "reports/audits/audit_summary.txt"
OUT_HIST_NUTRI = "reports/figures/total_calories_hist_nutrition5k.png"
OUT_BOX_NUTRI = "reports/figures/total_calories_box_nutrition5k.png"
OUT_HIST_FOOD = "reports/figures/meal_title_len_hist_food.png"


def compute_title_stats(df: pd.DataFrame, title_col: str) -> dict:
    title = df[title_col].astype(str).str.strip()
    title_len = title.str.len()
    word_count = title.str.split().str.len()

    return {
        "rows": len(df),
        "missing_title": int(df[title_col].isna().sum()),
        "title_len_min": int(title_len.min()) if len(title_len) else -1,
        "title_len_median": float(title_len.median()) if len(title_len) else float("nan"),
        "title_len_mean": float(title_len.mean()) if len(title_len) else float("nan"),
        "title_len_max": int(title_len.max()) if len(title_len) else -1,
        "word_count_min": int(word_count.min()) if len(word_count) else -1,
        "word_count_median": float(word_count.median()) if len(word_count) else float("nan"),
        "word_count_mean": float(word_count.mean()) if len(word_count) else float("nan"),
        "word_count_max": int(word_count.max()) if len(word_count) else -1,
        "one_word_titles": int((word_count < 2).sum()) if len(word_count) else 0,
    }


def compute_nutrition5k_stats(df: pd.DataFrame) -> dict:
    s = compute_title_stats(df, "meal_title")

    calories = pd.to_numeric(df["total_calories"], errors="coerce")

    comma_count = df["meal_title"].astype(str).str.count(",")
    and_count = df["meal_title"].astype(str).str.contains(r"\band\b", regex=True)
    ingredient_count = comma_count + and_count.astype(int) + 1

    s.update({
        "calories_count": int(calories.count()),
        "calories_min": float(calories.min()) if len(calories) else float("nan"),
        "calories_mean": float(calories.mean()) if len(calories) else float("nan"),
        "calories_median": float(calories.median()) if len(calories) else float("nan"),
        "calories_max": float(calories.max()) if len(calories) else float("nan"),
        "ingredient_count_min": int(ingredient_count.min()) if len(ingredient_count) else -1,
        "ingredient_count_median": float(ingredient_count.median()) if len(ingredient_count) else float("nan"),
        "ingredient_count_mean": float(ingredient_count.mean()) if len(ingredient_count) else float("nan"),
        "ingredient_count_max": int(ingredient_count.max()) if len(ingredient_count) else -1,
    })

    return s


def write_section(lines, label: str, s: dict, include_calories: bool = False, include_ingredients: bool = False):
    lines.append(label.upper())
    lines.append("-" * len(label))
    lines.append(f"rows: {s['rows']}")
    lines.append(f"missing_title: {s['missing_title']}")
    lines.append("")

    if include_calories:
        lines.append("TOTAL CALORIES")
        for k in ["calories_count", "calories_min", "calories_mean", "calories_median", "calories_max"]:
            lines.append(f"{k}: {s[k]}")
        lines.append("")

    lines.append("TITLE LENGTH (chars)")
    for k in ["title_len_min", "title_len_median", "title_len_mean", "title_len_max"]:
        lines.append(f"{k}: {s[k]}")
    lines.append("")

    lines.append("WORD COUNT")
    for k in ["word_count_min", "word_count_median", "word_count_mean", "word_count_max", "one_word_titles"]:
        lines.append(f"{k}: {s[k]}")
    lines.append("")

    if include_ingredients:
        lines.append("INGREDIENT COUNT (estimated from title)")
        for k in ["ingredient_count_min", "ingredient_count_median", "ingredient_count_mean", "ingredient_count_max"]:
            lines.append(f"{k}: {s[k]}")
        lines.append("")


def plot_total_calories(calories: pd.Series, out_hist: str, out_box: str):
    plt.figure()
    plt.hist(calories, bins=50)
    plt.title("Total Calories Distribution — Nutrition5k")
    plt.xlabel("Total calories")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_hist, dpi=200)
    plt.close()

    plt.figure()
    plt.boxplot(calories, vert=True)
    plt.title("Total Calories Box Plot — Nutrition5k")
    plt.ylabel("Total calories")
    plt.tight_layout()
    plt.savefig(out_box, dpi=200)
    plt.close()


def plot_food_title_lengths(df: pd.DataFrame, out_hist: str):
    title_len = df["meal_title"].astype(str).str.len()

    plt.figure()
    plt.hist(title_len, bins=50)
    plt.title("Meal Title Length Distribution — food_master")
    plt.xlabel("Title length (chars)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_hist, dpi=200)
    plt.close()


# main
os.makedirs("reports/audits", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

nutri_df = pd.read_csv(NUTRI_PATH)
food_df = pd.read_csv(FOOD_PATH)

nutri_stats = compute_nutrition5k_stats(nutri_df)
food_stats = compute_title_stats(food_df, "meal_title")

lines = ["DATA AUDIT SUMMARY", ""]
write_section(lines, "Nutrition5k dataset", nutri_stats, include_calories=True, include_ingredients=True)
write_section(lines, "food_master dataset", food_stats, include_calories=False, include_ingredients=False)

with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"Saved summary to: {OUT_SUMMARY}")

nutri_calories = pd.to_numeric(nutri_df["total_calories"], errors="coerce").dropna()
plot_total_calories(nutri_calories, OUT_HIST_NUTRI, OUT_BOX_NUTRI)

plot_food_title_lengths(food_df, OUT_HIST_FOOD)

print("Saved plots:")
print(f"- {OUT_HIST_NUTRI}")
print(f"- {OUT_BOX_NUTRI}")
print(f"- {OUT_HIST_FOOD}")