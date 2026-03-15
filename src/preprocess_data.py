import pandas as pd
import re
import csv

# ---PREPROCESS OF REDDIT DATASET "food_posts_calories.csv.zip"---

INPUT_PATH = "data/raw/food_posts_calories.csv.zip"
OUTPUT_PATH = "data/processed/food_master.csv"

# thresholds
MIN_ENERGY = 10
MIN_WORDS = 2

# Load Data
df = pd.read_csv(INPUT_PATH, usecols=["cleaned_title", "Energy"])

# Clean Titles
df["meal_title"] = df["cleaned_title"].astype(str).str.strip()
df.loc[df["meal_title"].isin(["", "nan", "None", "null"]), "meal_title"] = pd.NA

# Convert Energy to numeric
df["Energy"] = pd.to_numeric(df["Energy"], errors="coerce")

# Drop unrealistic/unusable rows
df = df.dropna(subset=["meal_title", "Energy"]).copy()

# word count
word_count = df["meal_title"].str.split().str.len()

# Filter invalid chars
token_re = re.compile(r"[^\W_]+", flags=re.UNICODE)
valid_token_count = df["meal_title"].str.findall(token_re).str.len()

# Apply filters
keep = (df["Energy"] >= MIN_ENERGY) & (word_count >= MIN_WORDS) & (valid_token_count >= MIN_WORDS)
df = df.loc[keep, ["meal_title", "Energy"]].reset_index(drop=True)

# Save
df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved {len(df)} rows to {OUTPUT_PATH}")




# ---PREPROCESS OF GOOGLE DATASET "nutrition5k_dataset_metadata_dish_metadata_cafe1.csv"---

# output format:
# total_calories, top1_name, top2_name, top3_name, top4_name, top5_name

INPUT_FILE = "data/raw/nutrition5k_dataset_metadata_dish_metadata_cafe1.csv"
OUTPUT_FILE = "data/processed/nutrition5k_master.csv"

seen_ingredient_sets = set()

with open(INPUT_FILE, "r", newline="", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # write header
    writer.writerow(["total_calories", "meal_title"])

    for row in reader:
        # delete row if first ingredient name is "deprecated"
        if row[7] == "deprecated":
            continue

        total_calories = float(row[1])

        # filter for min energy and max energy
        if total_calories < 30 or total_calories > 1500:
            continue

        ingredients = []
        ingredient_names = []

        i = 6
        while i + 6 < len(row):
            ingr_name = row[i + 1].strip()
            ingredient_names.append(ingr_name)

            try:
                ingr_calories = float(row[i + 3])
            except ValueError:
                ingr_calories = 0.0

            # keep only ingredient with more than 5% of total_calories
            if total_calories > 0 and ingr_calories > 0.05 * total_calories and ingr_name:
                ingredients.append((ingr_name, ingr_calories))

            i += 7

        # remove rows that have exactly the same ingredients
        ingredient_key = tuple(sorted(ingredient_names))

        if ingredient_key in seen_ingredient_sets:
            continue

        seen_ingredient_sets.add(ingredient_key)

        # sort by ingredient calories descending
        ingredients.sort(key=lambda x: x[1], reverse=True)

        # keep only the top 5 names
        top_names = [name for name, cal in ingredients[:5]]

        # skip rows where no ingredient passed the filter
        if len(top_names) == 0:
            continue

        # build meal title
        if len(top_names) == 1:
            meal_title = top_names[0]
        elif len(top_names) == 2:
            meal_title = f"{top_names[0]} and {top_names[1]}"
        else:
            meal_title = ", ".join(top_names[:-1]) + f" and {top_names[-1]}"

        writer.writerow([total_calories, meal_title])

print("Preprocessing done. Saved to", OUTPUT_FILE)

