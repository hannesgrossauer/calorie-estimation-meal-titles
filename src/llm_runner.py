import os
import json
import argparse
from typing import Any, Dict

import pandas as pd
import numpy as np
from openai import OpenAI

MODEL = "gpt-5-mini-2025-08-07"

PROMPTS_DIR = "configs/prompts"
SCHEMAS_DIR = "configs/schemas"

OUT_PATH = "data/processed/results.csv"


# Metrics
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    # just to be sure:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    abs_err = np.abs(y_pred - y_true)
    mae = float(np.mean(abs_err))
    medae = float(np.median(abs_err))

    acc10 = float(np.mean(abs_err <= (0.10 * y_true)) * 100.0)
    acc20 = float(np.mean(abs_err <= (0.20 * y_true)) * 100.0)
    acc30 = float(np.mean(abs_err <= (0.30 * y_true)) * 100.0)

    return {
        "MAE": mae,
        "MedAE": medae,
        "Acc@10%": acc10,
        "Acc@20%": acc20,
        "Acc@30%": acc30,
    }



# helpers
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# OpenAI call
def call_llm(client: OpenAI, prompt: str, schema: Dict[str, Any], schema_name: str):
    resp = client.responses.create(
        model=MODEL,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
    )
    data = json.loads(resp.output_text)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to CSV")
    parser.add_argument("--variant", required=True, choices=["direct", "pipeline"])
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing environment variable 'OPENAI_API_KEY'")

    client = OpenAI(api_key=api_key)

    # Load prompts and schemas
    if args.variant == "direct":
        prompt_direct = load_text(os.path.join(PROMPTS_DIR, "direct_cal.txt"))
        schema_direct = load_json(os.path.join(SCHEMAS_DIR, "schema_direct_calories.json"))
    else:
        prompt_ing = load_text(os.path.join(PROMPTS_DIR, "ingredients.txt"))
        prompt_cal = load_text(os.path.join(PROMPTS_DIR, "calories_from_ingredients.txt"))
        schema_ing = load_json(os.path.join(SCHEMAS_DIR, "schema_ingredients.json"))
        schema_cal = load_json(os.path.join(SCHEMAS_DIR, "schema_calories_from_ingredients.json"))

    # Data
    df = pd.read_csv(args.input)

    if "meal_title" not in df.columns:
        raise RuntimeError("Missing required column 'meal_title'.")

    df["meal_title"] = df["meal_title"].astype(str).str.strip()

    if args.limit and args.limit > 0 and len(df) > args.limit:
        df = df.sample(n=args.limit, random_state=args.seed).reset_index(drop=True)

    # Target column only exists for Nutrition5k
    target_col = "total_calories" if "total_calories" in df.columns else None

    # Main loop
    # Print updates
    counter = 1

    # Save Path
    input_name = os.path.splitext(os.path.basename(args.input))[0]
    pred_path = f"data/processed/predictions_llm_{args.variant}_{input_name}.csv"

    for _, row in df.iterrows():
        title = row["meal_title"]

        if args.variant == "direct":
            prompt = f"{prompt_direct}\n\nMeal title: {title}\n"
            out = call_llm(client, prompt, schema_direct, "direct_total_calories")

            pred_row = {
                "meal_title": title,
                "pred_calories": float(out["total_calories"]),
            }

            if target_col is not None:
                pred_row[target_col] = row[target_col]

            pd.DataFrame([pred_row]).to_csv(
                pred_path,
                mode="a",
                index=False,
                header=not os.path.exists(pred_path),
            )

        else:
            p1 = f"{prompt_ing}\n\nMeal title: {title}\n"
            ing = call_llm(client, p1, schema_ing, "ingredients")

            p2 = (
                    f"{prompt_cal}\n\n"
                    "Ingredients JSON:\n"
                    + json.dumps(ing, ensure_ascii=False)
                    + "\n"
            )
            kcal = call_llm(client, p2, schema_cal, "calories_from_ingredients")

            pred_row = {
                "meal_title": title,
                "pred_calories": float(kcal["total_kcal"]),
            }

            if target_col is not None:
                pred_row[target_col] = row[target_col]

            pred_row["ingredients"] = json.dumps(ing.get("ingredients", []), ensure_ascii=False)

            pd.DataFrame([pred_row]).to_csv(
                pred_path,
                mode="a",
                index=False,
                header=not os.path.exists(pred_path),
            )

        if counter % 5 == 0:
            if args.limit == 0:
                limit = len(df)
            else:
                limit = args.limit
            print(f"Meal {counter}/{limit}")
        counter += 1

    # Save metrics only if target exists
    pred_df = pd.read_csv(pred_path)
    if target_col is not None:
        metric = compute_metrics(pred_df[target_col].values, pred_df["pred_calories"].values)

        if args.variant == "direct":
            method_name = "LLM (direct total calories)"
        else:
            method_name = "LLM (ingredients->total calories)"

        if "test" in args.input:
            dataset_name = "Nutrition5k-test"
        elif "train" in args.input:
            dataset_name = "Nutrition5k-train"
        else:
            dataset_name = input_name

        rows = [
            {"Method": method_name, "Dataset": dataset_name, **metric}
        ]

        results = pd.DataFrame(rows)

        metric_cols = ["MAE", "MedAE", "Acc@10%", "Acc@20%", "Acc@30%"]
        for i in metric_cols:
            if i in results.columns:
                results[i] = pd.to_numeric(results[i], errors="coerce").round(2)
            else:
                results[i] = np.nan

        results = results[["Method", "Dataset"] + metric_cols]

        results.to_csv(OUT_PATH, mode="a", index=False, header=not os.path.exists(OUT_PATH), encoding="utf-8",)

        print("Saved results:", OUT_PATH)


if __name__ == "__main__":
    main()