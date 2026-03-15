# Estimating Calories from Short Meal Titles Using a Large Language Model (Bachelor Thesis)

This repository contains the code for the practical part of the bachelor thesis on **calorie estimation from short meal titles**.

The main task is to estimate the **total calories of a dish** from a short meal title using:
- baseline models
- LLM-based prediction with two variants:
  - **direct**: meal title → total calories
  - **pipeline**: meal title → ingredients → total calories

The main evaluation is based on a **Nutrition5k-derived dataset**.  
In addition, the LLM-based approaches are analyzed on a **Reddit-based dataset** of real-world meal titles.

## Main Files

- `src/preprocess_data.py` – preprocesses the dataset
- `src/data_audits.py` – creates dataset summaries and plots
- `src/split_data.py` – creates train/test splits
- `src/pred_baseline.py` – runs the baseline models
- `src/llm_runner.py` – runs the LLM-based approaches
- `src/run_all.py` – runs the full pipeline