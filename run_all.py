import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def run(*args: str) -> None:
    print("\n$", " ".join(args))
    subprocess.run([sys.executable, *args], cwd=REPO_ROOT, check=True)


def delete_file(path: str) -> None:
    file_path = REPO_ROOT / path
    if file_path.exists():
        file_path.unlink()
        print(f"Deleted: {path}")


def main() -> None:
    # Delete old generated files
    delete_file("data/processed/food_master.csv")
    delete_file("data/processed/nutrition5k_master.csv")
    delete_file("data/processed/train_set.csv")
    delete_file("data/processed/test_set.csv")
    delete_file("data/processed/results.csv")
    delete_file("data/processed/predictions_llm_direct_test_set.csv")
    delete_file("data/processed/predictions_llm_pipeline_test_set.csv")

    run("src/preprocess_data.py")
    run("src/data_audits.py")
    run("src/split_data.py")
    run("src/pred_baseline.py")
    run("src/llm_runner.py", "--input", "data/processed/food_master.csv", "--variant", "direct", "--limit", "1000")
    run("src/llm_runner.py", "--input", "data/processed/food_master.csv", "--variant", "pipeline", "--limit", "1000")

    print("\nAll steps completed.")


if __name__ == "__main__":
    main()