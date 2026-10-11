
"""Regenerate the preliminary hyperspectral Raman screening CSV."""

from pathlib import Path

import pandas as pd

from src.relevance import screen_papers


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "hyperspectral_candidates.csv"
OUTPUT = ROOT / "data" / "hyperspectral_screening.csv"


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Candidate file not found: {INPUT}")

    source = pd.read_csv(INPUT).fillna("")
    results = pd.DataFrame(screen_papers(source.to_dict("records")))
    results.to_csv(OUTPUT, index=False)

    print(f"Input papers: {len(source)}")
    print(f"Output file: {OUTPUT}")
    print("\nClassification counts:")
    print(results["relevance_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
