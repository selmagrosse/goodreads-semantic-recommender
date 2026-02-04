"""
Select the best available book description based on quality scores.

This script:
- Compares Google, OpenLibrary, and LLM descriptions per book
- Selects the description with the highest quality score
- Writes the selected text to a new column: 'Final Description'
- Optionally records the winning source in 'Final Description Source'
"""

import pandas as pd

INPUT_PATH = "data/interim/goodreads_library_descriptions_with_llm.csv"
OUTPUT_PATH = "data/processed/goodreads_library_final_descriptions.csv"

DESCRIPTION_SOURCES = [
    {
        "source": "Google",
        "desc_col": "Description Google",
        "score_col": "Quality Score Google",
    },
    {
        "source": "OpenLibrary",
        "desc_col": "Description OpenLibrary",
        "score_col": "Quality Score OpenLibrary",
    },
    {
        "source": "LLM",
        "desc_col": "Description LLM",
        "score_col": "Quality Score LLM",
    },
]

def select_best_description(row):
    """
    Select the best description for a single row based on quality score.
    Returns the description, its score, and its source.
    """
    candidates = []

    for src in DESCRIPTION_SOURCES:
        desc = row.get(src["desc_col"])
        score = row.get(src["score_col"], 0)

        if pd.notna(desc) and isinstance(desc, str) and desc.strip():
            candidates.append((score, desc, src["source"]))

    if not candidates:
        return pd.Series(
            {
                "Final Description": None,
                "Final Description Source": None,
                "Final Description Score": None,
            }
        )

    # Highest score wins; ties resolved by source order above
    candidates.sort(key=lambda x: x[0], reverse=True)

    best_score, best_desc, best_source = candidates[0]

    return pd.Series(
        {
            "Final Description": best_desc,
            "Final Description Source": best_source,
            "Final Description Score": best_score,
        }
    )

def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} books")

    result = df.apply(select_best_description, axis=1)
    df = pd.concat([df, result], axis=1)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved dataset with final descriptions to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()