"""
This script loads a cleaned Goodreads dataset containing fetched book descriptions
from external sources (Google Books and OpenLibrary), evaluates the quality of
these descriptions using a language model (LLM), and saves the results to a CSV.

Workflow:
1. Load the input CSV containing fetched book descriptions.
2. Evaluate the specified description columns using the `evaluate_descriptions` function.
   - Each description is normalized and scored on a scale from 1 (very bad) to 10 (excellent).
3. Save the resulting dataset with new quality score columns to an output CSV.

Columns to evaluate are specified in the `description_cols` dictionary:
    - Keys: description column names in the dataset
    - Values: names of the corresponding output quality score columns
"""

import pandas as pd
from evaluate_book_descriptions import evaluate_descriptions

def main():
    input_file = "data/interim/goodreads_library_descriptions_cleaned.csv"
    output_file = "data/interim/goodreads_library_descriptions_quality_scored.csv"
    description_cols = {
        "Description Google": "Quality Score Google",
        "Description OpenLibrary": "Quality Score OpenLibrary"
    }
    print("Loading data...")
    df = pd.read_csv(input_file)
    df = evaluate_descriptions(df, description_cols)

    # Save
    df.to_csv(output_file, index=False)
    print(f"\nSaved scored dataset to {output_file}")
if __name__ == "__main__":
    main()