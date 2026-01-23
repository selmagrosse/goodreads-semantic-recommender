"""
Cleans raw Goodreads export data into a structured dataset for recommendation and analysis.

Steps:
- Select relevant columns
- Add 'Has Page Count' indicator
- Merge publication year columns
- Merge 'Bookshelves' and 'Exclusive Shelf' data
- Remove duplicates based on Title and Author
- Clean ISBN metadata
- Add 'Bookshelves Clean' column
- Normalize bookshelf tags
- One-hot encode tags
- Delete original 'Bookshelves' column
- Save cleaned dataset
"""

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def main():
    # Load Goodreads dataset
    df = pd.read_csv('data/raw/goodreads_library.csv')

    # Keep Title, Author, ISBNs, number of pages, year published, category, rating, and average rating columns
    df = df[['Title', 'Author', 'ISBN', 'ISBN13', 'Number of Pages', 'Year Published', 'Original Publication Year', 'Bookshelves', 'Exclusive Shelf', 'My Rating', 'Average Rating']]

    # Add 'Has Page Count' column
    df['Number of Pages'] = pd.to_numeric(df['Number of Pages'], errors='coerce')   
    df['Has Page Count'] = df['Number of Pages'].notna().astype(int)

    # Merge 'Year Published' into 'Original Publication Year' ONLY when "Original Publication Year" is NaN
    df['Original Publication Year'] = df['Original Publication Year'].fillna(df['Year Published'])
    # Drop 'Year Published' column
    df = df.drop(columns=['Year Published'])

    # 'Bookshelves' column has NaNs where 'Exclusive Shelf' is 'read'
    # Fill NaNs in 'Bookshelves' with 'read' if 'Exclusive Shelf' is 'read'
    df.loc[(df['Bookshelves'].isna()) & (df['Exclusive Shelf'] == 'read'), 'Bookshelves'] = 'read'

    # Remove duplicates
    df = (df.sort_values(by=["Bookshelves", "My Rating"],
                                            ascending=[True, True]
                                            ).drop_duplicates(subset=["Title", "Author"], keep="first")
    )

    # Clean ISBN and ISBN13 columns
    df["ISBN"] = df["ISBN"].astype(str).str.strip().str.replace('=', '', regex=False).str.replace('"', '', regex=False)
    df["ISBN13"] = df["ISBN13"].astype(str).str.strip().str.replace('=', '', regex=False).str.replace('"', '', regex=False)
    # Replace empty strings in ISBN and ISBN13 with NaN
    df["ISBN"] = df["ISBN"].replace({"": pd.NA, "nan": pd.NA})
    df["ISBN13"] = df["ISBN13"].replace({"": pd.NA, "nan": pd.NA})

    # Clean up tags in 'Bookshelves' column
    df['Bookshelves Clean'] = (df['Bookshelves'].str.lower().str.strip().str.replace(r"\s*,\s*", ",", regex=True))

    # Create 'Tag List' column
    df["Tag List"] = df["Bookshelves Clean"].apply(lambda x: x.split(",") if x else [])

    # One-hot encode the tags
    mlb = MultiLabelBinarizer()
    tag_matrix = mlb.fit_transform(df['Tag List'])
    tag_df = pd.DataFrame(tag_matrix, columns=mlb.classes_, index=df.index)
    # Combine the original DataFrame with the tag matrix
    df = pd.concat([df, tag_df], axis=1)

    # Drop 'Bookshelves' column
    df = df.drop(columns=['Bookshelves'])

    # Save cleaned dataset
    df.to_csv('data/interim/goodreads_library_cleaned.csv', index=False)

if __name__ == "__main__":
    main()