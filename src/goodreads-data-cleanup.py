import pandas as pd

# Load Goodreads dataset
goodreads_df = pd.read_csv('data/goodreads_library.csv')

# Keep Title, Author, category, rating, and average rating columns
goodreads_df = goodreads_df[['Title', 'Author', 'Bookshelves', 'My Rating', 'Average Rating']]

# Fill missing values
goodreads_df = goodreads_df.fillna({
    'Bookshelves': '',
    'My Rating': 0, 
    'Average Rating': 0.0
})

# Add source
goodreads_df['source'] = 'Goodreads'

# Save cleaned dataset
goodreads_df.to_csv('data/cleaned_goodreads_library.csv', index=False, encoding='utf-8')
