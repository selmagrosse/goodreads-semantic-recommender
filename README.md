# Book Recommender

A personal book recommender built on a Goodreads library dataset. It enriches book descriptions from multiple sources, generates sentence embeddings, and uses a Gradio UI for browsing and discovery.

## Features

- **Browse by tag** — filter your library by personal tags (feminism, sci-fi, travel, etc.)
- **Semantic search** — find books by description
- **Yearly to-read list** — generate a reading plan by genre with optional metadata filters (year, average rating, pages) and a taste-aware seed that weights results toward books you've rated highly

## How it works

Goodreads CSV export → cleaning → description enrichment (Google Books → OpenLibrary → GPT-4o-mini fallback, scored by quality) → sentence embeddings → Chroma vector store → Gradio app.

Recommendations blend a text query embedding with a mean embedding of your 4+ star books, so results reflect both what you're searching for and what you tend to like.

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your OpenAI key**
   ```bash
   echo "OPENAI_API_KEY=your-key-here" > .env
   ```

3. **Export your Goodreads library** as CSV and place it in `data/raw/`

4. **Run the pipeline** (in order)
   ```bash
   python -m src.data_cleaning
   python -m src.fetch_descriptions
   python -m src.evaluate_fetched_descriptions
   python -m src.generate_descriptions_llm
   python -m src.select_final_description
   python -m src.ingest_embeddings
   ```

5. **Launch the app**
   ```bash
   python app/gradio_app.py
   ```

## Project structure

```
src/
  data_access/        # Library class — loads CSV, manages tags, computes taste seed
  recommender/        # VectorStore wrapper around Chroma
  filtering/          # Metadata filter logic
  data_cleaning.py    # Cleans raw Goodreads export
  fetch_descriptions.py         # Fetches descriptions from Google Books + OpenLibrary
  evaluate_fetched_descriptions.py  # Scores description quality
  generate_descriptions_llm.py  # GPT fallback for missing/low-quality descriptions
  select_final_description.py   # Picks best description per book
  ingest_embeddings.py          # Generates embeddings and loads into Chroma
app/
  gradio_app.py       # UI
data/
  raw/                # Goodreads export
  interim/            # Pipeline intermediates
  processed/          # Final dataset
vectorstore/          # Persistent Chroma database
```
