import gradio as gr
import pandas as pd
from src.data_access.library import Library, TAG_COLUMNS
from src.recommender.vectorstore import VectorStore
from src.filtering.metadata_filter import apply_dynamic_filters

# Metadata definitions
NUMERIC_COLUMNS = [
    "Original Publication Year",
    "Number of Pages",
    "My Rating",
    "Average Rating",
    "Final Description Score",
]

CATEGORICAL_COLUMNS = [
    "Final Description Source"
]

# -----------------------------
# Initialize
# -----------------------------
repo = Library()
for tag in TAG_COLUMNS:
    repo.df[tag] = pd.to_numeric(repo.df[tag], errors="coerce").fillna(0).astype(int)
vector_store = VectorStore()

# -----------------------------
# Helpers
# -----------------------------
def format_tags(tags_dict):
    active = [tag for tag in TAG_COLUMNS if tags_dict.get(tag)]
    return ", ".join(active) if active else "No tags"

# -----------------------------
# Browse by tag
# -----------------------------
def show_books_by_tag(tag):
    books = repo.get_books_by_tag(tag)
    if books.empty:
        return f"No books found for '{tag}'"
    md = f"# {tag.capitalize()} books\n\n"
    for _, b in books.iterrows():
        #cover_path = Library.get_cover_path(b)
        cover_path = Library.get_cover_url(b.get("ISBN", ""))
        md += f"### {b['Title']} — {b['Author']}\n"
        md += f"![cover]({cover_path})\n\n"
        md += f"My rating: {b['My Rating']} | Avg: {b['Average Rating']}\n\n"
        md += f"Tags: {format_tags(b)}\n\n"
        md += f"{b['Final Description']}\n\n---\n"
    return md

# -----------------------------
# Semantic search
# -----------------------------
def show_books_semantic(query, n_results=5):
    results = vector_store.recommend(query, n_results=n_results)
    books = results["metadatas"][0]
    docs = results["documents"][0]
    distances = results["distances"][0]

    if not books:
        return f"No books found for '{query}'"

    md = f"# Recommendations for '{query}'\n\n"
    for b, doc, d in zip(books, docs, distances):
        cover_path = Library.get_cover_path(b)
        cover_path = Library.get_cover_url(b.get("ISBN", ""))
        md += f"### {b['Title']} — {b['Author']} (score: {d:.3f})\n"
        md += f"![cover]({cover_path})\n\n"
        md += f"My rating: {b['My Rating']} | Avg: {b['Average Rating']}\n\n"
        md += f"Tags: {format_tags(b)}\n\n"
        if doc:
            md += doc + "\n\n"
        md += "---\n"
    return md

# -----------------------------
# Yearly To-Read logic
# -----------------------------
def build_filter_dict(filter_inputs):
    """
    Convert Gradio input components into a filter dictionary for apply_dynamic_filters.
    Only include filters that the user actually set.
    """
    filters = {}

    for item in filter_inputs:
        col = item[0]

        # Numeric: (col, min_input, max_input)
        if len(item) == 3:
            min_val = item[1]
            max_val = item[2]
            condition = {}
            try:
                if min_val is not None and min_val != "":
                    condition["min"] = float(min_val)
                if max_val is not None and max_val != "":
                    condition["max"] = float(max_val)
            except ValueError:
                continue  # ignore invalid numeric input
            if condition:  # Only add if at least min or max is set
                filters[col] = condition

        # Boolean tag: (col, checkbox)
        elif isinstance(item[1], bool):
            filters[col] = item[1]

        # Text: (col, textbox)
        else:
            val = item[1]
            if val not in [None, ""]:
                filters[col] = val

    return filters

def yearly_to_read(blocks):
    full_df = repo.df.copy()  # start with all books
    recommendations = []

    for g in blocks:
        genre = g["genre"]
        n = g["n"]
        filters = build_filter_dict(g["filters"].items())
        use_seed = g.get("use_seed", True)
        print(f"Processing genre '{genre}' with filters: {filters}")

        # Apply metadata filters (including "to-read" if selected)
        df_filtered = apply_dynamic_filters(full_df, filters)
        if df_filtered.empty:
            continue

        # Extract valid ISBNs for matching
        valid_isbns = set(df_filtered["ISBN13"].dropna().astype(str))

        # Semantic recommendations (fetch more than n to allow filtering)
        results = vector_store.recommend(text_query=genre, repo=repo, use_seed=use_seed, n_results=50)

        count = 0
        for b, doc, d in zip(results["metadatas"][0], results["documents"][0], results["distances"][0]):
            book_isbn = str(b.get("ISBN13", "")).strip()
            if book_isbn in valid_isbns:
                cover_path = Library.get_cover_path(b)
                cover_path = Library.get_cover_url(b.get("ISBN", ""))
                recommendations.append({
                    "title": b["Title"],
                    "author": b["Author"],
                    "cover_path": cover_path,
                    "score": d,
                    "description": doc or ""
                })
                count += 1
                if count >= n:  # only top N per genre
                    break

    if not recommendations:
        return "No books match your filters."

    md = "## Your Yearly Recommendations\n\n"
    for b in recommendations:
        md += f"### {b['title']} — {b['author']} (score: {b['score']:.3f})\n"
        md += f"![cover]({b['cover_path']})\n\n"
        md += f"{b['description']}\n\n---\n"

    return md
# -----------------------------
# Dynamic rows logic
# -----------------------------
MAX_ROWS = 8

def update_rows(count):
    updates = [count]
    for i in range(MAX_ROWS):
        updates.append(gr.update(visible=i < count))
    return updates

def add_row(count):
    return update_rows(min(count + 1, MAX_ROWS))

def remove_row(count):
    return update_rows(max(count - 1, 0))

# -----------------------------
# UI
# -----------------------------
with gr.Blocks(title="My Book Library & Recommender") as demo:

    gr.Markdown("## Browse by tag, search semantically, or generate yearly to-read list")

    # ---- TAB 1: Browse by Tag
    with gr.Tab("Browse by Tag"):
        tag_dropdown = gr.Dropdown(choices=repo.get_available_tags(), label="Select a category")
        tag_output = gr.Markdown()
        tag_dropdown.change(show_books_by_tag, tag_dropdown, tag_output)

    # ---- TAB 2: Semantic Search
    with gr.Tab("Semantic Search"):
        query_input = gr.Textbox(label="Type your query")
        results_output = gr.Markdown()
        gr.Button("Search").click(show_books_semantic, query_input, results_output)

    # ---- TAB 3: Yearly To-Read
    with gr.Tab("Yearly To-Read"):

        row_count = gr.State(0)
        add_btn = gr.Button("➕ Add genre")
        remove_btn = gr.Button("➖ Remove last")
        generate_btn = gr.Button("Generate list")
        yearly_output = gr.Markdown()

        rows = []
        inputs = []

        for i in range(MAX_ROWS):
            with gr.Row(visible=False) as row:
                # Always visible
                genre = gr.Textbox(label=f"Prompt {i+1}")
                number = gr.Number(label="How many books", precision=0, value=None)
                use_seed = gr.Checkbox(label="Use my taste profile", value=True)

                # Collapsible metadata filters
                with gr.Accordion(label="Optional metadata filters", open=False):
                    
                    # Numeric filters
                    numeric_inputs = []
                    for col in NUMERIC_COLUMNS:
                        min_input = gr.Textbox(label=f"{col} min", placeholder="Leave empty if no filter")
                        max_input = gr.Textbox(label=f"{col} max", placeholder="Leave empty if no filter")
                        numeric_inputs.extend([min_input, max_input])
                    
                    # Tag filters
                    tag_inputs = []
                    for tag in TAG_COLUMNS:
                        tag_inputs.append(gr.Checkbox(label=tag))
                    
                    # Categorical filters
                    cat_inputs = []
                    for col in CATEGORICAL_COLUMNS:
                        cat_inputs.append(gr.Textbox(label=col, placeholder="Leave empty if no filter"))

                # Collect all inputs for this row
                row_inputs = [genre, number, use_seed] + numeric_inputs + tag_inputs + cat_inputs
                inputs += row_inputs
                rows.append(row)

        # Buttons
        add_btn.click(add_row, row_count, [row_count] + rows)
        remove_btn.click(remove_row, row_count, [row_count] + rows)

        # Generate function
        def collect_inputs_with_metadata(*vals):
            blocks = []
            step = 3 + len(NUMERIC_COLUMNS)*2 + len(TAG_COLUMNS) + len(CATEGORICAL_COLUMNS)
            for i in range(0, len(vals), step):
                genre = vals[i]
                n = vals[i+1]
                use_seed = vals[i+2]
                if not genre or not n or int(n) <= 0:
                    continue

                filters = {}

                # Numeric filters
                for j, col in enumerate(NUMERIC_COLUMNS):
                    min_val = vals[i + 3 + j*2]
                    max_val = vals[i + 3 + j*2 + 1]
                    cond = {}
                    if min_val not in [None, ""]:
                        try:
                            cond["min"] = float(min_val)
                        except ValueError:
                            pass
                    if max_val not in [None, ""]:
                        try:
                            cond["max"] = float(max_val)
                        except ValueError:
                            pass
                    if cond:
                        filters[col] = cond

                # Tag filters
                tag_start = i + 3 + len(NUMERIC_COLUMNS)*2
                for k, tag in enumerate(TAG_COLUMNS):
                    val = vals[tag_start + k]
                    if val is True:           # only include if explicitly checked
                        filters[tag] = True

                # Categorical filters
                cat_start = tag_start + len(TAG_COLUMNS)
                for l, col in enumerate(CATEGORICAL_COLUMNS):
                    val = vals[cat_start + l]
                    if val:
                        filters[col] = val

                blocks.append({"genre": genre.strip(), "n": int(n), "use_seed": use_seed, "filters": filters})

            if not blocks:
                return "Please add at least one genre."
            return yearly_to_read(blocks)

        generate_btn.click(collect_inputs_with_metadata, inputs, yearly_output)

# -----------------------------
# Launch
# -----------------------------
if __name__ == "__main__":
    demo.launch(allowed_paths=["covers/"])