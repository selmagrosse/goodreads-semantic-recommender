import gradio as gr
import pandas as pd
from src.data_access.library import Library, TAG_COLUMNS
from src.recommender.vectorstore import VectorStore

# Initialize
repo = Library()
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
        cover_path = Library.get_cover_path(b)
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
def yearly_to_read(blocks):
    to_read_books = repo.df[repo.df["to-read"] == 1].copy()
    recommendations = []

    for g in blocks:
        genre = g["genre"]
        n = g["n"]

        results = vector_store.recommend(
            text_query=genre,
            n_results=n,
        )

        for b, doc, d in zip(results["metadatas"][0], results["documents"][0], results["distances"][0]):
            cover_path = Library.get_cover_path(b)
            recommendations.append({
                "title": b["Title"],
                "author": b["Author"],
                "cover_path": cover_path,
                "score": d,
                "description": doc or ""
            })

    md = "## Your To-Read Recommendations\n\n"
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

def collect_inputs(*vals):
    blocks = []
    for i in range(0, len(vals), 2):
        genre = vals[i]
        n = vals[i+1]
        if genre and n and int(n) > 0:
            blocks.append({"genre": genre.strip(), "n": int(n)})
    if not blocks:
        return "Please add at least one genre."
    return yearly_to_read(blocks)

# -----------------------------
# UI
# -----------------------------
with gr.Blocks(title="My Book Library & Recommender") as demo:

    gr.Markdown("## Browse by tag, search semantically, or generate yearly to-read list")

    # ---- TAB 1
    with gr.Tab("Browse by Tag"):
        tag_dropdown = gr.Dropdown(choices=repo.get_available_tags(), label="Select a category")
        tag_output = gr.Markdown()
        tag_dropdown.change(show_books_by_tag, tag_dropdown, tag_output)

    # ---- TAB 2
    with gr.Tab("Semantic Search"):
        query_input = gr.Textbox(label="Type your query")
        results_output = gr.Markdown()
        gr.Button("Search").click(show_books_semantic, query_input, results_output)

    # ---- TAB 3 
    with gr.Tab("Yearly To-Read"):

        row_count = gr.State(0)

        add_btn = gr.Button("➕ Add genre")
        remove_btn = gr.Button("➖ Remove last")

        rows = []
        inputs = []

        for i in range(MAX_ROWS):
            with gr.Row(visible=False) as row:
                genre = gr.Textbox(label=f"Prompt {i+1}")
                number = gr.Number(label="How many books", precision=0)
                inputs += [genre, number]
            rows.append(row)

        generate_btn = gr.Button("Generate list")
        yearly_output = gr.Markdown()

        add_btn.click(add_row, row_count, [row_count] + rows)
        remove_btn.click(remove_row, row_count, [row_count] + rows)
        generate_btn.click(collect_inputs, inputs, yearly_output)


if __name__ == "__main__":
    demo.launch(allowed_paths=["covers/"])