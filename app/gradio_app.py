import gradio as gr
import pandas as pd
from src.data_access.library import Library, TAG_COLUMNS
from src.recommender.vectorstore import VectorStore

repo = Library()
vector_store = VectorStore()

# Helper to show active tags
def format_tags(tags_dict):
    active = [tag for tag in TAG_COLUMNS if tags_dict.get(tag)]
    return ", ".join(active) if active else "No tags"

# Tag-based browsing function
def show_books_by_tag(tag):
    books = repo.get_books_by_tag(tag)

    if books.empty:
        return f"No books found for '{tag}'"

    md = f"# {tag.capitalize()} books\n\n"

    for _, b in books.iterrows():
        md += f"### {b['Title']} — {b['Author']}\n"
        isbn = b.get("ISBN", "")
        cover_url = Library.get_cover_url(isbn)
        md += f"![cover]({cover_url})\n\n"
        md += f"My rating: {b['My Rating']} | Avg: {b['Average Rating']}\n\n"
        md += f"Tags: {format_tags(b)}\n\n"

        if b["Final Description"]:
            md += f"{b['Final Description'][:300]}...\n\n"

        md += "---\n"

    return md

# Semantic search function
def show_books_semantic(query, n_results=5):
    results = vector_store.recommend(query, n_results=n_results)
    books = results["metadatas"][0]
    docs = results["documents"][0]
    distances = results["distances"][0]

    if not books:
        return f"No books found for '{query}'"

    md = f"# Recommendations for '{query}'\n\n"
    for b, doc, d in zip(books, docs, distances):
        md += f"### {b['Title']} — {b['Author']} (score: {d:.3f})\n"
        isbn = b.get("ISBN", "")
        cover_url = Library.get_cover_url(isbn)
        md += f"![cover]({cover_url})\n\n"
        md += f"My rating: {b['My Rating']} | Avg: {b['Average Rating']}\n\n"
        md += f"Tags: {format_tags(b)}\n\n"
        if doc:
            md += f"{doc[:300]}...\n\n"
        md += "---\n"
    return md

# Gradio UI
with gr.Blocks(title="My Book Library & Recommender") as demo:
    gr.Markdown("## Browse by tag or search semantically")

    with gr.Tab("Browse by Tag"):
        tag_dropdown = gr.Dropdown(
            choices=repo.get_available_tags(),
            label="Select a category"
        )
        tag_output = gr.Markdown()
        tag_dropdown.change(show_books_by_tag, inputs=tag_dropdown, outputs=tag_output)

    with gr.Tab("Semantic Search"):
        query_input = gr.Textbox(label="Type your query")
        results_output = gr.Markdown()
        query_button = gr.Button("Search")
        query_button.click(show_books_semantic, inputs=query_input, outputs=results_output)

if __name__ == "__main__":
    demo.launch()