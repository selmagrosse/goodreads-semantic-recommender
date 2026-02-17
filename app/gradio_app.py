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

# Yearly To-Read Recommender
def yearly_to_read(genre1, n1, genre2, n2):
    # Filter only to-read books
    to_read_books = repo.df[repo.df["to-read"] == 1].copy()
    recommendations = []

    for genre, n in [(genre1, n1), (genre2, n2)]:
        if not genre.strip() or n <= 0:
            continue
        results = vector_store.recommend(text_query=genre, n_results=int(n))
        for b, doc, d in zip(results["metadatas"][0], results["documents"][0], results["distances"][0]):
            isbn = b.get("ISBN", "")
            cover_url = repo.get_cover_url(isbn)
            recommendations.append({
                "title": b["Title"],
                "author": b["Author"],
                "cover_url": cover_url,
                "score": d,
                "description": doc or ""
            })

    # Format Markdown
    md = "## Your To-Read Recommendations\n\n"
    for b in recommendations:
        md += f"### {b['title']} — {b['author']} (score: {b['score']:.3f})\n"
        md += f"![cover]({b['cover_url']})\n\n"
        md += f"{b['description'][:300]}...\n\n"
        md += "---\n"
    return md

# Gradio UI
with gr.Blocks(title="My Book Library & Recommender") as demo:
    gr.Markdown("## Browse by tag, search semantically, or generate yearly to-read list")

    # Tab 1: Browse by tag
    with gr.Tab("Browse by Tag"):
        tag_dropdown = gr.Dropdown(choices=repo.get_available_tags(), label="Select a category")
        tag_output = gr.Markdown()
        tag_dropdown.change(show_books_by_tag, inputs=tag_dropdown, outputs=tag_output)

    # Tab 2: Semantic search
    with gr.Tab("Semantic Search"):
        query_input = gr.Textbox(label="Type your query")
        results_output = gr.Markdown()
        query_button = gr.Button("Search")
        query_button.click(show_books_semantic, inputs=query_input, outputs=results_output)

    # Tab 3: Yearly To-Read
    with gr.Tab("Yearly To-Read"):
        with gr.Row():
            genre1_input = gr.Textbox(label="Genre 1 (topic/genre)")
            n1_input = gr.Number(label="Number of books for Genre 1", value=20, precision=0)
        with gr.Row():
            genre2_input = gr.Textbox(label="Genre 2 (topic/genre)")
            n2_input = gr.Number(label="Number of books for Genre 2", value=20, precision=0)
        generate_btn = gr.Button("Generate")
        yearly_output = gr.Markdown()
        generate_btn.click(yearly_to_read, inputs=[genre1_input, n1_input, genre2_input, n2_input], outputs=yearly_output)

if __name__ == "__main__":
    demo.launch()