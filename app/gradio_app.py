import gradio as gr
from src.data_access.library import Library

repo = Library()

def show_books(tag):
    books = repo.get_books_by_tag(tag)

    if books.empty:
        return f"No books found for '{tag}'"

    md = f"# {tag.capitalize()} books\n\n"

    for _, b in books.iterrows():
        md += f"### {b['Title']} — {b['Author']}\n"
        md += f"![cover]({b['Cover URL']})\n\n"
        md += f"My rating: {b['My Rating']} | Avg: {b['Average Rating']}\n\n"

        if b["Final Description"]:
            md += f"{b['Final Description'][:300]}...\n\n"

        md += "---\n"

    return md


demo = gr.Interface(
    fn=show_books,
    inputs=gr.Dropdown(choices=repo.get_available_tags(), label="Select a category"),
    outputs=gr.Markdown(),
    title="My Book Library",
    description="Browse my books by category"
)

if __name__ == "__main__":
    demo.launch()