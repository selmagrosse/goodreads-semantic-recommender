import gradio as gr

# Dummy mapping and recommend function for demonstration
# Replace these with your actual mapping and index
mapping = {
    "1": {"title": "Sapiens", "tags": ["history", "non-fiction"]},
    "2": {"title": "Meditations", "tags": ["philosophy", "classic"]},
    "3": {"title": "1984", "tags": ["fiction", "dystopia"]},
}

def recommend_books(query_text, index=None, mapping=None):
    """
    Dummy recommendation function.
    """
    # Simple keyword match example
    query_keywords = query_text.lower().split()
    results = []
    for book_id, book in mapping.items():
        if any(word in " ".join(book["tags"]).lower() for word in query_keywords):
            results.append(book["title"])
    return results[:5]  # top 5 recommendations

# Gradio interface function
def gradio_recommender(query_text):
    recommendations = recommend_books(query_text, mapping=mapping)
    if recommendations:
        return "\n".join(recommendations)
    else:
        return "No recommendations found."

# Create Gradio interface
iface = gr.Interface(
    fn=gradio_recommender,
    inputs=gr.Textbox(lines=2, placeholder="Enter your reading preferences..."),
    outputs="text",
    title="Book Recommender",
    description="Type keywords or genres, and get book recommendations!"
)

# Launch app
iface.launch()