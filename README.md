---
title: Semantic Image Search
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Semantic-Image-Search

A CLIP-powered semantic image search engine with a Gradio web interface. Search through any collection of images using descriptive natural language or reference images.

## Features
- **Zero repository images needed**: Users can upload their own images or a `.zip` archive directly in the web interface.
- **Multimodal search**: Search by natural language query (text) or by uploading an image.
- **Session-isolated**: Each visitor's uploaded database and search session is private to their browser tab.
- **Batched CLIP inference**: Embeddings are computed efficiently on CPU or GPU in mini-batches.

## Running Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open the displayed local URL in your browser, upload your images or `.zip` in Step 1, and search in Step 2!
