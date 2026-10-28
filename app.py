# 0. Hugging Face ZeroGPU compatibility: MUST be imported before torch or any CUDA library!
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(fn=None, **kwargs):
            if fn is None:
                return lambda f: f
            return fn

import os
from pathlib import Path
import tempfile
import time
import zipfile

import clip
from PIL import Image
import gradio as gr
import numpy as np
import torch

# 1. Load CLIP model globally on CPU at startup (ZeroGPU moves to CUDA inside @spaces.GPU)
model_name = "ViT-B/32"
print(f"Loading CLIP ({model_name})...")
model, preprocess = clip.load(model_name, device="cpu")
model.eval()

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_current_device():
    """Return 'cuda' if GPU is active (inside @spaces.GPU), else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def extract_images_from_files(files):
    """Collect image paths from uploaded files and extract any .zip archives."""
    if not files:
        return []

    collected_paths = []
    for item in files:
        file_path = item.name if hasattr(item, "name") else str(item)
        p = Path(file_path)

        if p.suffix.lower() == ".zip":
            try:
                extract_dir = tempfile.mkdtemp(prefix="hf_search_zip_")
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
                for root, _, filenames in os.walk(extract_dir):
                    for fn in filenames:
                        if Path(fn).suffix.lower() in VALID_EXTENSIONS:
                            collected_paths.append(os.path.join(root, fn))
            except Exception as e:
                print(f"Error extracting {file_path}: {e}")
        elif p.suffix.lower() in VALID_EXTENSIONS:
            collected_paths.append(file_path)

    return collected_paths


def encode_image_paths(paths):
    """Preprocess and compute CLIP normalized embeddings for given paths."""
    tensors = []
    valid_paths = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            tensors.append(preprocess(img))
            valid_paths.append(p)
        except Exception as e:
            print(f"Skipping {p}: {e}")

    if not tensors:
        return valid_paths, None, 0.0

    curr_device = get_current_device()
    model.to(curr_device)

    start = time.perf_counter()
    embeddings_list = []
    batch_size = 32
    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i : i + batch_size]).to(curr_device)
        with torch.inference_mode():
            raw = model.encode_image(batch).float()
            normed = raw / raw.norm(dim=1, keepdim=True)
            embeddings_list.append(normed.cpu())

    embeddings = torch.cat(embeddings_list, dim=0)
    if curr_device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return valid_paths, embeddings, elapsed


def get_initial_state():
    """Load local images from 'Images/' folder if present, otherwise start empty."""
    folder = Path("Images")
    if not folder.exists():
        return (
            {"paths": [], "embeddings": None},
            "⚠️ *No images loaded yet. Upload your images above to get started.*",
            [],
        )

    local_paths = [
        f.as_posix()
        for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
    ]
    if not local_paths:
        return (
            {"paths": [], "embeddings": None},
            "⚠️ *No images loaded yet. Upload your images above to get started.*",
            [],
        )

    paths, embeddings, elapsed = encode_image_paths(local_paths)
    if embeddings is None:
        return (
            {"paths": [], "embeddings": None},
            "⚠️ *No valid images found in 'Images/' folder. Upload images above.*",
            [],
        )

    status = f"✅ Loaded {len(paths)} local images from 'Images/' in {elapsed:.2f}s."
    preview = [(p, Path(p).name) for p in paths[:24]]
    return {"paths": paths, "embeddings": embeddings}, status, preview


# Pre-compute initial state if local images exist
initial_state, initial_status, initial_preview = get_initial_state()


@spaces.GPU
def handle_upload(files):
    """Handler for user uploading images or zip via Gradio."""
    if not files:
        return (
            {"paths": [], "embeddings": None},
            "⚠️ *No files selected. Please upload images or a ZIP file.*",
            [],
        )

    paths = extract_images_from_files(files)
    if not paths:
        return (
            {"paths": [], "embeddings": None},
            "❌ *No valid image files (.jpg, .png, .webp) found in the upload.*",
            [],
        )

    valid_paths, embeddings, elapsed = encode_image_paths(paths)
    if embeddings is None or len(valid_paths) == 0:
        return (
            {"paths": [], "embeddings": None},
            "❌ *Failed to preprocess any of the uploaded images.*",
            [],
        )

    state = {"paths": valid_paths, "embeddings": embeddings}
    msg = f"✅ **Successfully indexed {len(valid_paths)} images in {elapsed:.2f}s!** You can now search them below."
    preview = [(p, Path(p).name) for p in valid_paths[:24]]
    return state, msg, preview


def clear_database():
    """Reset image database."""
    return (
        {"paths": [], "embeddings": None},
        "ℹ️ *Database cleared. Upload images above to begin.*",
        [],
        None,
    )


@spaces.GPU
def search(message, history, top_k, state):
    """Search user query against the uploaded images stored in session state."""
    if (
        not state
        or state.get("embeddings") is None
        or len(state.get("paths", [])) == 0
    ):
        return (
            "⚠️ **No images in database!** Please upload and index your images in Step 1 first.",
            [],
        )

    curr_device = get_current_device()
    model.to(curr_device)

    embeddings = state["embeddings"].to(curr_device)
    image_paths_np = np.array(state["paths"])

    text_query = message.get("text", "")
    files = message.get("files", [])

    # Image Query
    if files:
        query_path = files[0]
        try:
            query_img = Image.open(query_path).convert("RGB")
            query_tensor = preprocess(query_img).unsqueeze(0).to(curr_device)
            with torch.inference_mode():
                query_embedding = model.encode_image(query_tensor).float()
            status_msg = f"Searched using uploaded image: `{Path(query_path).name}`"
        except Exception as e:
            return f"❌ Error processing query image: {e}", []

    # Text Query
    elif text_query:
        tokens = clip.tokenize([text_query]).to(curr_device)
        with torch.inference_mode():
            query_embedding = model.encode_text(tokens).float()
        status_msg = f"Searched using text: '{text_query}'"

    else:
        return "Please enter text or upload an image to search.", []

    # Normalize query embedding
    query_embedding = query_embedding / query_embedding.norm(dim=1, keepdim=True)

    # Calculate cosine similarity scores
    scores = (embeddings @ query_embedding.T).squeeze(1)
    scores_np = scores.cpu().numpy()

    # Sort high to low
    order = np.argsort(scores_np)[::-1]
    actual_top_k = min(int(top_k), len(image_paths_np))
    top_k_order = order[:actual_top_k]

    top_k_images = image_paths_np[top_k_order].tolist()
    top_k_scores = scores_np[top_k_order].tolist()

    gallery_items = [
        (img_path, f"Score: {score:.3f}")
        for img_path, score in zip(top_k_images, top_k_scores)
    ]

    return status_msg, gallery_items


# Gradio Interface
with gr.Blocks(title="Semantic Image Search") as demo:
    # Session state holds user-specific indexed dataset
    dataset_state = gr.State(value=initial_state)

    results = gr.Gallery(
        columns=1,
        height=800,
        object_fit="scale-down",
        render=False,
    )
    top_k_slider = gr.Slider(
        minimum=1,
        maximum=20,
        step=1,
        value=5,
        label="Choose number of top results",
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("<center><h1>Semantic Image Search</h1></center>")

            with gr.Accordion("📁 Step 1: Upload Your Image Database", open=True):
                file_upload = gr.File(
                    file_count="multiple",
                    file_types=["image", ".zip"],
                    label="Select images or a .zip archive to search through",
                )
                with gr.Row():
                    index_btn = gr.Button("⚡ Index Images", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear Database", variant="secondary")

                index_status = gr.Markdown(value=initial_status)

                with gr.Accordion("👁️ Preview Uploaded Database", open=False):
                    db_preview = gr.Gallery(
                        value=initial_preview,
                        columns=4,
                        height=200,
                        object_fit="contain",
                        label="Active Database Preview",
                    )

            gr.Markdown("### 🔍 Step 2: Search")
            gr.ChatInterface(
                fn=search,
                multimodal=True,
                additional_inputs=[top_k_slider, dataset_state],
                additional_outputs=[results],
                api_name="chat",
            )

        with gr.Column(scale=1):
            gr.Markdown("<center><h1>Search Results</h1></center>")
            results.render()

    # Event handlers for uploading & clearing
    file_upload.change(
        fn=handle_upload,
        inputs=[file_upload],
        outputs=[dataset_state, index_status, db_preview],
    )
    index_btn.click(
        fn=handle_upload,
        inputs=[file_upload],
        outputs=[dataset_state, index_status, db_preview],
    )
    clear_btn.click(
        fn=clear_database,
        inputs=[],
        outputs=[dataset_state, index_status, db_preview, file_upload],
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    share = os.environ.get("GRADIO_SHARE", "False").lower() in ("true", "1")
    demo.launch(server_name="0.0.0.0", server_port=port, share=share)
