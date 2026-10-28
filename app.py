import os
from pathlib import Path
from PIL import Image
import time

import clip
import gradio as gr
import numpy as np
import torch


# 1. Load the pre-trained CLIP model and its processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ViT-B/32"
model, preprocess = clip.load(model_name, device=device)
model.eval()


# 2. Load images and preprocesses
folder_path = Path("Images")
folder_path.mkdir(parents=True, exist_ok=True)

valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
image_files = [
    f.relative_to(folder_path.parent).as_posix()
    for f in folder_path.iterdir()
    if f.is_file() and f.suffix.lower() in valid_extensions
]

image_tensors = []
valid_paths = []
for path in image_files:
    try:
        image = Image.open(path).convert("RGB")
        image_tensors.append(preprocess(image))
        valid_paths.append(path)
    except Exception as e:
        print(f"Skipping {path}: {e}")

image_paths_np = np.array(valid_paths)

# 3. Encode and Store
if len(image_tensors) > 0:
    image_batch = torch.stack(image_tensors).to(device)
    start = time.perf_counter()
    with torch.inference_mode():
        raw_image_embeddings = model.encode_image(image_batch).float()

    image_embeddings = raw_image_embeddings / raw_image_embeddings.norm(dim=1, keepdim=True)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"Loaded and encoded {len(image_paths_np)} images in {elapsed:.3f} seconds.")
else:
    image_embeddings = None
    print("Notice: No images found in the 'Images' folder yet. Place image files inside 'Images/' to search.")


# Search Function
def search(message, history, top_k):
    if image_embeddings is None or len(image_paths_np) == 0:
        return "No images found in the 'Images' directory. Please add images to search.", []

    text_query = message.get("text", "")
    files = message.get("files", [])

    # Image Query
    if files:
        query_path = files[0]
        query_img = Image.open(query_path).convert("RGB")
        query_tensor = preprocess(query_img).unsqueeze(0).to(device)
        
        with torch.inference_mode():
            query_embedding = model.encode_image(query_tensor).float()
        status_msg = f"Searched using uploaded image: `{Path(query_path).name}`"

    # Text Query
    elif text_query:
        tokens = clip.tokenize([text_query]).to(device)
        with torch.inference_mode():
            query_embedding = model.encode_text(tokens).float()
        status_msg = f"Searched using text: '{text_query}'"

    else:
        return "Please enter text or upload an image to search.", []

    # Normalize query embedding
    query_embedding = query_embedding / query_embedding.norm(dim=1, keepdim=True)

    # Calculate similarity scores
    scores = (image_embeddings @ query_embedding.T).squeeze(1)
    scores_np = scores.cpu().numpy()

    # Sort high to low
    order = np.argsort(scores_np)[::-1]
    
    actual_top_k = min(int(top_k), len(image_paths_np))
    top_k_order = order[:actual_top_k]
    
    top_k_images = image_paths_np[top_k_order].tolist()
    top_k_scores = scores_np[top_k_order].tolist()
    
    gallery_items = [(img_path, f"Score: {score:.3f}") for img_path, score in zip(top_k_images, top_k_scores)]
    
    return status_msg, gallery_items


# Gradio Interface 
with gr.Blocks() as demo:
    results = gr.Gallery(
        columns=1, 
        height=800,
        object_fit="scale-down", 
        render=False
    )
    top_k_slider = gr.Slider(
        minimum=1, maximum=10, step=1, value=5, label="Choose number of top results",
    )
    with gr.Row():
        with gr.Column():
            gr.Markdown("<center><h1>Semantic Image Search</h1></center>")
            gr.ChatInterface(
                fn=search,
                multimodal=True,
                additional_inputs=[top_k_slider],
                additional_outputs=[results],
                api_name="chat",
            )
        with gr.Column():
            gr.Markdown("<center><h1>Search Results</h1></center>")
            results.render()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    share = os.environ.get("GRADIO_SHARE", "False").lower() in ("true", "1")
    demo.launch(server_name="0.0.0.0", server_port=port, share=share)
