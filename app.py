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
image_paths_np = np.array(
    [
        f.relative_to(folder_path.parent).as_posix()
        for f in folder_path.iterdir()
        if f.is_file()
    ]
)
image_tensors = []
for path in image_paths_np:
    image = Image.open(path).convert("RGB")
    image_tensors.append(preprocess(image))

image_batch = torch.stack(image_tensors).to(device)

# print("Number of paths: ", len(image_paths_np))
# print("One image tensor:", tuple(image_tensors[0].shape))
# print("Complete batch:  ", tuple(image_batch.shape))


# 3. Encode and Store
start = time.perf_counter()
with torch.inference_mode():
    raw_image_embeddings = model.encode_image(image_batch).float()

image_embeddings = raw_image_embeddings / raw_image_embeddings.norm(dim=1, keepdim=True)
if device == "cuda":
    torch.cuda.synchronize()
elapsed = time.perf_counter() - start

# print("Embedding matrix shape:", tuple(image_embeddings.shape))
# print("Embedding dtype:       ", image_embeddings.dtype)
# print("Raw norms:             ", raw_image_embeddings.norm(dim=1).cpu().numpy())
# print("Normalized norms:      ", image_embeddings.norm(dim=1).cpu().numpy())
# print(f"Encoding time:          {elapsed:.3f} seconds")


# Search Function
def search(message, history, top_k):

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
    object_fit="contain", 
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
                # examples=[
                #     ["a photo of a bottle", 5],
                #     ["a photo of a cup of coffee", 5],
                # ],
                multimodal=True,
                additional_inputs=[top_k_slider],
                additional_outputs=[results],
                api_name="chat",
            )
        with gr.Column():
            gr.Markdown("<center><h1>Search Results</h1></center>")
            results.render()

demo.launch()
