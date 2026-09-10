from pathlib import Path

import clip
import gradio as gr
import numpy as np
import torch
from PIL import Image

# The following code from Open CV website: https://opencv.org/clip/
# 1. Load the pre-trained CLIP model and its processor
# The processor handles preparing the image and text for the model
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

# 3. Encode and Store
with torch.inference_mode():
    raw_image_embeddings = model.encode_image(image_batch).float()

image_embeddings = raw_image_embeddings / raw_image_embeddings.norm(dim=1, keepdim=True)
if device == "cuda":
    torch.cuda.synchronize()


# Search Function
def search(message, history, top_k):
    # Encode the query
    tokens = clip.tokenize([message]).to(device)
    with torch.inference_mode():
        text_embedding = model.encode_text(tokens).float()
    text_embedding = text_embedding / text_embedding.norm(dim=1, keepdim=True)

    # Compare the query with each image
    scores = (image_embeddings @ text_embedding.T).squeeze(1)
    scores_np = scores.cpu().numpy()

    order = np.argsort(scores_np)[::-1]
    top_k_order = order[:top_k]
    top_k_images = image_paths_np[top_k_order].tolist()

    # output_string = (
    #     f"Indices, high to low: {order}\n"
    #     f"Names, high to low:   {image_paths_np[order]}\n"
    #     f"Scores, high to low:  {scores_np[order]}"
    # )
    # return output_string, top_k_images

    return "", top_k_images


with gr.Blocks() as demo:
    results = gr.Gallery(render=False)
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
                additional_inputs=[top_k_slider],
                additional_outputs=[results],
                api_name="chat",
            )
        with gr.Column():
            gr.Markdown("<center><h1>Search Results</h1></center>")
            results.render()

demo.launch()
