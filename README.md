# Semantic-Image-Search

**Installation**
To run this application, you need Python 3.8+ and the following dependencies.

First, install PyTorch according to your system's hardware (CPU or GPU) by following the instructions at pytorch.org. Then, install the remaining requirements and OpenAI's CLIP repository:

```bash
pip install gradio pillow numpy
pip install -q ftfy regex tqdm scikit-image git+https://github.com/openai/CLIP.git

```

**Setting Up the Image Cache**

1. Create a folder named `Images` in the same directory as your Python script.
2. Place all the images (`.jpg`, `.png`, etc.) you want to make searchable inside this `Images` folder.
3. When you start the Gradio app, the script will automatically read all files in this folder, process them through the CLIP model, and store their embeddings in a PyTorch tensor in memory. This serves as your search cache.

**Running a Query**

1. Execute the Python script from your terminal:

```bash
python app.py

```

2. Open the provided `[http://127.0.0.1:7860/](http://127.0.0.1:7860/)` local URL in your web browser.
3. To search using **text**: Type a descriptive query (e.g., "a dog playing in grass") into the chat box and press enter.
4. To search using an **image**: Click the paperclip icon in the chat box, upload a reference image, and press enter.
5. Use the "Number of Results (Top K)" slider to control how many ranked matches the gallery displays.