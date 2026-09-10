import gradio as gr

def chat(message, history):
    if "python" in message.lower():
        return ["gazebo.png", "landmark_skeletal.png"]
    elif "javascript" in message.lower():
        return ["ping-pong-ball.png", "working_demo.png"]
    else:
        return "Please ask about Python or JavaScript.", None

with gr.Blocks() as demo:
    results = gr.Gallery(render=False)
    with gr.Row():
        with gr.Column():
            gr.Markdown("<center><h1>Semantic Image Search</h1></center>")
            gr.ChatInterface(
                chat,
                examples=["A black bottle", "A cup of coffee"],
                additional_outputs=[results],
                api_name="chat",
            )
        with gr.Column():
            gr.Markdown("<center><h1>Top 5 results</h1></center>")
            results.render()

demo.launch()