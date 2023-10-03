import torch
import gradio as gr
from utils import (
    predict,
    get_html,
    get_examples
)

examples = get_examples()
placeholder = 'Enter a word/phrase or multiple words/phrases separated by commas...'


with gr.Blocks() as interface:
  gr.HTML(value=get_html, show_label=True)
  with gr.Row():
    inputs = [gr.Image(type="pil"),
              gr.Textbox(label='Text Prompts', placeholder=placeholder, lines=3)]

  with gr.Row():
    outputs = gr.AnnotatedImage(label="Segmentation Masks")

  with gr.Row():
    button = gr.Button("Visualize Segments")
    button.click(predict, inputs=inputs, outputs=outputs)

  with gr.Row():
    gr.Examples(examples=examples, inputs=inputs, outputs=outputs, fn=predict, cache_examples=True)


interface.launch()