import gradio as gr
import tensorflow as tf

interface = gr.Interface(
    fn=predict_image, 
    inputs=gr.inputs.Image(shape=(width, height)), 
    outputs="text",
    live=True,
    capture_session=True
)
interface.launch()
