import gradio as gr
import onnxruntime as ort
import numpy as np
from PIL import Image

from src.utils import load_config

xbd_config = load_config('xbd.yaml')

providers = ['CPUExecutionProvider']
session = ort.InferenceSession("models/damagenet.onnx", providers=providers)

def predict(pre_img, post_img):
    def preprocess(img):
        img = img.resize((1024, 1024))
        img_arr = np.array(img).astype(np.float32) / 255.0
        
        mean = np.array(xbd_config['stats']['mean'])
        std = np.array(xbd_config['stats']['std'])
        img_arr = (img_arr - mean) / std
        
        img_arr = img_arr.astype(np.float32) 
        
        img_arr = np.transpose(img_arr, (2, 0, 1))
        return np.expand_dims(img_arr, axis=0)

    pre_tensor = preprocess(pre_img)
    post_tensor = preprocess(post_img)

    # The keys 'pre' and 'post' must match the 'input_names' you used in export
    ort_inputs = {
        session.get_inputs()[0].name: pre_tensor,
        session.get_inputs()[1].name: post_tensor
    }
    outputs = session.run(None, ort_inputs)
    
    mask = outputs[0]
    if len(mask.shape) == 4:
        mask = np.argmax(mask, axis=1) # Get class with highest probability
    
    # Clean up for display (Scale to 0-255)
    mask_display = (mask[0] * (255 // mask.max()) if mask.max() > 0 else mask[0]).astype(np.uint8)
    return Image.fromarray(mask_display)

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(label="Pre-Disaster Satellite Image", type="pil"),
        gr.Image(label="Post-Disaster Satellite Image", type="pil")
    ],
    outputs=gr.Image(label="Damage Segmentation Mask"),
    title="DamageNet: xBD Damage Assessment",
    description="Upload pre and post-disaster images to segment damage levels."
)

if __name__ == "__main__":
    demo.launch()
