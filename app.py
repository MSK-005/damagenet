import gradio as gr
import onnxruntime as ort
import numpy as np
from PIL import Image

from src.utils import load_config

xbd_config = load_config('xbd.yaml')

# Damage class colormap: background, partial damage, destroyed
COLORMAP = np.array([
    [0,   0,   0,   0  ],  # 0 — no damage    (transparent)
    [255, 200, 0,   180],  # 1 — partial       (yellow, semi-transparent)
    [255, 0,   0,   200],  # 2 — destroyed     (red, semi-transparent)
], dtype=np.uint8)

providers = ['CPUExecutionProvider']
session = ort.InferenceSession("models/damagenet.onnx", providers=providers)


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.resize((1024, 1024), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array(xbd_config['stats']['mean'], dtype=np.float32)
    std  = np.array(xbd_config['stats']['std'],  dtype=np.float32)
    arr  = (arr - mean) / std
    arr  = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0)


def colorize_mask(mask: np.ndarray) -> Image.Image:
    """Map class indices to RGBA colors."""
    rgba = COLORMAP[mask]  # (H, W, 4)
    return Image.fromarray(rgba, mode='RGBA')


def overlay_mask_on_image(post_img: Image.Image, mask: np.ndarray) -> Image.Image:
    """Blend colorized mask over the post-disaster image."""
    post_resized = post_img.resize((1024, 1024), Image.BILINEAR).convert('RGBA')
    mask_rgba    = colorize_mask(mask)
    blended      = Image.alpha_composite(post_resized, mask_rgba)
    return blended.convert('RGB')


def predict(pre_img: Image.Image, post_img: Image.Image) -> Image.Image:
    pre_tensor  = preprocess(pre_img)
    post_tensor = preprocess(post_img)

    ort_inputs = {
        session.get_inputs()[0].name: pre_tensor,
        session.get_inputs()[1].name: post_tensor,
    }
    outputs = session.run(None, ort_inputs)

    # outputs[0]: (1, num_classes, H, W) logits
    mask = np.argmax(outputs[0], axis=1)[0]  # (H, W)

    return overlay_mask_on_image(post_img, mask)


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(label="Pre-Disaster Satellite Image",  type="pil"),
        gr.Image(label="Post-Disaster Satellite Image", type="pil"),
    ],
    outputs=gr.Image(label="Damage Assessment Overlay"),
    title="DamageNet — Satellite Damage Assessment",
    description=(
        "Upload pre- and post-disaster satellite imagery to assess structural damage. "
        "Yellow indicates partial damage, red indicates destruction."
    ),
)

if __name__ == "__main__":
    demo.launch()