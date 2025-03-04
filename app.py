from flask import Flask, request, jsonify, render_template
from diffusers import StableDiffusionPipeline
import torch
import os

# Set up Flask app and define template folder
app = Flask(__name__, template_folder="templates")

# Ensure the static directory exists for saving images
os.makedirs("static", exist_ok=True)

# Load Stable Diffusion Model
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model with correct data type
if device == "cpu":
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
else:
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

pipe.to(device)

@app.route('/')
def home():
    return render_template("index.html")  # Serve the frontend

@app.route('/generate-logo', methods=['POST'])
def generate_logo():
    data = request.json
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "Prompt cannot be empty"}), 400

    try:
        print(f"Generating logo for: {prompt}")  # Debugging log
        image = pipe(prompt).images[0]
        image_path = "static/logo.png"
        image.save(image_path)
        print(f"Logo saved successfully at {image_path}")

        return jsonify({"image_url": f"/{image_path}"})
    except Exception as e:
        print(f"Error generating logo: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
