import torch
import os
import io
import base64
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from models import Generator
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)

# Configuration
N_CLASSES = 10
LATENT_DIM = 100
IMG_SHAPE = (1, 28, 28)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
generator = Generator(N_CLASSES, LATENT_DIM, IMG_SHAPE).to(DEVICE)

def get_latest_checkpoint():
    files = [f for f in os.listdir('.') if f.startswith('generator_epoch_') and f.endswith('.pth')]
    if not files:
        return None
    # Sort by epoch number
    files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]), reverse=True)
    return files[0]

latest_ckpt = get_latest_checkpoint()
if latest_ckpt:
    print(f"Loading checkpoint: {latest_ckpt}")
    generator.load_state_dict(torch.load(latest_ckpt, map_location=DEVICE))
    generator.eval()
else:
    print("No checkpoint found. Please train the model first.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    digit = data.get('digit', 0)
    
    if not(0 <= digit <= 9):
        return jsonify({"error": "Invalid digit"}), 400

    # Generate image
    z = torch.randn(1, LATENT_DIM).to(DEVICE)
    label = torch.LongTensor([digit]).to(DEVICE)
    
    with torch.no_grad():
        gen_img = generator(z, label)
    
    # Preprocess image for display
    gen_img = 0.5 * gen_img + 0.5  # denormalize
    gen_img = gen_img.cpu().numpy()[0, 0, :, :]
    gen_img = (gen_img * 255).astype(np.uint8)
    
    # Convert to base64
    img = Image.fromarray(gen_img)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return jsonify({"image": img_str})

if __name__ == '__main__':
    from waitress import serve
    print("Starting server on port 7860...")
    serve(app, host='0.0.0.0', port=7860)
