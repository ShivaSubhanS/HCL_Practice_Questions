import torch
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import base64
from transformers import ViTForImageClassification, ViTImageProcessor

app = Flask(__name__)

MODEL_PATH = "best_vit_model.pth"
MODEL_NAME = "google/vit-base-patch16-224"
CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
NUM_CLASSES = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model...")
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
model = ViTForImageClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True
).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Model loaded successfully! Best Val Acc: {checkpoint['val_acc']:.4f}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(DEVICE)
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class_idx].item()
        
        all_probabilities = {
            CLASS_NAMES[i]: float(probabilities[0][i].item()) 
            for i in range(NUM_CLASSES)
        }
        
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'predicted_class': CLASS_NAMES[predicted_class_idx],
            'confidence': confidence,
            'all_probabilities': all_probabilities,
            'image': img_str
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
