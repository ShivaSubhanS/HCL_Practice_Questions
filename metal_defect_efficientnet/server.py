from flask import Flask, request, render_template_string, jsonify
from tensorflow import keras
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

MODEL_PATH = '/mnt/d/code/hcl/efficientnet_industrial/best_efficientnet_model.h5'
IMG_SIZE = 224  
CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

model = keras.models.load_model(MODEL_PATH)
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NEU Surface Defect Detection</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            max-width: 800px;
            width: 100%;
            padding: 40px;
            animation: fadeIn 0.5s ease-in;
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
        }

        .header h1 {
            color: #333;
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .header p {
            color: #666;
            font-size: 1.1em;
        }

        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            background: #f8f9ff;
            transition: all 0.3s ease;
            cursor: pointer;
            margin-bottom: 30px;
        }

        .upload-area:hover {
            border-color: #764ba2;
            background: #f0f2ff;
            transform: scale(1.02);
        }

        .upload-area.dragover {
            border-color: #764ba2;
            background: #e8ebff;
            transform: scale(1.05);
        }

        .upload-icon {
            font-size: 4em;
            color: #667eea;
            margin-bottom: 20px;
        }

        .upload-text {
            color: #666;
            font-size: 1.1em;
            margin-bottom: 15px;
        }

        .file-input {
            display: none;
        }

        .upload-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 25px;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }

        .upload-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }

        .upload-btn:active {
            transform: translateY(0);
        }

        .preview-section {
            display: none;
            margin-top: 30px;
        }

        .preview-section.active {
            display: block;
            animation: slideIn 0.5s ease-in;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        .image-preview {
            text-align: center;
            margin-bottom: 20px;
        }

        .preview-img {
            max-width: 100%;
            max-height: 400px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }

        .results-section {
            display: none;
            margin-top: 30px;
            padding: 25px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 15px;
            animation: fadeIn 0.5s ease-in;
        }

        .results-section.active {
            display: block;
        }

        .result-title {
            font-size: 1.5em;
            color: #333;
            margin-bottom: 20px;
            text-align: center;
            font-weight: bold;
        }

        .prediction-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }

        .prediction-label {
            font-size: 1.3em;
            color: #333;
            margin-bottom: 10px;
            font-weight: bold;
        }

        .confidence-bar {
            background: #e0e0e0;
            height: 30px;
            border-radius: 15px;
            overflow: hidden;
            position: relative;
        }

        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 1s ease-in-out;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 15px;
            color: white;
            font-weight: bold;
        }

        .top-predictions {
            margin-top: 20px;
        }

        .top-prediction-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            background: white;
            margin-bottom: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        }

        .prediction-name {
            font-weight: 500;
            color: #555;
        }

        .prediction-confidence {
            font-weight: bold;
            color: #667eea;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }

        .loading.active {
            display: block;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @media (max-width: 600px) {
            .container {
                padding: 20px;
            }

            .header h1 {
                font-size: 1.8em;
            }

            .upload-area {
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 NEU Surface Defect Detection</h1>
            <p>Upload an image to detect surface defects</p>
        </div>

        <div class="upload-area" id="uploadArea">
            <div class="upload-icon">📷</div>
            <div class="upload-text">Drag & Drop your image here or click to browse</div>
            <input type="file" id="fileInput" class="file-input" accept="image/*">
            <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
                Choose Image
            </button>
        </div>

        <div class="preview-section" id="previewSection">
            <div class="image-preview">
                <img id="previewImg" class="preview-img" src="" alt="Preview">
            </div>
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing image...</p>
        </div>

        <div class="results-section" id="resultsSection">
            <div class="result-title"> Detection Results</div>
            <div class="prediction-card">
                <div class="prediction-label">Predicted Defect: <span id="predictedClass"></span></div>
                <div class="confidence-bar">
                    <div class="confidence-fill" id="confidenceFill" style="width: 0%">
                        <span id="confidenceText">0%</span>
                    </div>
                </div>
            </div>

            <div class="top-predictions">
                <h3 style="margin-bottom: 15px; color: #333;">Top 3 Predictions:</h3>
                <div id="topPredictions"></div>
            </div>
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const previewSection = document.getElementById('previewSection');
        const previewImg = document.getElementById('previewImg');
        const resultsSection = document.getElementById('resultsSection');
        const loading = document.getElementById('loading');

        // Drag and drop handlers
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });

        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                alert('Please select an image file');
                return;
            }

            const reader = new FileReader();
            reader.onload = (e) => {
                previewImg.src = e.target.result;
                previewSection.classList.add('active');
                resultsSection.classList.remove('active');
                // Automatically start prediction after image is loaded
                setTimeout(() => predictDefect(), 500);
            };
            reader.readAsDataURL(file);
        }

        async function predictDefect() {
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            loading.classList.add('active');
            resultsSection.classList.remove('active');

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (result.error) {
                    alert('Error: ' + result.error);
                    return;
                }

                // Display results
                document.getElementById('predictedClass').textContent = result.predicted_class;
                document.getElementById('confidenceFill').style.width = result.confidence + '%';
                document.getElementById('confidenceText').textContent = result.confidence.toFixed(2) + '%';

                // Display top 3 predictions
                const topPredictionsDiv = document.getElementById('topPredictions');
                topPredictionsDiv.innerHTML = '';
                result.top_predictions.forEach(pred => {
                    const item = document.createElement('div');
                    item.className = 'top-prediction-item';
                    item.innerHTML = `
                        <span class="prediction-name">${pred.class}</span>
                        <span class="prediction-confidence">${pred.confidence.toFixed(2)}%</span>
                    `;
                    topPredictionsDiv.appendChild(item);
                });

                resultsSection.classList.add('active');
            } catch (error) {
                alert('Error during prediction: ' + error.message);
            } finally {
                loading.classList.remove('active');
            }
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read and preprocess image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array)[0]
        
        # Get predicted class
        predicted_idx = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(predictions[predicted_idx] * 100)
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions)[-3:][::-1]
        top_predictions = [
            {
                'class': CLASS_NAMES[idx],
                'confidence': float(predictions[idx] * 100)
            }
            for idx in top_3_idx
        ]
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'all_predictions': {CLASS_NAMES[i]: float(predictions[i] * 100) for i in range(len(CLASS_NAMES))}
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\nServer starting...")
    
    app.run(debug=True, host='0.0.0.0', port=5000)