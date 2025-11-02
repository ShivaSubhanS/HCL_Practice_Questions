from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

import uuid

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'runs/segment/steel_defect_yolo11/weights/best.pt'

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

temp_results = {}

model = None

def load_model():
    global model
    try:
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_defect_info(class_id):
    defect_info = {
        0: {
            'name': 'Class 1 Defect',
            'description': 'Surface defect type 1',
            'color': (255, 0, 0)
        },
        1: {
            'name': 'Class 2 Defect',
            'description': 'Surface defect type 2',
            'color': (0, 255, 0)
        },
        2: {
            'name': 'Class 3 Defect',
            'description': 'Surface defect type 3',
            'color': (0, 0, 255)
        },
        3: {
            'name': 'Class 4 Defect',
            'description': 'Surface defect type 4',
            'color': (255, 255, 0)
        }
    }
    return defect_info.get(class_id, {
        'name': 'Unknown Defect',
        'description': 'Unknown defect type',
        'color': (128, 128, 128)
    })

def process_image_from_bytes(image_bytes, conf_threshold=0.25, iou_threshold=0.45):
    global model
    
    if model is None:
        return None, "Model not loaded"
    
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        temp_filename = f"temp_{uuid.uuid4()}.jpg"
        temp_path = os.path.join('/tmp', temp_filename)
        cv2.imwrite(temp_path, img)
        
        try:
            results = model.predict(
                source=temp_path,
                conf=conf_threshold,
                iou=iou_threshold,
                save=False,
                verbose=False,
                imgsz=1280
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        overlay = img_rgb.copy()
        detections = []
        
        for result in results:
            if result.masks is not None and len(result.masks) > 0:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                
                for idx, (mask, box, cls, conf) in enumerate(zip(masks, boxes, classes, confidences)):
                    mask_resized = cv2.resize(
                        mask.astype(np.uint8),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST
                    )
                    
                    defect_info = get_defect_info(cls)
                    color = defect_info['color']
                    
                    mask_3ch = np.zeros_like(img_rgb)
                    mask_3ch[mask_resized > 0] = color
                    overlay = cv2.addWeighted(overlay, 1, mask_3ch, 0.4, 0)
                    
                    contours, _ = cv2.findContours(
                        mask_resized,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(overlay, contours, -1, color, 2)
                    
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                    
                    defect_area = np.sum(mask_resized > 0)
                    total_area = h * w
                    area_percentage = (defect_area / total_area) * 100
                    
                    label = f"{defect_info['name']}: {conf:.2f}"
                    (text_w, text_h), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        overlay,
                        (x1, y1 - text_h - 10),
                        (x1 + text_w + 10, y1),
                        color,
                        -1
                    )
                    cv2.putText(
                        overlay,
                        label,
                        (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )
                    
                    detections.append({
                        'id': idx + 1,
                        'class_id': int(cls),
                        'class_name': defect_info['name'],
                        'description': defect_info['description'],
                        'confidence': float(conf),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'area_pixels': int(defect_area),
                        'area_percentage': float(area_percentage),
                        'color': color
                    })
        
        summary = {
            'total_defects': len(detections),
            'image_size': {'width': w, 'height': h},
            'detections': detections,
            'has_defects': len(detections) > 0
        }
        
        class_counts = {}
        for det in detections:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        summary['class_distribution'] = class_counts
        
        return overlay, summary
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, str(e)

def image_to_base64(image_array):
    image_pil = Image.fromarray(image_array)
    buffered = BytesIO()
    image_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload JPG, JPEG, or PNG'}), 400
    try:
        file_bytes = file.read()
        
        result_image, summary = process_image_from_bytes(file_bytes, 0.25, 0.45)
        if result_image is None:
            return jsonify({'error': f'Processing failed: {summary}'}), 500
        
        nparr = np.frombuffer(file_bytes, np.uint8)
        original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        original_base64 = image_to_base64(original_img)
        result_base64 = image_to_base64(result_image)
        
        return jsonify({
            'success': True,
            'original_image': original_base64,
            'result_image': result_base64,
            'summary': summary,
            'filename': secure_filename(file.filename)
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })
if __name__ == '__main__':
    if load_model():
        print("Starting Flask server...")
        print("Open your browser and go to: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please check MODEL_PATH in the code.")
        print(f"Expected model at: {MODEL_PATH}")