from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

MODEL_PATH = '/mnt/d/code/hcl/unet/checkpoints/best_encoder.h5'
DATASET_DIR = '/mnt/d/code/hcl/unet/dataset/'
OUTPUT_DIR = os.path.join(DATASET_DIR, 'segmentation_results')
INPUT_SIZE = 256

def load_and_preprocess_image(img_path, feature_path):
    image = cv2.imread(img_path)
    original_size = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    feature = cv2.imread(feature_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    feature_resized = cv2.resize(feature, (INPUT_SIZE, INPUT_SIZE))
    
    image_norm = image_resized.astype(np.float32) / 255.0
    feature_norm = feature_resized.astype(np.float32) / 255.0
    feature_norm = np.expand_dims(feature_norm, axis=-1)
    
    image_batch = np.expand_dims(image_norm, axis=0)
    feature_batch = np.expand_dims(feature_norm, axis=0)
    
    return image_batch, feature_batch, original_size, image

def postprocess_mask(mask, original_size):
    mask = mask[0, :, :, 0]
    mask = (mask + 1.0) / 2.0
    mask = (mask * 255).astype(np.uint8)
    mask_resized = cv2.resize(mask, (original_size[1], original_size[0]))
    _, binary_mask = cv2.threshold(mask_resized, 135, 255, cv2.THRESH_BINARY)
    return binary_mask

def visualize_results(original_image, predicted_mask, ground_truth):
    original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    
    pred_bgr = cv2.cvtColor(predicted_mask, cv2.COLOR_GRAY2BGR)
    gt_bgr = cv2.cvtColor(ground_truth, cv2.COLOR_GRAY2BGR)
    
    result = np.hstack([original_bgr, gt_bgr, pred_bgr])
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)
    thickness = 1
    
    cv2.putText(result, "Original", (10, 30), font, font_scale, color, thickness)
    cv2.putText(result, "GT Mask", (410, 30), font, font_scale, color, thickness)
    cv2.putText(result, "Pred Mask", (810, 30), font, font_scale, color, thickness)
    
    return result

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = load_model(MODEL_PATH)
    with open(os.path.join(DATASET_DIR, 'test.txt'), 'r') as f:
        test_images = [line.strip() for line in f.readlines()]
    
    for _, img_name in enumerate(test_images[:10], 1):
        print(f"Processing: {img_name}")
        
        try:
            img_path = os.path.join(os.path.join(DATASET_DIR, 'Images'), img_name)
            feature_path = os.path.join(os.path.join(DATASET_DIR, 'Heads'), img_name)
            gt_path = os.path.join(os.path.join(DATASET_DIR, 'Masks'), img_name)
            
            image_batch, feature_batch, original_size, original_image = load_and_preprocess_image(
                img_path, feature_path
            )
            
            prediction = model.predict([image_batch, feature_batch], verbose=0)
            binary_mask= postprocess_mask(prediction, original_size)
            
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            result = visualize_results(original_image, binary_mask, gt_mask)
            
            base_name = os.path.splitext(img_name)[0]
            result_path = os.path.join(OUTPUT_DIR, f"{base_name}_result.jpg")
            cv2.imwrite(result_path, result)

        except Exception as e:
            print(f"  Error: {str(e)}")
            continue

if __name__ == "__main__":
    main()