from tensorflow import keras
import numpy as np
import os
import cv2
import glob
import random

IMG_SIZE = 256
MODEL_PATH = 'models/best_carpet_autoencoder.h5'
DATA_PATH = '/mnt/d/code/hcl/autoencoder'

def load_and_preprocess_image(img_path, img_size=256):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype('float32') / 255.0
    return img

def calculate_reconstruction_error(model, image):
    img_batch = np.expand_dims(image, axis=0)
    reconstruction = model.predict(img_batch, verbose=0)[0]
    mse = np.mean((image - reconstruction) ** 2)
    mae = np.mean(np.abs(image - reconstruction))
    return reconstruction, mse, mae

def test_random_images():
    model = keras.models.load_model(MODEL_PATH, compile=False)
    test_categories = ['good', 'color', 'cut', 'hole', 'metal_contamination', 'thread']
    test_path = os.path.join(DATA_PATH, 'carpet', 'test')
    results = []
    for category in test_categories:
        category_path = os.path.join(test_path, category)
        img_files = glob.glob(os.path.join(category_path, '*.png'))
        random_img_path = random.choice(img_files)
        img_name = os.path.basename(random_img_path)
        image = load_and_preprocess_image(random_img_path, IMG_SIZE)
        reconstruction, mse, mae = calculate_reconstruction_error(model, image)
        results.append({
            'category': category,
            'image_name': img_name,
            'image_path': random_img_path,
            'original': image,
            'reconstruction': reconstruction,
            'mse': mse,
            'mae': mae,
            'has_defect': category != 'good'
        })
        print(f"\n{category.upper()} - {img_name}")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
    suggest_threshold(results)
    return results

def suggest_threshold(results):
    normal_errors = [r['mse'] for r in results if not r['has_defect']]
    defect_errors = [r['mse'] for r in results if r['has_defect']]
    if len(normal_errors) > 0 and len(defect_errors) > 0:
        avg_normal = np.mean(normal_errors)
        avg_defect = np.mean(defect_errors)
        max_normal = np.max(normal_errors)
        min_defect = np.min(defect_errors)
        print(f"\n\nNormal images:")
        print(f"  Average MSE: {avg_normal:.6f}")
        print(f"  Max MSE: {max_normal:.6f}")
        print(f"\nDefective images:")
        print(f"  Average MSE: {avg_defect:.6f}")
        print(f"  Min MSE: {min_defect:.6f}")
        suggested_threshold = (max_normal + min_defect) / 2
        print(f"\nSuggested threshold: {suggested_threshold:.6f}")
        print(f"  (Images with MSE > {suggested_threshold:.6f} are DEFECTIVE)")

if __name__ == '__main__':
    results = test_random_images()
