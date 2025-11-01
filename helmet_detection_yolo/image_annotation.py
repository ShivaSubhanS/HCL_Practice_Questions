import os
import cv2
import torch
import numpy as np
import sys
import glob
sys.path.append("/mnt/d/code/hcl/helmet_detection/sam2")
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2

# Paths
INPUT_IMG_DIR = 'helmet_dataset/input_images/images'
OUTPUT_IMG_DIR = 'helmet_dataset/images'
OUTPUT_LABEL_DIR = 'helmet_dataset/labels'
PREVIEW_DIR = 'helmet_dataset/preview'

os.makedirs(INPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
try:
    sam2_checkpoint = "sam21-hiera-large" 
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml" 
    
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    print("SAM 2.1 Hiera Large model loaded successfully from local checkpoint.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

clicked_points = []
skip_image = False
current_image = None
all_polygons = []
window_name = "SAM2 Image Annotation - Left Click: Segment | Space: Next | Right Arrow: Skip"

def mask_to_polygon(mask):
    """Convert binary mask to polygon coordinates"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    max_contour = max(contours, key=cv2.contourArea)
    return max_contour.reshape(-1, 2)

def save_mask_preview(img, masks, scores, point, img_name, point_num, selected_idx):
    """Save visualization of all masks with selected one highlighted"""
    height, width = img.shape[:2]
    display_width = width * min(3, len(masks))
    display_img = np.zeros((height, display_width, 3), dtype=np.uint8)
    
    colors = [(0, 255, 0), (255, 255, 0), (255, 0, 255)]  
    
    for i, (mask, color, score) in enumerate(zip(masks, colors, scores)):
        start_x = i * width
        display_img[:, start_x:start_x+width] = img.copy()
        
        overlay = np.zeros_like(img)
        overlay[mask > 0] = color
        display_img[:, start_x:start_x+width] = cv2.addWeighted(
            display_img[:, start_x:start_x+width], 0.7, overlay, 0.3, 0
        )
        
        cv2.circle(display_img, (point[0] + start_x, point[1]), 5, (255, 0, 0), -1)
        
        label = f'Mask {i} ({score:.3f})'
        cv2.putText(display_img, label, (start_x + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if i == selected_idx:
            cv2.rectangle(display_img, (start_x, 0), (start_x + width, height), 
                         (0, 0, 255), 5)
            cv2.putText(display_img, 'SELECTED', (start_x + 10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    preview_path = os.path.join(PREVIEW_DIR, f"{img_name}_helmet{point_num}.jpg")
    cv2.imwrite(preview_path, display_img)
    print(f"Saved preview to {preview_path}")

def mouse_callback(event, x, y, flags, param):
    """Handle mouse events"""
    global clicked_points, skip_image, current_image, all_polygons
    
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Click {len(clicked_points)} registered at ({x}, {y})")
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        skip_image = True
        print("Image skipped by user")

def process_point_with_sam(img, point, img_name, point_num):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    
    if point[0] >= width or point[1] >= height or point[0] < 0 or point[1] < 0:
        print(f"Warning: Point {point} is out of bounds")
        return None
    
    predictor.set_image(img_rgb)
    
    # SAM2 prediction
    input_point = np.array([point])
    input_label = np.array([1]) 
    
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )
    
    print(f"  Generated {len(masks)} masks with scores: {[f'{s:.4f}' for s in scores]}")
    
    selected_idx = np.argmax(scores)
    mask = masks[selected_idx]
    print(f"  Selected mask {selected_idx} with highest score {scores[selected_idx]:.4f}")
    
    save_mask_preview(img, masks, scores, point, img_name, point_num, selected_idx)
    
    polygon = mask_to_polygon(mask)
    if polygon is None:
        print(f"  Warning: No contour found")
        return None
    
    print(f"  Extracted polygon with {len(polygon)} points")
    
    polygon_norm = polygon.astype(np.float32)
    polygon_norm[:, 0] /= width
    polygon_norm[:, 1] /= height
    
    return polygon_norm

def save_image_annotations(img, polygons, img_name):
    if not polygons:
        print(f"No annotations to save for {img_name}")
        return False
    
    img_path = os.path.join(OUTPUT_IMG_DIR, f"{img_name}.jpg")
    cv2.imwrite(img_path, img)
    
    label_path = os.path.join(OUTPUT_LABEL_DIR, f"{img_name}.txt")
    with open(label_path, 'w') as f:
        for polygon_norm in polygons:
            polygon_flat = polygon_norm.flatten()
            polygon_str = ' '.join(f'{coord:.6f}' for coord in polygon_flat)
            f.write(f"0 {polygon_str}\n")  
    
    print(f"✓ Saved {img_name} with {len(polygons)} helmet(s)\n")
    return True

def main():
    global clicked_points, skip_image, current_image, all_polygons
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(INPUT_IMG_DIR, ext)))
        image_files.extend(glob.glob(os.path.join(INPUT_IMG_DIR, ext.upper())))
    
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"\nNo images found in {INPUT_IMG_DIR}")
        print(f"Please add images to {INPUT_IMG_DIR} and try again.")
        print("Supported formats: JPG, JPEG, PNG, BMP")
        return
    
    total_images = len(image_files)
    print(f"\nFound {total_images} images in {INPUT_IMG_DIR}")
    print("\nControls:")
    print("  LEFT CLICK        - Segment helmet at clicked point (click multiple times for multiple helmets)")
    print("  RIGHT ARROW/CLICK - Skip current image")
    print("  SPACE             - Save annotations and move to next image")
    print("  Q                 - Quit annotation\n")
    
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    annotated_count = 0
    skipped_count = 0
    
    for img_idx, img_path in enumerate(image_files):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading {img_path}, skipping...")
            continue
        
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        current_image = img.copy()
        clicked_points = []
        all_polygons = []
        skip_image = False
        processed_points = 0
        
        print(f"Image {img_idx+1}/{total_images}: {img_name}")
        
        while True:
            display = current_image.copy()
            
            for i, point in enumerate(clicked_points):
                color = (0, 255, 0) if i < processed_points else (0, 0, 255)
                cv2.circle(display, point, 5, color, -1)
                cv2.putText(display, str(i+1), (point[0]+10, point[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            cv2.putText(display, f"Image {img_idx+1}/{total_images}: {img_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, f"Annotated: {annotated_count} | Skipped: {skipped_count}", (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, f"Helmets on this image: {len(clicked_points)}", (10, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display, "Left Click: Add Helmet | Space: Save & Next | Right Arrow/Right Click: Skip | Q: Quit", 
                       (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting annotation...")
                cv2.destroyAllWindows()
                print(f"\nFinal stats:")
                print(f"  Total images processed: {img_idx+1}")
                print(f"  Annotated: {annotated_count}")
                print(f"  Skipped: {skipped_count}")
                return
            
            elif key == 83:  
                skip_image = True
                print(f"Skipped {img_name}\n")
                skipped_count += 1
                break
            
            elif key == ord(' '):
                if len(clicked_points) > 0:
                    print(f"\nProcessing {len(clicked_points)} helmet(s) on {img_name}...")
                    for i, point in enumerate(clicked_points):
                        if i >= processed_points:
                            print(f"Processing helmet {i+1}/{len(clicked_points)}...")
                            polygon = process_point_with_sam(current_image, point, img_name, i+1)
                            if polygon is not None:
                                all_polygons.append(polygon)
                            processed_points += 1
                    
                    if save_image_annotations(current_image, all_polygons, img_name):
                        annotated_count += 1
                    break
                elif skip_image:
                    print(f"Skipped {img_name}\n")
                    skipped_count += 1
                    break
                else:
                    print("No annotations. Right-click/Right Arrow to skip or left-click to annotate helmets.")
            
            if len(clicked_points) > processed_points:
                for i in range(processed_points, len(clicked_points)):
                    print(f"\nProcessing helmet {i+1} at {clicked_points[i]}...")
                    polygon = process_point_with_sam(current_image, clicked_points[i], img_name, i+1)
                    if polygon is not None:
                        all_polygons.append(polygon)
                    processed_points += 1
            
            if skip_image:
                print(f"Skipped {img_name}\n")
                skipped_count += 1
                break
    
    cv2.destroyAllWindows()
    
    print(f"Annotation complete!")
    print(f"  Total images processed: {total_images}")
    print(f"  Annotated: {annotated_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"\nDataset saved in:")
    print(f"  Images: {OUTPUT_IMG_DIR}")
    print(f"  Labels: {OUTPUT_LABEL_DIR}")
    print(f"  Previews: {PREVIEW_DIR}")

if __name__ == "__main__":
    main()
