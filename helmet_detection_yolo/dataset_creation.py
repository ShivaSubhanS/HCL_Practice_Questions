import os
import cv2
import torch
import numpy as np
import sys
sys.path.append("/mnt/d/code/hcl/smart_traffic_light/sam2")
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2

# Paths
OUTPUT_IMG_DIR = 'helmet_dataset/images'
OUTPUT_LABEL_DIR = 'helmet_dataset/labels'
PREVIEW_DIR = 'helmet_dataset/preview'
VIDEO_FILE = 'helmet.mp4'  # Change this to your video file

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load SAM 2.1 Hiera Large model from local checkpoint
try:
    # Build SAM2 model from local checkpoint
    sam2_checkpoint = "sam21-hiera-large"  # Your downloaded checkpoint file
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"  # Config for hiera-large
    
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    print("SAM 2.1 Hiera Large model loaded successfully from local checkpoint.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

# Global variables for mouse interaction
clicked_points = []
skip_frame = False
current_frame = None
all_polygons = []
window_name = "SAM2 Interactive Annotation - Left Click: Segment | Space: Next | Right Arrow: Skip"

def mask_to_polygon(mask):
    """Convert binary mask to polygon coordinates"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Select the largest contour
    max_contour = max(contours, key=cv2.contourArea)
    return max_contour.reshape(-1, 2)

def save_mask_preview(img, masks, scores, point, frame_idx, point_num, selected_idx):
    """Save visualization of all masks with selected one highlighted"""
    height, width = img.shape[:2]
    display_width = width * min(3, len(masks))
    display_img = np.zeros((height, display_width, 3), dtype=np.uint8)
    
    colors = [(0, 255, 0), (255, 255, 0), (255, 0, 255)]  # Green, Yellow, Magenta
    
    for i, (mask, color, score) in enumerate(zip(masks, colors, scores)):
        start_x = i * width
        display_img[:, start_x:start_x+width] = img.copy()
        
        overlay = np.zeros_like(img)
        overlay[mask > 0] = color
        display_img[:, start_x:start_x+width] = cv2.addWeighted(
            display_img[:, start_x:start_x+width], 0.7, overlay, 0.3, 0
        )
        
        # Draw click point
        cv2.circle(display_img, (point[0] + start_x, point[1]), 5, (255, 0, 0), -1)
        
        # Add label with score
        label = f'Mask {i} ({score:.3f})'
        cv2.putText(display_img, label, (start_x + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Highlight selected mask
        if i == selected_idx:
            cv2.rectangle(display_img, (start_x, 0), (start_x + width, height), 
                         (0, 0, 255), 5)
            cv2.putText(display_img, 'SELECTED', (start_x + 10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    preview_path = os.path.join(PREVIEW_DIR, f"frame_{frame_idx:05d}_helmet{point_num}.jpg")
    cv2.imwrite(preview_path, display_img)
    print(f"Saved preview to {preview_path}")

def mouse_callback(event, x, y, flags, param):
    """Handle mouse events"""
    global clicked_points, skip_frame, current_frame, all_polygons
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Left click - segment helmet at this point
        clicked_points.append((x, y))
        print(f"Click {len(clicked_points)} registered at ({x}, {y})")
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click - skip this frame
        skip_frame = True
        print("Frame skipped by user")

def process_point_with_sam(img, point, frame_idx, point_num):
    """Process a single point with SAM segmentation and return polygon"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Validate point
    if point[0] >= w or point[1] >= h or point[0] < 0 or point[1] < 0:
        print(f"Warning: Point ({point[0]}, {point[1]}) is outside image bounds.")
        return None
    
    # Set image for predictor
    predictor.set_image(img_rgb)
    
    # Prepare input
    input_point = np.array([point])
    input_label = np.array([1])  # Foreground point
    
    # Predict masks
    try:
        if device == "cuda":
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True
                )
        else:
            with torch.inference_mode():
                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True
                )
        
        print(f"  Generated {len(masks)} masks with scores: {[f'{s:.4f}' for s in scores]}")
    except Exception as e:
        print(f"  Error during prediction: {e}")
        return None
    
    # Select mask with highest score (best accuracy)
    selected_idx = np.argmax(scores)
    mask = masks[selected_idx]
    
    print(f"  Selected mask {selected_idx} with highest score {scores[selected_idx]:.4f}")
    
    # Save preview for this point
    save_mask_preview(img, masks, scores, point, frame_idx, point_num, selected_idx)
    
    # Convert mask to polygon
    polygon = mask_to_polygon(mask)
    if polygon is None:
        print(f"  Warning: No contour found for point {point_num}")
        return None
    
    print(f"  Extracted polygon with {len(polygon)} points")
    
    # Normalize coordinates
    polygon_norm = polygon.astype(np.float32)
    polygon_norm[:, 0] /= w
    polygon_norm[:, 1] /= h
    
    return polygon_norm

def save_frame_annotations(img, polygons, frame_idx):
    """Save image and all annotations for a frame"""
    if not polygons:
        print(f"No annotations to save for frame {frame_idx:05d}")
        return False
    
    # Save image
    img_path = os.path.join(OUTPUT_IMG_DIR, f"frame_{frame_idx:05d}.jpg")
    cv2.imwrite(img_path, img)
    
    # Save all labels in YOLO format
    label_path = os.path.join(OUTPUT_LABEL_DIR, f"frame_{frame_idx:05d}.txt")
    with open(label_path, 'w') as f:
        for polygon_norm in polygons:
            polygon_flat = polygon_norm.flatten()
            polygon_str = ' '.join(f'{coord:.6f}' for coord in polygon_flat)
            f.write(f"0 {polygon_str}\n")  # Class 0 for helmet
    
    print(f"✓ Saved frame with {len(polygons)} helmet(s) for frame {frame_idx:05d}\n")
    return True

def main():
    global clicked_points, skip_frame, current_frame, all_polygons
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_FILE}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\nVideo loaded: {total_frames} frames at {fps:.2f} FPS")
    print("\nControls:")
    print("  LEFT CLICK        - Segment helmet at clicked point (click multiple times for multiple helmets)")
    print("  RIGHT ARROW/CLICK - Skip current frame")
    print("  SPACE             - Save annotations and move to next frame")
    print("  Q                 - Quit annotation\n")
    
    # Create window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    frame_idx = 0
    annotated_count = 0
    skipped_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Reached end of video")
            break
        
        current_frame = frame.copy()
        clicked_points = []
        all_polygons = []
        skip_frame = False
        processed_points = 0
        
        # Wait for user action
        while True:
            # Display frame with current annotations
            display = current_frame.copy()
            
            # Draw all clicked points
            for i, point in enumerate(clicked_points):
                color = (0, 255, 0) if i < processed_points else (0, 0, 255)
                cv2.circle(display, point, 5, color, -1)
                cv2.putText(display, str(i+1), (point[0]+10, point[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw info
            cv2.putText(display, f"Frame {frame_idx}/{total_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, f"Annotated: {annotated_count} | Skipped: {skipped_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Helmets on this frame: {len(clicked_points)}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display, "Left Click: Add Helmet | Space: Save & Next | Right Arrow/Right Click: Skip | Q: Quit", 
                       (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting annotation...")
                cap.release()
                cv2.destroyAllWindows()
                print(f"\nFinal stats:")
                print(f"  Total frames processed: {frame_idx}")
                print(f"  Annotated: {annotated_count}")
                print(f"  Skipped: {skipped_count}")
                return
            
            elif key == 83:  # Right arrow key
                # Skip frame
                skip_frame = True
                print(f"Skipped frame {frame_idx:05d}\n")
                skipped_count += 1
                break
            
            elif key == ord(' '):
                # Space - save annotations and move to next frame
                if len(clicked_points) > 0:
                    # Process any unprocessed points
                    print(f"\nProcessing {len(clicked_points)} helmet(s) on frame {frame_idx:05d}...")
                    for i, point in enumerate(clicked_points):
                        if i >= processed_points:
                            print(f"Processing helmet {i+1}/{len(clicked_points)}...")
                            polygon = process_point_with_sam(current_frame, point, frame_idx, i+1)
                            if polygon is not None:
                                all_polygons.append(polygon)
                            processed_points += 1
                    
                    # Save frame with all annotations
                    if save_frame_annotations(current_frame, all_polygons, frame_idx):
                        annotated_count += 1
                    break
                elif skip_frame:
                    print(f"Skipped frame {frame_idx:05d}\n")
                    skipped_count += 1
                    break
                else:
                    print("No annotations. Right-click to skip or left-click to annotate helmets.")
            
            # Process new clicks immediately
            if len(clicked_points) > processed_points:
                for i in range(processed_points, len(clicked_points)):
                    print(f"\nProcessing helmet {i+1} at {clicked_points[i]}...")
                    polygon = process_point_with_sam(current_frame, clicked_points[i], frame_idx, i+1)
                    if polygon is not None:
                        all_polygons.append(polygon)
                    processed_points += 1
            
            # Handle skip
            if skip_frame:
                print(f"Skipped frame {frame_idx:05d}\n")
                skipped_count += 1
                break
        
        frame_idx += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nAnnotation complete!")
    print(f"  Total frames processed: {frame_idx}")
    print(f"  Annotated: {annotated_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"\nDataset saved in: {OUTPUT_IMG_DIR} and {OUTPUT_LABEL_DIR}")

if __name__ == "__main__":
    main()