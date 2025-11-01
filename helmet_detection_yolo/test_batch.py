import cv2
import numpy as np
from ultralytics import YOLO
import os
import glob
from pathlib import Path

# Configuration
MODEL_PATH = 'runs/segment/helmet_detector/weights/best.pt'
INPUT_DIR = 'helmet_dataset/input_images'  # For batch image testing
OUTPUT_DIR = 'test_results'
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.7

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colors
HELMET_COLOR = (0, 255, 0)  # Green
NO_HELMET_COLOR = (0, 0, 255)  # Red (for future use)

def draw_segmentation_result(img, result, show_confidence=True, show_masks=True, show_boxes=True):
    """Draw segmentation results on image"""
    img_display = img.copy()
    
    if len(result.boxes) == 0:
        # No detections
        cv2.putText(img_display, "No Helmets Detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return img_display
    
    # Get detection data
    masks = result.masks.data.cpu().numpy() if result.masks is not None else None
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    
    # Create overlay for semi-transparent masks
    overlay = img_display.copy()
    
    for i, (box, conf) in enumerate(zip(boxes, confidences)):
        # Draw mask if available
        if masks is not None and show_masks:
            mask = masks[i]
            mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
            mask_binary = (mask_resized > 0.5).astype(np.uint8)
            
            # Apply colored overlay
            overlay[mask_binary == 1] = overlay[mask_binary == 1] * 0.5 + np.array(HELMET_COLOR) * 0.5
            
            # Draw contours
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_display, contours, -1, HELMET_COLOR, 2)
        
        # Draw bounding box
        if show_boxes:
            cv2.rectangle(img_display, 
                         (int(box[0]), int(box[1])), 
                         (int(box[2]), int(box[3])), 
                         HELMET_COLOR, 2)
        
        # Add label
        if show_confidence:
            label = f"Helmet {i+1}: {conf:.2f}"
        else:
            label = f"Helmet {i+1}"
        
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_display,
                     (int(box[0]), int(box[1]) - label_size[1] - 10),
                     (int(box[0]) + label_size[0], int(box[1])),
                     HELMET_COLOR, -1)
        cv2.putText(img_display, label,
                   (int(box[0]), int(box[1]) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Blend overlay with original
    img_display = cv2.addWeighted(img_display, 0.7, overlay, 0.3, 0)
    
    # Add summary text
    summary = f"Total Helmets: {len(result.boxes)}"
    cv2.putText(img_display, summary, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, HELMET_COLOR, 2)
    
    return img_display

def test_single_image(model, image_path, save=True, display=True):
    """Test on a single image"""
    print(f"\nProcessing: {os.path.basename(image_path)}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"  Error: Could not load image")
        return None
    
    # Run inference
    results = model(image_path, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
    result = results[0]
    
    # Print results
    print(f"  Detected: {len(result.boxes)} helmet(s)")
    if len(result.boxes) > 0:
        confidences = result.boxes.conf.cpu().numpy()
        for i, conf in enumerate(confidences):
            print(f"    Helmet {i+1}: confidence {conf:.4f}")
    
    # Visualize
    img_display = draw_segmentation_result(img, result)
    
    # Save
    if save:
        output_filename = os.path.basename(image_path).replace('.', '_result.')
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        cv2.imwrite(output_path, img_display)
        print(f"  ✓ Saved to: {output_path}")
    
    # Display
    if display:
        cv2.imshow('Helmet Detection', img_display)
        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()
            return False  # Signal to stop
    
    return True

def test_batch_images(model, input_dir):
    """Test on all images in directory"""
    # Get all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"\nFound {len(image_files)} images")
    print("="*60)
    
    total_helmets = 0
    
    for img_path in image_files:
        continue_testing = test_single_image(model, img_path, save=True, display=True)
        if not continue_testing:
            break
        
        # Count total helmets
        results = model(img_path, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        total_helmets += len(results[0].boxes)
    
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("Batch Testing Complete!")
    print(f"  Total images processed: {len(image_files)}")
    print(f"  Total helmets detected: {total_helmets}")
    print(f"  Average helmets per image: {total_helmets/len(image_files):.2f}")
    print("="*60)

def test_video(model, video_path, output_path=None, display=True):
    """Test on video file"""
    print(f"\nProcessing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video")
        return
    
    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    
    # Video writer
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, 'video_result.mp4')
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_helmets = 0
    
    print("\nProcessing frames... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run inference
        results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        result = results[0]
        
        # Visualize
        frame_display = draw_segmentation_result(frame, result)
        
        # Add frame info
        info = f"Frame: {frame_count}/{total_frames}"
        cv2.putText(frame_display, info, (width - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame_display)
        
        # Display
        if display:
            cv2.imshow('Helmet Detection - Video', frame_display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Count helmets
        total_helmets += len(result.boxes)
        
        # Progress
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...", end='\r')
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n✓ Video saved to: {output_path}")
    print(f"  Total helmets detected: {total_helmets}")
    print(f"  Average helmets per frame: {total_helmets/frame_count:.2f}")

def main():
    print("="*60)
    print("YOLO11 Helmet Detection - Testing Script")
    print("="*60)
    
    # Load model
    print(f"\nLoading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("✓ Model loaded successfully!\n")
    
    # Menu
    print("Select testing mode:")
    print("  1. Single image")
    print("  2. Batch images (all in folder)")
    print("  3. Video file")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        img_path = input("Enter image path: ").strip()
        if os.path.exists(img_path):
            test_single_image(model, img_path, save=True, display=True)
            cv2.destroyAllWindows()
        else:
            print(f"Error: File not found: {img_path}")
    
    elif choice == '2':
        if os.path.exists(INPUT_DIR):
            test_batch_images(model, INPUT_DIR)
        else:
            print(f"Error: Directory not found: {INPUT_DIR}")
    
    elif choice == '3':
        video_path = input("Enter video path: ").strip()
        if os.path.exists(video_path):
            test_video(model, video_path, display=True)
        else:
            print(f"Error: File not found: {video_path}")
    
    else:
        print("Invalid choice")
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)

if __name__ == "__main__":
    main()
