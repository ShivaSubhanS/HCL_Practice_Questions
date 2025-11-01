import cv2
import numpy as np
from ultralytics import YOLO
import os

HELMET_MODEL_PATH = 'runs/segment/helmet_detector/weights/best.pt'
INPUT_PATH = '/mnt/d/code/hcl/helmet_detection/helmet_dataset/input_images/tiruhel.jpeg'
OUTPUT_DIR = 'test_results'
PERSON_CONF = 0.3  
HELMET_CONF = 0.016
IOU_THRESHOLD = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Loading models...")
print("  1. Loading pretrained YOLO11 for person detection...")
person_model = YOLO('yolo11n.pt') 
print("  2. Loading custom helmet detection model...")
helmet_model = YOLO(HELMET_MODEL_PATH)
print("Models loaded successfully!\n")

def calculate_iou(box1, box2):
    """Calculate IoU (Intersection over Union) between two bounding boxes"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

print(f"Processing: {INPUT_PATH}\n")
img = cv2.imread(INPUT_PATH)
if img is None:
    print(f"Error: Could not load image from {INPUT_PATH}")
    exit(1)

print("STEP 1: Person Detection")
person_results = person_model(INPUT_PATH, conf=PERSON_CONF, iou=0.7, classes=[0])  # class 0 = person
person_result = person_results[0]

person_boxes = []
if len(person_result.boxes) > 0:
    person_boxes_raw = person_result.boxes.xyxy.cpu().numpy()
    person_confidences = person_result.boxes.conf.cpu().numpy()
    
    print(f"Detected {len(person_boxes_raw)} person(s):")
    for i, (box, conf) in enumerate(zip(person_boxes_raw, person_confidences)):
        person_boxes.append(box)
        print(f"  Person {i+1}: Confidence={conf:.3f}, BBox=[{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]")
else:
    print("No persons detected")

print("STEP 2: Helmet Detection (Custom Model)")
helmet_results = helmet_model(INPUT_PATH, conf=HELMET_CONF, iou=0.7)
helmet_result = helmet_results[0]

helmet_boxes = []
helmet_masks = []
if len(helmet_result.boxes) > 0:
    helmet_boxes_raw = helmet_result.boxes.xyxy.cpu().numpy()
    helmet_confidences = helmet_result.boxes.conf.cpu().numpy()
    
    print(f"Detected {len(helmet_boxes_raw)} helmet(s):")
    for i, (box, conf) in enumerate(zip(helmet_boxes_raw, helmet_confidences)):
        helmet_boxes.append(box)
        print(f"  Helmet {i+1}: Confidence={conf:.3f}, BBox=[{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]")
    
    if helmet_result.masks is not None:
        helmet_masks = helmet_result.masks.data.cpu().numpy()
else:
    print("No helmets detected")

print("STEP 3: Comparison Analysis")

matches = []
if len(person_boxes) > 0 and len(helmet_boxes) > 0:
    print(f"Comparing {len(person_boxes)} person(s) with {len(helmet_boxes)} helmet(s)...\n")
    
    for i, person_box in enumerate(person_boxes):
        for j, helmet_box in enumerate(helmet_boxes):
            iou = calculate_iou(person_box, helmet_box)
            if iou >= IOU_THRESHOLD:
                matches.append((i, j, iou))
                print(f"MATCH: Person {i+1} ↔ Helmet {j+1} (IoU: {iou:.3f})")
    
    if matches:
        print(f"\nFound {len(matches)} matching region(s)!")
        print("  → Person detection and helmet detection agree on these regions")
    else:
        print("\nNo matches found")
        print("  → Detections are in different regions (IoU < threshold)")
else:
    if len(person_boxes) == 0 and len(helmet_boxes) == 0:
        print("SAME: Both models detected nothing")
        print("  → Results agree: no detections")
    elif len(person_boxes) == 0:
        print("DIFFERENT: Person model detected nothing, but helmet model found helmets")
    else:
        print("DIFFERENT: Person model found persons, but helmet model detected nothing")

print("Creating Visualization")

img_person = img.copy()
for i, box in enumerate(person_boxes):
    cv2.rectangle(img_person,
                 (int(box[0]), int(box[1])),
                 (int(box[2]), int(box[3])),
                 (255, 0, 0), 3)  
    cv2.putText(img_person, f"Person {i+1}",
               (int(box[0]), int(box[1]) - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

img_display = img_person.copy()
overlay = img_person.copy()

if len(helmet_masks) > 0:
    for i, (box, mask) in enumerate(zip(helmet_boxes, helmet_masks)):
        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        
        overlay[mask_binary == 1] = overlay[mask_binary == 1] * 0.5 + np.array([0, 255, 0]) * 0.5
        
        cv2.rectangle(img_display,
                     (int(box[0]), int(box[1])),
                     (int(box[2]), int(box[3])),
                     (0, 255, 0), 3)  
        cv2.putText(img_display, f"Helmet {i+1}",
                   (int(box[0]), int(box[1]) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

img_display = cv2.addWeighted(img_display, 0.7, overlay, 0.3, 0)

for person_idx, helmet_idx, iou in matches:
    p_box = person_boxes[person_idx]
    h_box = helmet_boxes[helmet_idx]
    
    p_center = (int((p_box[0] + p_box[2]) / 2), int((p_box[1] + p_box[3]) / 2))
    h_center = (int((h_box[0] + h_box[2]) / 2), int((h_box[1] + h_box[3]) / 2))
    
    cv2.line(img_display, p_center, h_center, (0, 0, 255), 2)
    cv2.putText(img_display, f"IoU:{iou:.2f}",
               ((p_center[0] + h_center[0])//2, (p_center[1] + h_center[1])//2),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

summary_y = 30
cv2.putText(img_display, f"Persons: {len(person_boxes)} | Helmets: {len(helmet_boxes)} | Matches: {len(matches)}",
           (10, summary_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

legend_y = img.shape[0] - 60
cv2.putText(img_display, "Blue = Person | Green = Helmet | Red Line = Match",
           (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

base_name = os.path.basename(INPUT_PATH)
name_without_ext = os.path.splitext(base_name)[0]
ext = os.path.splitext(base_name)[1]

person_output = os.path.join(OUTPUT_DIR, f"{name_without_ext}_1_person{ext}")
cv2.imwrite(person_output, img_person)
print(f"Person detection saved to: {person_output}")

img_helmet = img.copy()
overlay_helmet = img.copy()
if len(helmet_masks) > 0:
    for i, (box, mask) in enumerate(zip(helmet_boxes, helmet_masks)):
        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        overlay_helmet[mask_binary == 1] = overlay_helmet[mask_binary == 1] * 0.5 + np.array([0, 255, 0]) * 0.5
        cv2.rectangle(img_helmet,
                     (int(box[0]), int(box[1])),
                     (int(box[2]), int(box[3])),
                     (0, 255, 0), 3)
        cv2.putText(img_helmet, f"Helmet {i+1}",
                   (int(box[0]), int(box[1]) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    img_helmet = cv2.addWeighted(img_helmet, 0.7, overlay_helmet, 0.3, 0)

helmet_output = os.path.join(OUTPUT_DIR, f"{name_without_ext}_2_helmet{ext}")
cv2.imwrite(helmet_output, img_helmet)
print(f"Helmet detection saved to: {helmet_output}")

comparison_output = os.path.join(OUTPUT_DIR, f"{name_without_ext}_3_comparison{ext}")
cv2.imwrite(comparison_output, img_display)
print(f"Combined comparison saved to: {comparison_output}")

print("\nDisplaying results...")
print("  Step 1: Person Detection (Blue) - Press any key to continue")
cv2.imshow('Step 1: Person Detection', img_person)
cv2.waitKey(0)

print("  Step 2: Helmet Detection (Green) - Press any key to continue")
cv2.imshow('Step 2: Helmet Detection', img_helmet)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(matches) > 0:
    print(f"SAME: {len(matches)} matching region(s) detected by both models")
elif len(person_boxes) == 0 and len(helmet_boxes) == 0:
    print("SAME: Both models detected nothing")
else:
    print("DIFFERENT: Models detected different regions or counts")
    print(f"  Persons: {len(person_boxes)}, Helmets: {len(helmet_boxes)}")