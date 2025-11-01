import os
import yaml
from ultralytics import YOLO
import shutil

DATASET_ROOT = 'helmet_dataset'
IMAGES_DIR = os.path.join(DATASET_ROOT, 'images')
LABELS_DIR = os.path.join(DATASET_ROOT, 'labels')

TRAIN_RATIO = 1

def create_dataset_structure():
    """Create YOLO dataset structure with train/val splits"""
    print("Creating YOLO dataset structure...")
    
    train_img_dir = os.path.join(DATASET_ROOT, 'train', 'images')
    train_lbl_dir = os.path.join(DATASET_ROOT, 'train', 'labels')
    val_img_dir = os.path.join(DATASET_ROOT, 'val', 'images')
    val_lbl_dir = os.path.join(DATASET_ROOT, 'val', 'labels')
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)
    
    images = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith('.jpg')])
    total_images = len(images)
    
    if total_images == 0:
        print("Error: No images found in dataset!")
        return False
    
    print(f"Found {total_images} annotated images")
    
    split_idx = int(total_images * TRAIN_RATIO)
    train_images = images[:split_idx]
    val_images = images[:split_idx]
    
    print(f"Train set: {len(train_images)} images")
    print(f"Val set: {len(val_images)} images")
    
    for img_name in train_images:
        lbl_name = img_name.replace('.jpg', '.txt')
        
        src_img = os.path.join(IMAGES_DIR, img_name)
        dst_img = os.path.join(train_img_dir, img_name)
        shutil.copy2(src_img, dst_img)
        
        src_lbl = os.path.join(LABELS_DIR, lbl_name)
        dst_lbl = os.path.join(train_lbl_dir, lbl_name)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)
    
    for img_name in val_images:
        lbl_name = img_name.replace('.jpg', '.txt')
        
        src_img = os.path.join(IMAGES_DIR, img_name)
        dst_img = os.path.join(val_img_dir, img_name)
        shutil.copy2(src_img, dst_img)
        
        src_lbl = os.path.join(LABELS_DIR, lbl_name)
        dst_lbl = os.path.join(val_lbl_dir, lbl_name)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)
    
    print("Dataset structure created successfully\n")
    return True

def create_yaml_config():
    """Create YAML configuration file for YOLO training"""
    yaml_content = {
        'path': os.path.abspath(DATASET_ROOT),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 1,  # Number of classes
        'names': ['helmet']  # Class names
    }
    
    yaml_path = os.path.join(DATASET_ROOT, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Created config file: {yaml_path}\n")
    return yaml_path

def train_yolo11_segmentation(yaml_path):
    """Train YOLO11 segmentation model"""
    print("Starting YOLO11 Segmentation Training")
    
    print("Loading YOLO11n-seg model (this may download the model if not cached)...")
    model = YOLO('yolo11n-seg.pt')  
    
    print("Model loaded successfully\n")
    
    # Training parameters
    results = model.train(
        data=yaml_path,
        epochs=100,              
        imgsz=640,               
        batch=16,                
        patience=20,             
        save=True,               
        device=0,                
        workers=8,               
        project='runs/segment',  
        name='helmet_detector',  
        exist_ok=True,           
        pretrained=True,         
        optimizer='auto',        
        verbose=True,            
        seed=42,                 
        deterministic=True,      
        single_cls=True,         
        rect=False,              
        cos_lr=False,            
        close_mosaic=10,         
        resume=False,            
        amp=True,                
        fraction=1.0,            
        profile=False,           
        
        hsv_h=0.015,             
        hsv_s=0.7,               
        hsv_v=0.4,               
        degrees=0.0,             
        translate=0.1,           
        scale=0.5,               
        shear=0.0,               
        perspective=0.0,         
        flipud=0.0,              
        fliplr=0.5,              
        mosaic=1.0,              
        mixup=0.0,               
        copy_paste=0.0,          
    )
    
    print("Training completed!")
    
    return results

def validate_model(model_path, yaml_path):
    print("\nValidating model...")
    model = YOLO(model_path)
    
    metrics = model.val(
        data=yaml_path,
        imgsz=640,
        batch=16,
        device=0,
    )
    
    print("\nValidation Metrics:")
    print(f"  mAP50: {metrics.seg.map50:.4f}")
    print(f"  mAP50-95: {metrics.seg.map:.4f}")
    print(f"  Precision: {metrics.seg.mp:.4f}")
    print(f"  Recall: {metrics.seg.mr:.4f}")
    
    return metrics

def export_model(model_path):
    """Export model to different formats"""
    print("\nExporting model...")
    model = YOLO(model_path)
    
    model.export(format='onnx', dynamic=True, simplify=True)
    print("Exported to ONNX")
    
    model.export(format='torchscript')
    print("Exported to TorchScript")
    
    print("\nAll exports completed!")

def main():
    print("YOLO11 Helmet Segmentation Training Pipeline")
    
    if not create_dataset_structure():
        return
    
    yaml_path = create_yaml_config()
    results = train_yolo11_segmentation(yaml_path)
    
    best_model = 'runs/segment/helmet_detector/weights/best.pt'
    if os.path.exists(best_model):
        validate_model(best_model, yaml_path)
        
        export_choice = input("\nDo you want to export the model? (y/n): ")
        if export_choice.lower() == 'y':
            export_model(best_model)
    
    print(f"\nBest model saved at: {best_model}")
    print(f"Training results: runs/segment/helmet_detector/")
    print("\nTo use the model for inference:")
    print("  from ultralytics import YOLO")
    print(f"  model = YOLO('{best_model}')")
    print("  results = model('path/to/image.jpg')")

if __name__ == "__main__":
    main()