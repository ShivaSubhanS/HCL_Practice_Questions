import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import shutil
from sklearn.model_selection import StratifiedKFold
import yaml
from ultralytics import YOLO
import gc
import torch

def rle_to_mask(rle_string, height=256, width=1600):
    if pd.isna(rle_string):
        return np.zeros((height, width), dtype=np.uint8)
    rle_numbers = [int(x) for x in rle_string.split()]
    pairs = np.array(rle_numbers).reshape(-1, 2)
    mask = np.zeros(height * width, dtype=np.uint8)
    for start, length in pairs:
        start = start - 1
        mask[start:start + length] = 1
    return mask.reshape((height, width), order='F')

def mask_to_yolo_polygon(mask, min_area=50):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return []
    height, width = mask.shape
    polygons = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        if len(contour) < 3:
            continue
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue
        polygon = approx.flatten().tolist()
        normalized = []
        for i in range(0, len(polygon), 2):
            x_norm = polygon[i] / width
            y_norm = polygon[i + 1] / height
            normalized.extend([x_norm, y_norm])
        polygons.append(normalized)
    return polygons

def analyze_dataset_statistics(train_csv_path):
    df = pd.read_csv(train_csv_path)
    class_counts = df[df['EncodedPixels'].notna()].groupby('ClassId').size()
    total_defects = class_counts.sum()
    for class_id in sorted(df['ClassId'].unique()):
        count = class_counts.get(class_id, 0)
        percentage = (count / total_defects * 100) if total_defects > 0 else 0
        print(f"  Class {class_id}: {count:5d} defects ({percentage:5.2f}%)")
    total_images = df['ImageId'].nunique()
    images_with_defects = df[df['EncodedPixels'].notna()]['ImageId'].nunique()
    images_without_defects = total_images - images_with_defects
    print(f"  Total images: {total_images}")
    print(f"  Images with defects: {images_with_defects} ({images_with_defects/total_images*100:.2f}%)")
    print(f"  Images without defects: {images_without_defects} ({images_without_defects/total_images*100:.2f}%)")
    defects_per_image = df[df['EncodedPixels'].notna()].groupby('ImageId').size()
    print(f"  Average: {defects_per_image.mean():.2f}")
    print(f"  Max: {defects_per_image.max()}")
    print(f"  Min: {defects_per_image.min()}")
    print(f"\nAnalyzing defect sizes...")
    defect_sizes = []
    for _, row in df[df['EncodedPixels'].notna()].iterrows():
        mask = rle_to_mask(row['EncodedPixels'])
        defect_area = np.sum(mask)
        defect_sizes.append(defect_area)
    defect_sizes = np.array(defect_sizes)
    print(f"  Mean defect size: {defect_sizes.mean():.0f} pixels")
    print(f"  Median defect size: {np.median(defect_sizes):.0f} pixels")
    print(f"  Min defect size: {defect_sizes.min():.0f} pixels")
    print(f"  Max defect size: {defect_sizes.max():.0f} pixels")
    return {
        'total_images': total_images,
        'images_with_defects': images_with_defects,
        'class_distribution': class_counts.to_dict()
    }

def create_stratified_split(df, n_splits=5, val_fold=0):
    image_labels = df.groupby('ImageId')['EncodedPixels'].apply(
        lambda x: 1 if x.notna().any() else 0
    ).reset_index()
    image_labels.columns = ['ImageId', 'has_defect']
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(skf.split(image_labels['ImageId'], image_labels['has_defect']))
    train_idx, val_idx = splits[val_fold]
    train_images = image_labels.iloc[train_idx]['ImageId'].values
    val_images = image_labels.iloc[val_idx]['ImageId'].values
    return train_images, val_images

def prepare_yolo_dataset(data_root, output_root, train_csv_path,
                        use_stratified=True, val_fold=0, val_split=0.2):
    data_root = Path(data_root)
    output_root = Path(output_root)
    dirs = {
        'train_images': output_root / 'images' / 'train',
        'val_images': output_root / 'images' / 'val',
        'train_labels': output_root / 'labels' / 'train',
        'val_labels': output_root / 'labels' / 'val'
    }
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    print("Loading train.csv...")
    df = pd.read_csv(train_csv_path)
    grouped = df.groupby('ImageId')
    if use_stratified:
        train_images, val_images = create_stratified_split(df, n_splits=5, val_fold=val_fold)
    else:
        from sklearn.model_selection import train_test_split
        all_images = df['ImageId'].unique()
        train_images, val_images = train_test_split(
            all_images, test_size=val_split, random_state=42
        )
    print(f"Total images: {len(df['ImageId'].unique())}")
    print(f"Train images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")
    process_split(
        train_images, grouped, data_root / 'train_images',
        dirs['train_images'], dirs['train_labels']
    )
    process_split(
        val_images, grouped, data_root / 'train_images',
        dirs['val_images'], dirs['val_labels']
    )
    create_yaml_config(output_root)
    return output_root

def process_split(image_list, grouped_df, source_img_dir, target_img_dir, target_label_dir):
    stats = {'total': 0, 'with_defects': 0, 'without_defects': 0, 'total_defects': 0}
    for img_name in tqdm(image_list):
        src_img = source_img_dir / img_name
        dst_img = target_img_dir / img_name
        if not src_img.exists():
            print(f"Warning: Image {img_name} not found")
            continue
        shutil.copy2(src_img, dst_img)
        stats['total'] += 1
        if img_name not in grouped_df.groups:
            label_path = target_label_dir / img_name.replace('.jpg', '.txt')
            label_path.touch()
            stats['without_defects'] += 1
            continue
        img_defects = grouped_df.get_group(img_name)
        label_path = target_label_dir / img_name.replace('.jpg', '.txt')
        has_defects = False
        with open(label_path, 'w') as f:
            for _, row in img_defects.iterrows():
                class_id = int(row['ClassId']) - 1
                rle = row['EncodedPixels']
                if pd.isna(rle):
                    continue
                mask = rle_to_mask(rle)
                polygons = mask_to_yolo_polygon(mask, min_area=50)
                for polygon in polygons:
                    if len(polygon) >= 6:
                        coords_str = ' '.join([f'{coord:.6f}' for coord in polygon])
                        f.write(f"{class_id} {coords_str}\n")
                        has_defects = True
                        stats['total_defects'] += 1
        if has_defects:
            stats['with_defects'] += 1
        else:
            stats['without_defects'] += 1
    print(f"  Total images: {stats['total']}")
    print(f"  With defects: {stats['with_defects']}")
    print(f"  Without defects: {stats['without_defects']}")
    print(f"  Total defect instances: {stats['total_defects']}")

def create_yaml_config(output_root):
    config = {
        'path': str(output_root.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'class_1',
            1: 'class_2',
            2: 'class_3',
            3: 'class_4'
        },
        'nc': 4
    }
    yaml_path = output_root / 'steel_defect.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Created YAML config at {yaml_path}")

def train_yolo11_seg(yaml_path, epochs=100, imgsz=1280, batch=4, device=0,
                    project='runs/segment', model_size='s', pretrained=True):
    
    model_name = f'yolo11{model_size}-seg.pt' if pretrained else f'yolo11{model_size}-seg.yaml'
    print(f"Using model: {model_name}")
    model = YOLO(model_name)
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name='steel_defect_yolo11',
        patience=5,
        save=True,
        plots=True,
        exist_ok=True,
        cache=True,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.0001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.3,
        dfl=1.5,
        hsv_h=0.005,
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=3.0,
        translate=0.05,
        scale=0.2,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        rect=False,
        overlap_mask=True,
        mask_ratio=4,
        amp=True,
        fraction=1.0,
        workers=8,
        seed=42,
        close_mosaic=20,
        val=True,
        save_period=10,
        nbs=64,
        label_smoothing=0.0,
    )
    return results

def prepare_classification_dataset(data_root, output_root, train_csv_path, val_split=0.2):
    data_root = Path(data_root)
    output_root = Path(output_root)
    for split in ['train', 'val']:
        for cls in ['defect', 'no_defect']:
            (output_root / split / cls).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(train_csv_path)
    images_with_defects = set(df[df['EncodedPixels'].notna()]['ImageId'].unique())
    all_images = set(df['ImageId'].unique())
    images_without_defects = all_images - images_with_defects
    from sklearn.model_selection import train_test_split
    defect_train, defect_val = train_test_split(
        list(images_with_defects), test_size=val_split, random_state=42
    )
    no_defect_train, no_defect_val = train_test_split(
        list(images_without_defects), test_size=val_split, random_state=42
    )
    source_dir = data_root / 'train_images'
    for img_name in tqdm(defect_train):
        shutil.copy2(source_dir / img_name, output_root / 'train' / 'defect' / img_name)
    for img_name in tqdm(defect_val):
        shutil.copy2(source_dir / img_name, output_root / 'val' / 'defect' / img_name)
    for img_name in tqdm(no_defect_train):
        shutil.copy2(source_dir / img_name, output_root / 'train' / 'no_defect' / img_name)
    for img_name in tqdm(no_defect_val):
        shutil.copy2(source_dir / img_name, output_root / 'val' / 'no_defect' / img_name)
    print(f"Classification dataset saved at {output_root}")

def predict_on_test_images(model_path, test_images_dir, output_dir,
                           conf_threshold=0.25, iou_threshold=0.45):
    model = YOLO(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_images_dir = Path(test_images_dir)
    test_images = sorted(test_images_dir.glob('*.jpg'))
    print(f"Running inference on {len(test_images)} test images")
    for idx, img_path in enumerate(tqdm(test_images)):
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            iou=iou_threshold,
            save=False,
            verbose=False,
            imgsz=1024
        )
        for result in results:
            save_path = output_dir / img_path.name
            result.save(filename=str(save_path))
        if (idx + 1) % 100 == 0:
            del results
            gc.collect()
            torch.cuda.empty_cache()

def create_submission_csv(model_path, test_images_dir, output_csv,
                         conf_threshold=0.25, iou_threshold=0.45):
    def mask_to_rle(mask):
        pixels = mask.T.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)
    model = YOLO(model_path)
    test_images_dir = Path(test_images_dir)
    test_images = sorted(test_images_dir.glob('*.jpg'))
    submission_data = []
    print(f"Generating submission for {len(test_images)} test images...")
    
    for idx, img_path in enumerate(tqdm(test_images)):
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
            imgsz=1024
        )
        img_name = img_path.name
        found_defects = False
        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                for mask, cls in zip(masks, classes):
                    mask_resized = cv2.resize(
                        mask.astype(np.uint8), (1600, 256),
                        interpolation=cv2.INTER_NEAREST
                    )
                    if np.sum(mask_resized) > 50:
                        rle = mask_to_rle(mask_resized)
                        class_id = cls + 1
                        submission_data.append({
                            'ImageId': img_name,
                            'EncodedPixels': rle,
                            'ClassId': class_id
                        })
                        found_defects = True
        if (idx + 1) % 100 == 0:
            del results
            gc.collect()
            torch.cuda.empty_cache()
        if not found_defects:
            submission_data.append({
                'ImageId': img_name,
                'EncodedPixels': '1 1',
                'ClassId': 0
            })
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(output_csv, index=False)
    print(f"Submission saved to {output_csv}")

if __name__ == "__main__":
    DATA_ROOT = "/mnt/d/code/hcl/steel_defect_yolo/severstal-steel-defect-detection"
    OUTPUT_ROOT = "/mnt/d/code/hcl/steel_defect_yolo/yolo_dataset"
    TRAIN_CSV = f"{DATA_ROOT}/train.csv"
    stats = analyze_dataset_statistics(TRAIN_CSV)
    prepare_yolo_dataset(
        data_root=DATA_ROOT,
        output_root=OUTPUT_ROOT,
        train_csv_path=TRAIN_CSV,
        use_stratified=True,
        val_fold=0
    )
    yaml_config = f"{OUTPUT_ROOT}/steel_defect.yaml"
    train_yolo11_seg(
        yaml_path=yaml_config,
        epochs=35,
        imgsz=1024,
        batch=6,
        device=0,
        project='runs/segment',
        model_size='s',
        pretrained=True
    )
    BEST_MODEL = "runs/segment/steel_defect_yolo11/weights/best.pt"
    TEST_IMAGES = f"{DATA_ROOT}/test_images"
    OUTPUT_DIR = "predictions"
    predict_on_test_images(
        model_path=BEST_MODEL,
        test_images_dir=TEST_IMAGES,
        output_dir=OUTPUT_DIR,
        conf_threshold=0.25,
        iou_threshold=0.45
    )
    create_submission_csv(
        model_path=BEST_MODEL,
        test_images_dir=TEST_IMAGES,
        output_csv="submission.csv",
        conf_threshold=0.25,
        iou_threshold=0.45
    )