import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

TRAIN_DATA_DIR = "./NEU-DET/train/images"
VAL_DATA_DIR = "./NEU-DET/validation/images"
MODEL_NAME = "google/vit-base-patch16-224"
CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NEUDataset(Dataset):
    def __init__(self, image_paths, labels, processor, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        self.augment = augment
        
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
            ])
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        encoding = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding['pixel_values'].squeeze()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {'pixel_values': pixel_values, 'labels': label}

X_train, y_train = [], []
for class_idx, class_name in enumerate(CLASS_NAMES):
    class_dir = os.path.join(TRAIN_DATA_DIR, class_name)
    for img_name in os.listdir(class_dir):
        if img_name.endswith(('.jpg', '.png', '.bmp')):
            X_train.append(os.path.join(class_dir, img_name))
            y_train.append(class_idx)

X_val, y_val = [], []
for class_idx, class_name in enumerate(CLASS_NAMES):
    class_dir = os.path.join(VAL_DATA_DIR, class_name)
    for img_name in os.listdir(class_dir):
        if img_name.endswith(('.jpg', '.png', '.bmp')):
            X_val.append(os.path.join(class_dir, img_name))
            y_val.append(class_idx)

print(f"Total train images: {len(X_train)}")
print(f"Total validation images: {len(X_val)}")

X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42, stratify=y_val)

processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
model = ViTForImageClassification.from_pretrained(
    MODEL_NAME, num_labels=6, ignore_mismatched_sizes=True
).to(DEVICE)

train_dataset = NEUDataset(X_train, y_train, processor, augment=True)
val_dataset = NEUDataset(X_val, y_val, processor, augment=False)
test_dataset = NEUDataset(X_test, y_test, processor, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, 
    patience=3, verbose=True
)

best_val_acc = 0.0
epochs_without_improvement = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss, train_preds, train_labels = 0, [], []
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
        pixel_values = batch['pixel_values'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=-1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())
    
    train_loss /= len(train_loader)
    train_acc = accuracy_score(train_labels, train_preds)
    
    model.eval()
    val_loss, val_preds, val_labels = 0, [], []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
            pixel_values = batch['pixel_values'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            val_loss += outputs.loss.item()
            
            preds = torch.argmax(outputs.logits, dim=-1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    val_loss /= len(val_loader)
    val_acc = accuracy_score(val_labels, val_preds)
    
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
    
    scheduler.step(val_acc)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_without_improvement = 0
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
        }, 'best_vit_model.pth')
        print(f"Saved best model (Val Acc: {val_acc:.4f})")
    else:
        epochs_without_improvement += 1
        print(f"No improvement for {epochs_without_improvement} epoch(s)")
        
        if epochs_without_improvement >= 5:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best validation accuracy: {best_val_acc:.4f}")
            break

checkpoint = torch.load('best_vit_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

test_preds, test_labels = [], []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        pixel_values = batch['pixel_values'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        outputs = model(pixel_values=pixel_values)
        preds = torch.argmax(outputs.logits, dim=-1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_acc = accuracy_score(test_labels, test_preds)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Best Val Accuracy: {best_val_acc:.4f}")
print(f"Best model from epoch: {checkpoint['epoch']}")