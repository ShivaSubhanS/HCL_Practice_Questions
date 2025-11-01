import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import glob

IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 500
LEARNING_RATE = 0.001

class CarpetDefectAutoencoder:
    def __init__(self, img_size=256):
        self.img_size = img_size
        self.model = None
        
    def build_model(self):
        # Encoder
        inputs = layers.Input(shape=(self.img_size, self.img_size, 3))
        
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        # Decoder
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(encoded)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        
        self.model = models.Model(inputs, decoded)
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )

def load_carpet_data(data_path, img_size=256):

    train_path = os.path.join(data_path, 'carpet', 'train', 'good')
    images = []
    if os.path.exists(train_path):
        img_files = glob.glob(os.path.join(train_path, '*.png'))
        for img_file in img_files:
            img = cv2.imread(img_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            img = img.astype('float32') / 255.0
            images.append(img)
    return np.array(images)

def create_data_generator(images, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((images, images))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def train_autoencoder(data_path, output_dir='models'):
    os.makedirs(output_dir, exist_ok=True)
    
    train_images = load_carpet_data(data_path, IMG_SIZE)
    train_imgs, val_imgs = train_test_split(train_images, test_size=0.2, random_state=42)
    
    train_dataset = create_data_generator(train_imgs, BATCH_SIZE)
    val_dataset = create_data_generator(val_imgs, BATCH_SIZE)
    
    autoencoder = CarpetDefectAutoencoder(IMG_SIZE)
    autoencoder.build_model()
    autoencoder.compile_model(LEARNING_RATE)
    
    print(autoencoder.model.summary())
    
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_carpet_autoencoder.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    history = autoencoder.model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    autoencoder.model.save(os.path.join(output_dir, 'final_carpet_autoencoder.h5'))
    
    return autoencoder.model, history

if __name__ == '__main__':
    model, history = train_autoencoder('/mnt/d/code/hcl/autoencoder')