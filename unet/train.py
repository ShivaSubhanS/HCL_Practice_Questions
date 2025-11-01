import tensorflow as tf
import os
import numpy as np
import cv2
from tqdm import tqdm

INPUT_DIR = '/mnt/d/code/hcl/unet/dataset/' 
CHECKPOINT_DIR = '/mnt/d/code/hcl/unet/checkpoints/' 
BATCH_SIZE = 8
EPOCHS = 50
PATIENCE = 10

def load_image(img_path):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    
    mask_path = tf.strings.regex_replace(img_path, "Images", "Masks")
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    
    feat_path = tf.strings.regex_replace(img_path, "Images", "Heads")
    feat = tf.io.read_file(feat_path)
    feat = tf.image.decode_png(feat, channels=3)
    
    image = tf.cast(image, tf.float32)
    mask = tf.cast(mask, tf.float32)
    feat = tf.cast(feat, tf.float32)
    
    mask = tf.image.rgb_to_grayscale(mask)
    feat = tf.image.rgb_to_grayscale(feat)
    
    return image, mask, feat

def augment_images(input_img, real_img, feature):
    if tf.random.uniform(()) > 0.5:
        input_img = tf.image.flip_left_right(input_img)
        real_img = tf.image.flip_left_right(real_img)
        feature = tf.image.flip_left_right(feature)
    
    if tf.random.uniform(()) > 0.5:
        input_img = tf.image.flip_up_down(input_img)
        real_img = tf.image.flip_up_down(real_img)
        feature = tf.image.flip_up_down(feature)
    
    return input_img, real_img, feature

def preprocess_train(input_img, real_img, feature):
    input_img = tf.image.central_crop(input_img, 256/400)
    real_img = tf.image.central_crop(real_img, 256/400)
    feature = tf.image.central_crop(feature, 256/400)
    
    input_img = tf.image.resize(input_img, [256, 256])
    real_img = tf.image.resize(real_img, [256, 256])
    feature = tf.image.resize(feature, [256, 256])
    
    input_img, real_img, feature = augment_images(input_img, real_img, feature)
    
    input_img = tf.clip_by_value(input_img / 255.0, 0, 1)
    real_img = tf.clip_by_value(real_img / 255.0, 0, 1)
    feature = feature / 255.0
    real_img = tf.round(real_img)
    
    return input_img, real_img, feature

def preprocess_test(input_img, real_img, feature):
    input_img = tf.image.central_crop(input_img, 256/400)
    real_img = tf.image.central_crop(real_img, 256/400)
    feature = tf.image.central_crop(feature, 256/400)
    
    input_img = tf.image.resize(input_img, [256, 256])
    real_img = tf.image.resize(real_img, [256, 256])
    feature = tf.image.resize(feature, [256, 256])
    
    input_img = tf.clip_by_value(input_img / 255.0, 0, 1)
    real_img = tf.clip_by_value(real_img / 255.0, 0, 1)
    feature = feature / 255.0
    real_img = tf.round(real_img)
    
    return input_img, real_img, feature

def build_generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    features = tf.keras.layers.Input(shape=[256, 256, 1])
    x = tf.keras.layers.concatenate([inputs, features])
    
    # Encoder
    e1 = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same')(x)
    e1 = tf.keras.layers.LeakyReLU()(e1)
    
    e2 = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same')(e1)
    e2 = tf.keras.layers.BatchNormalization()(e2)
    e2 = tf.keras.layers.LeakyReLU()(e2)
    
    e3 = tf.keras.layers.Conv2D(256, 4, strides=2, padding='same')(e2)
    e3 = tf.keras.layers.BatchNormalization()(e3)
    e3 = tf.keras.layers.LeakyReLU()(e3)
    
    e4 = tf.keras.layers.Conv2D(512, 4, strides=2, padding='same')(e3)
    e4 = tf.keras.layers.BatchNormalization()(e4)
    e4 = tf.keras.layers.LeakyReLU()(e4)
    
    e5 = tf.keras.layers.Conv2D(512, 4, strides=2, padding='same')(e4)
    e5 = tf.keras.layers.BatchNormalization()(e5)
    e5 = tf.keras.layers.LeakyReLU()(e5)
    
    e6 = tf.keras.layers.Conv2D(512, 4, strides=2, padding='same')(e5)
    e6 = tf.keras.layers.BatchNormalization()(e6)
    e6 = tf.keras.layers.LeakyReLU()(e6)
    
    e7 = tf.keras.layers.Conv2D(512, 4, strides=2, padding='same')(e6)
    e7 = tf.keras.layers.BatchNormalization()(e7)
    e7 = tf.keras.layers.LeakyReLU()(e7)
    
    e8 = tf.keras.layers.Conv2D(512, 4, strides=2, padding='same')(e7)
    e8 = tf.keras.layers.ReLU()(e8)
    
    # Decoder
    d1 = tf.keras.layers.Conv2DTranspose(512, 4, strides=2, padding='same')(e8)
    d1 = tf.keras.layers.BatchNormalization()(d1)
    d1 = tf.keras.layers.Dropout(0.5)(d1)
    d1 = tf.keras.layers.ReLU()(d1)
    d1 = tf.keras.layers.Concatenate()([d1, e7])
    
    d2 = tf.keras.layers.Conv2DTranspose(512, 4, strides=2, padding='same')(d1)
    d2 = tf.keras.layers.BatchNormalization()(d2)
    d2 = tf.keras.layers.Dropout(0.5)(d2)
    d2 = tf.keras.layers.ReLU()(d2)
    d2 = tf.keras.layers.Concatenate()([d2, e6])
    
    d3 = tf.keras.layers.Conv2DTranspose(512, 4, strides=2, padding='same')(d2)
    d3 = tf.keras.layers.BatchNormalization()(d3)
    d3 = tf.keras.layers.Dropout(0.5)(d3)
    d3 = tf.keras.layers.ReLU()(d3)
    d3 = tf.keras.layers.Concatenate()([d3, e5])
    
    d4 = tf.keras.layers.Conv2DTranspose(512, 4, strides=2, padding='same')(d3)
    d4 = tf.keras.layers.BatchNormalization()(d4)
    d4 = tf.keras.layers.ReLU()(d4)
    d4 = tf.keras.layers.Concatenate()([d4, e4])
    
    d5 = tf.keras.layers.Conv2DTranspose(256, 4, strides=2, padding='same')(d4)
    d5 = tf.keras.layers.BatchNormalization()(d5)
    d5 = tf.keras.layers.ReLU()(d5)
    d5 = tf.keras.layers.Concatenate()([d5, e3])
    
    d6 = tf.keras.layers.Conv2DTranspose(128, 4, strides=2, padding='same')(d5)
    d6 = tf.keras.layers.BatchNormalization()(d6)
    d6 = tf.keras.layers.ReLU()(d6)
    d6 = tf.keras.layers.Concatenate()([d6, e2])
    
    d7 = tf.keras.layers.Conv2DTranspose(64, 4, strides=2, padding='same')(d6)
    d7 = tf.keras.layers.BatchNormalization()(d7)
    d7 = tf.keras.layers.ReLU()(d7)
    d7 = tf.keras.layers.Concatenate()([d7, e1])
    
    output = tf.keras.layers.Conv2DTranspose(1, 4, strides=2, padding='same', activation='tanh')(d7)
    
    return tf.keras.Model(inputs=[inputs, features], outputs=output)

def build_discriminator():
    inp = tf.keras.layers.Input(shape=[256, 256, 3])
    tar = tf.keras.layers.Input(shape=[256, 256, 1])
    feature = tf.keras.layers.Input(shape=[256, 256, 1])
    x = tf.keras.layers.concatenate([inp, tar, feature])
    
    x = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Conv2D(256, 4, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.ZeroPadding2D()(x)
    x = tf.keras.layers.Conv2D(512, 4, strides=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.ZeroPadding2D()(x)
    output = tf.keras.layers.Conv2D(1, 4, strides=1)(x)
    
    return tf.keras.Model(inputs=[inp, tar, feature], outputs=output)

@tf.function
def train_step(input_img, target, feature, generator, discriminator, 
               gen_optimizer, disc_optimizer, loss_fn, lambda_val):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator([input_img, feature], training=True)
        
        disc_real = discriminator([input_img, target, feature], training=True)
        disc_fake = discriminator([input_img, gen_output, feature], training=True)
        
        gan_loss = loss_fn(tf.ones_like(disc_fake), disc_fake)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        gen_loss = gan_loss + (lambda_val * l1_loss)
        
        real_loss = loss_fn(tf.ones_like(disc_real), disc_real)
        fake_loss = loss_fn(tf.zeros_like(disc_fake), disc_fake)
        disc_loss = real_loss + fake_loss
    
    gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
    
    return gen_loss, disc_loss, l1_loss

@tf.function
def test_step(input_img, target, feature, generator):
    gen_output = generator([input_img, feature], training=False)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    return l1_loss

def save_sample(generator, test_input, target, feature, epoch, save_dir):
    prediction = generator([test_input, feature], training=False)
    
    img = (test_input[0].numpy() * 255).astype(np.uint8)
    tar = (target[0].numpy() * 255).astype(np.uint8).squeeze()
    pred = (prediction[0].numpy() * 255).astype(np.uint8).squeeze()
    feat = (feature[0].numpy() * 255).astype(np.uint8).squeeze()
    
    tar_3d = cv2.cvtColor(tar, cv2.COLOR_GRAY2BGR)
    pred_3d = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
    feat_3d = cv2.cvtColor(feat, cv2.COLOR_GRAY2BGR)
    
    out = np.hstack((img, tar_3d, pred_3d, feat_3d))
    cv2.imwrite(f'{save_dir}/epoch_{epoch}.jpg', out)

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(f'{INPUT_DIR}/progress', exist_ok=True)
    with open(f'{INPUT_DIR}/train.txt') as f:
        train_list = [f'{INPUT_DIR}/Images/{line.strip()}' for line in f.readlines()]
    
    with open(f'{INPUT_DIR}/test.txt') as f:
        test_list = [f'{INPUT_DIR}/Images/{line.strip()}' for line in f.readlines()]
    
    print(f'Train samples: {len(train_list)}')
    print(f'Test samples: {len(test_list)}')
    
    train_ds = tf.data.Dataset.from_tensor_slices(train_list)
    train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    test_ds = tf.data.Dataset.from_tensor_slices(test_list)
    test_ds = test_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    test_sample_ds = test_ds.unbatch().batch(1).take(1)
    
    generator = build_generator()
    discriminator = build_discriminator()
    
    gen_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    disc_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        
        train_gen_losses = []
        train_disc_losses = []
        train_l1_losses = []
        
        train_bar = tqdm(train_ds, desc='Training', unit='batch')
        for input_img, target, feature in train_bar:
            g_loss, d_loss, l1_loss = train_step(input_img, target, feature, generator, 
                                                  discriminator, gen_optimizer, disc_optimizer, 
                                                  loss_fn, 100)
            train_gen_losses.append(g_loss.numpy())
            train_disc_losses.append(d_loss.numpy())
            train_l1_losses.append(l1_loss.numpy())
            
            train_bar.set_postfix({
                'G_Loss': f'{g_loss.numpy():.4f}',
                'D_Loss': f'{d_loss.numpy():.4f}',
                'L1': f'{l1_loss.numpy():.4f}'
            })
        
        test_l1_losses = []
        
        test_bar = tqdm(test_ds, desc='Testing', unit='batch')
        for input_img, target, feature in test_bar:
            test_l1 = test_step(input_img, target, feature, generator)
            test_l1_losses.append(test_l1.numpy())
            
            test_bar.set_postfix({'L1': f'{test_l1.numpy():.4f}'})
        
        avg_train_gen = np.mean(train_gen_losses)
        avg_train_disc = np.mean(train_disc_losses)
        avg_train_l1 = np.mean(train_l1_losses)
        avg_test_l1 = np.mean(test_l1_losses)
        
        print(f'\nResults:')
        print(f'  Train - Gen Loss: {avg_train_gen:.4f}, Disc Loss: {avg_train_disc:.4f}, L1: {avg_train_l1:.4f}')
        print(f'  Test  - L1: {avg_test_l1:.4f}')
        
        if avg_test_l1 < best_loss:
            best_loss = avg_test_l1
            patience_counter = 0
            generator.save(f'{CHECKPOINT_DIR}/best_generator.h5')
            discriminator.save(f'{CHECKPOINT_DIR}/best_discriminator.h5')
            print(f'Best model saved! Test L1: {best_loss:.4f}')
        else:
            patience_counter += 1
            print(f'No improvement (patience: {patience_counter}/{PATIENCE})')
        
        if patience_counter >= PATIENCE:
            print(f'\nEarly stopping triggered at epoch {epoch+1}')
            for test_input, test_target, test_feature in test_sample_ds:
                save_sample(generator, test_input, test_target, test_feature, epoch+1, f'{INPUT_DIR}/progress')
            break
        
        if patience_counter > 0 and patience_counter % 3 == 0:
            new_lr = gen_optimizer.learning_rate * 0.5
            gen_optimizer.learning_rate.assign(new_lr)
            disc_optimizer.learning_rate.assign(new_lr)
            print(f'Learning rate reduced to {new_lr.numpy():.6f}')
    
    
    print(f'Best Test L1 Loss: {best_loss:.4f}')
    
    for test_input, test_target, test_feature in test_sample_ds:
        save_sample(generator, test_input, test_target, test_feature, epoch+1, f'{INPUT_DIR}/progress')
if __name__ == "__main__":
    main()