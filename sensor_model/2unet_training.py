import os
import glob
import tensorflow as tf
from tensorflow.keras.layers import (Input, MaxPooling2D, UpSampling2D, concatenate,
                                     ZeroPadding2D, Cropping2D, SeparableConv2D, Conv2D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# -------------------------
# 1. Define a Lightweight U-Net Model
# -------------------------
def build_unet(input_size=(500, 500, 3)):
    inputs = Input(input_size)
    
    # Encoder block 1
    c1 = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)  # 250x250
    
    # Encoder block 2
    c2 = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)  # 125x125
    
    # Encoder block 3
    c3 = SeparableConv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = SeparableConv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)  # ~62x62
    
    # Encoder block 4
    c4 = SeparableConv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = SeparableConv2D(256, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)  # ~31x31
    
    # Bottleneck
    c5 = SeparableConv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = SeparableConv2D(512, (3, 3), activation='relu', padding='same')(c5)
    
    # Decoder block 1
    u6 = UpSampling2D((2, 2))(c5)  # 31x31 -> 62x62
    u6 = concatenate([u6, c4])
    c6 = SeparableConv2D(256, (3, 3), activation='relu', padding='same')(u6)
    c6 = SeparableConv2D(256, (3, 3), activation='relu', padding='same')(c6)
    
    # Decoder block 2
    u7 = UpSampling2D((2, 2))(c6)  # 62x62 -> 124x124
    # Crop c3 from 125x125 to 124x124 for concatenation
    c3_crop = Cropping2D(cropping=((0, 1), (0, 1)))(c3)
    u7 = concatenate([u7, c3_crop])
    c7 = SeparableConv2D(128, (3, 3), activation='relu', padding='same')(u7)
    c7 = SeparableConv2D(128, (3, 3), activation='relu', padding='same')(c7)
    
    # Decoder block 3
    u8 = UpSampling2D((2, 2))(c7)  # 124x124 -> 248x248
    # Crop c2 from 250x250 to 248x248
    c2_crop = Cropping2D(cropping=((1, 1), (1, 1)))(c2)
    u8 = concatenate([u8, c2_crop])
    c8 = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(u8)
    c8 = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(c8)
    
    # Decoder block 4
    u9 = UpSampling2D((2, 2))(c8)  # 248x248 -> 496x496
    # Crop c1 from 500x500 to 496x496
    c1_crop = Cropping2D(cropping=((2, 2), (2, 2)))(c1)
    u9 = concatenate([u9, c1_crop])
    c9 = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(u9)
    c9 = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(c9)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    # Pad from 496x496 to 500x500 so output matches ground truth
    outputs = ZeroPadding2D(padding=((2, 2), (2, 2)))(outputs)
    
    model = Model(inputs, outputs)
    return model

# -------------------------------------------
# 2. Data Loading and Preprocessing with Augmentation
# -------------------------------------------
def load_image(path, mask=False):
    """Load an image from a file and resize to 500x500."""
    img = tf.io.read_file(path)
    if mask:
        img = tf.image.decode_jpeg(img, channels=1)
    else:
        img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # scales to [0,1]
    img = tf.image.resize(img, [500, 500])
    return img

def load_image_and_mask(image_path, mask_path):
    image = load_image(image_path, mask=False)
    mask = load_image(mask_path, mask=True)
    return image, mask

def augment(image, mask):
    """Apply random flips and rotations to the image and mask."""
    # Random horizontal flip
    if tf.random.uniform([]) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    # Random vertical flip
    if tf.random.uniform([]) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    # Random 90° rotations
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    mask = tf.image.rot90(mask, k)
    return image, mask

def get_dataset(image_dir='training', mask_dir='labels', batch_size=8, augment_data=True):
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.jpg')))
    
    if len(image_paths) != len(mask_paths):
        raise ValueError("The number of images and masks do not match.")
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(lambda img_p, mask_p: load_image_and_mask(img_p, mask_p),
                          num_parallel_calls=tf.data.AUTOTUNE)
    if augment_data:
        dataset = dataset.map(lambda img, mask: augment(img, mask),
                              num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# -------------------------------------------
# 3. Loss Functions: Dice Loss and Combined Loss
# -------------------------------------------
def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dl = dice_loss(y_true, y_pred)
    return bce + dl

# -------------------------------------------
# 4. Training Script Entry Point
# -------------------------------------------
def main():
    # Hyperparameters – note that with only 125 datapoints,
    # more epochs (and augmentation) can help; consider generating more data if possible.
    batch_size = 8
    epochs = 30  # Increased from 15 to 30 epochs
    learning_rate = 1e-4
    
    # Create the dataset with augmentation
    dataset = get_dataset(image_dir='training', mask_dir='labels', batch_size=batch_size, augment_data=True)
    
    # Build and compile the lightweight U-Net model
    model = build_unet(input_size=(500, 500, 3))
    model.compile(optimizer=Adam(learning_rate), loss=combined_loss, metrics=['accuracy'])
    model.summary()
    
    # Train the model
    model.fit(dataset, epochs=epochs)
    
    # Save the trained model
    model.save('unet_fire_model_hi_contrast_light_augmented.h5')
    print("Model saved as 'unet_fire_model_hi_contrast_light_augmented.h5'.")

if __name__ == "__main__":
    main()
