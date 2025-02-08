import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# -------------------------
# Custom Loss Functions
# -------------------------
def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

# -------------------------
# Utility Functions
# -------------------------
def load_image(image_path, target_size=(500, 500)):
    """
    Loads an image from disk, decodes it, converts pixel values to [0,1],
    and resizes it to the target size.
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # scales to [0,1]
    img = tf.image.resize(img, target_size)
    return img

def generate_heatmap_overlay(image_path, model_path, output_path, blend_alpha=0.5):
    """
    Loads the trained U-Net model, processes the input satellite image,
    predicts the fire probability mask, applies contrast stretching to the
    predicted heatmap, applies a colormap, overlays it on the original image,
    and saves the final blended image as a JPEG.

    Parameters:
      image_path: Path to the input satellite image.
      model_path: Path to the saved U-Net model (e.g., 'unet_fire_model.h5').
      output_path: Path where the overlay JPEG will be saved.
      blend_alpha: The alpha blending factor (0.0 - only original image, 
                   1.0 - only heatmap).
    """
    # Load the trained model with custom objects
    custom_objects = {'combined_loss': combined_loss, 'dice_loss': dice_loss}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    # Load and preprocess the original image
    image_tensor = load_image(image_path, target_size=(500, 500))
    input_tensor = tf.expand_dims(image_tensor, axis=0)  # create batch dimension
    
    # Predict the fire probability mask (heatmap)
    prediction = model.predict(input_tensor)
    heatmap = np.squeeze(prediction)  # shape becomes (500, 500)
    
    print("Heatmap values:\n", heatmap)
    # Print the range of values to diagnose dynamic range issues
    min_val, max_val = np.min(heatmap), np.max(heatmap)
    print(f"Predicted heatmap range: min={min_val:.4f}, max={max_val:.4f}")
    
    # Contrast stretch the heatmap if needed
    if max_val - min_val > 0:
        heatmap_norm = (heatmap - min_val) / (max_val - min_val)
    else:
        heatmap_norm = heatmap  # avoid division by zero
    
    # Apply a colormap (e.g., 'jet'); this returns an RGBA image.
    colored_heatmap = cm.get_cmap('jet')(heatmap_norm)
    # Remove the alpha channel and scale to 0-255
    colored_heatmap = np.uint8(colored_heatmap[:, :, :3] * 255)
    
    # Convert the original image (Tensor) to a NumPy array (0-255 uint8)
    original_image = (image_tensor.numpy() * 255).astype(np.uint8)
    
    # Convert both to PIL images
    original_pil = Image.fromarray(original_image)
    heatmap_pil = Image.fromarray(colored_heatmap)
    
    # Blend the original image with the heatmap.
    blended = Image.blend(original_pil, heatmap_pil, alpha=blend_alpha)
    
    # Save the final overlay as a JPEG
    blended.save(output_path, format='JPEG')
    print("Overlay saved to:", output_path)

if __name__ == '__main__':
    # Set your file paths:
    image_path = 'training/output1_sat_1_burned_2.jpg'      # Path to your satellite image
    model_path = 'unet_fire_model_hi_contrast_light_augmented.h5'  # Path to your trained U-Net model
    output_path = 'heatmap_overlay.jpg'                      # Output path for the blended image
    
    generate_heatmap_overlay(image_path, model_path, output_path, blend_alpha=0.5)
