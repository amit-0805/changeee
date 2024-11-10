import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model

# Load your model
model = load_model(r"C:\Users\amita\Desktop\change_detection\models\checkpoints\best_model.h5")  # Update with your actual model path

def preprocess_single_image(background_image, foreground_image):
    # Resize images to the model input size
    background_image = cv2.resize(background_image, (256, 256))
    foreground_image = cv2.resize(foreground_image, (256, 256))
    
    # Concatenate the images along the last axis
    combined_image = np.concatenate((background_image, foreground_image), axis=-1)
    
    # Normalize the combined image
    combined_image = combined_image.astype(np.float32) / 255.0
    
    return np.expand_dims(combined_image, axis=0)  # Add batch dimension

def predict_changes(background_image_path, foreground_image_path):
    background_image = cv2.imread(background_image_path)
    foreground_image = cv2.imread(foreground_image_path)
    
    # Check if images are loaded
    if background_image is None or foreground_image is None:
        print("Error: Could not load one or both images.")
        return

    preprocessed_image = preprocess_single_image(background_image, foreground_image)

    # Predict the mask
    predicted_mask = model.predict(preprocessed_image)

    # Check the shape of the predicted mask
    print(f"Predicted mask shape: {predicted_mask.shape}")

    # Apply a threshold to create a binary mask
    binary_mask = (predicted_mask[0, :, :, 0] > 0.5).astype(np.uint8)  # Thresholding

    # Optionally apply morphological operations to clean the mask
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Display results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB))
    plt.title("Background Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(foreground_image, cv2.COLOR_BGR2RGB))
    plt.title("Foreground Image")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(binary_mask, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis('off')

    plt.show()

# Example usage with the given path
background_image_path = "C:\\Users\\amita\\Desktop\\change_detection\\dataset\\test\\back\\00480.png"
foreground_image_path = "C:\\Users\\amita\\Desktop\\change_detection\\dataset\\test\\fore\\00480.png"  # Update with your actual foreground image path

predict_changes(background_image_path, foreground_image_path)
