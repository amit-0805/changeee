import numpy as np
from tensorflow.keras.models import load_model as keras_load_model
from utils.helpers import load_single_image_pair, preprocess_single_image
import matplotlib.pyplot as plt

import numpy as np
from tensorflow.keras.models import load_model
from utils.helpers import load_single_image_pair, preprocess_single_image
import matplotlib.pyplot as plt
from cnn_model import dice_coefficient  # Import the custom metric

def load_model_from_checkpoint():
    # Load the entire model from the checkpoint
    model = load_model('models/checkpoints/best_model.h5', 
                       custom_objects={'dice_coefficient': dice_coefficient})
    print("Model loaded successfully!")
    return model

def test_model():
    # Load test images (past and present)
    back_img, fore_img, groundtruth_img = load_single_image_pair('C:/Users/amita/Desktop/change_detection/dataset/test')
    
    # Preprocess images
    input_data = preprocess_single_image(back_img, fore_img)  # Should return shape (1, 256, 256, 6)
    
    # Load the model
    model = load_model_from_checkpoint()
    
    # Predict change mask
    predicted_mask = model.predict(input_data)  # This should now work with the correct input shape
    
    # Apply threshold to create a binary mask
    thresholded_mask = (predicted_mask > 0.5).astype(np.uint8)

    # Ensure groundtruth_img is preprocessed to the right shape (for visualization)
    if groundtruth_img.ndim == 3:  # If groundtruth is a colored image
        groundtruth_img = groundtruth_img[:, :, 0]  # Use the first channel

    # Display the predicted mask and compare with ground truth
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title('Predicted Mask')
    plt.imshow(thresholded_mask[0, :, :, 0], cmap='gray')
    
    plt.subplot(1, 3, 2)
    plt.title('Ground Truth Mask')
    plt.imshow(groundtruth_img, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Back Image')
    plt.imshow(back_img)

    plt.show()

if __name__ == "__main__":
    test_model()
