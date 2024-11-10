import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize  # Import resize for resizing images
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_images(dataset_path):
    back_images = []
    fore_images = []
    groundtruth_images = []

    # Define paths for back, fore, and groundtruth images directly
    back_path = os.path.join(dataset_path, 'back')
    fore_path = os.path.join(dataset_path, 'fore')
    groundtruth_path = os.path.join(dataset_path, 'groundtruth')

    # Check if the directories exist
    if not os.path.exists(back_path) or not os.path.exists(fore_path) or not os.path.exists(groundtruth_path):
        raise FileNotFoundError(f"One or more directories do not exist: {back_path}, {fore_path}, {groundtruth_path}")

    # Load images
    for img_name in os.listdir(back_path):
        try:
            back_img = imread(os.path.join(back_path, img_name))
            fore_img = imread(os.path.join(fore_path, img_name))
            groundtruth_img = imread(os.path.join(groundtruth_path, img_name))

            # Resize images to (256, 256)
            back_img = resize(back_img, (256, 256), anti_aliasing=True)
            fore_img = resize(fore_img, (256, 256), anti_aliasing=True)
            groundtruth_img = resize(groundtruth_img, (256, 256), anti_aliasing=True)

            back_images.append(back_img)
            fore_images.append(fore_img)
            groundtruth_images.append(groundtruth_img)
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")

    return np.array(back_images), np.array(fore_images), np.array(groundtruth_images)

def preprocess_images(back_images, fore_images, groundtruth_images):
    # Check if all inputs have the same number of samples
    num_back = back_images.shape[0]
    num_fore = fore_images.shape[0]
    num_groundtruth = groundtruth_images.shape[0]
    
    # Raise an error if the number of samples is inconsistent
    if not (num_back == num_fore == num_groundtruth):
        raise ValueError(f"Mismatch in number of images: back ({num_back}), fore ({num_fore}), groundtruth ({num_groundtruth})")

    # Ensure all inputs are float32 and normalized to [0, 1]
    back_images = back_images.astype(np.float32) / 255.0
    fore_images = fore_images.astype(np.float32) / 255.0
    groundtruth_images = groundtruth_images.astype(np.float32) / 255.0

    # Ensure images have shape (height, width, channels)
    back_images = back_images.reshape(-1, 256, 256, 3)  # Adjust for RGB channels (256x256)
    fore_images = fore_images.reshape(-1, 256, 256, 3)  # Adjust for RGB channels (256x256)
    groundtruth_images = groundtruth_images.reshape(-1, 256, 256, 1)  # Single channel for ground truth (256x256)

    # Concatenate background and foreground images
    X = np.concatenate((back_images, fore_images), axis=-1)
    y = groundtruth_images  # Ground truth is already normalized

    print(f"Number of samples after preprocessing: {X.shape[0]}, {y.shape[0]}")
    print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")
    
    return X, y

def create_data_generator(X, y, batch_size=32):
    data_gen_args = dict(rotation_range=10,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.1,
                         horizontal_flip=True,
                         fill_mode='nearest')
    
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    seed = 1
    image_generator = image_datagen.flow(X, batch_size=batch_size, seed=seed)
    mask_generator = mask_datagen.flow(y, batch_size=batch_size, seed=seed)
    
    return zip(image_generator, mask_generator)

def load_single_image_pair(test_folder):
    back_img = imread(os.path.join(test_folder, 'back', '00480.png'))
    fore_img = imread(os.path.join(test_folder, 'fore', '00480.png'))
    groundtruth_img = imread(os.path.join(test_folder, 'groundtruth', '00480.png'))

    # Resize images to (256, 256)
    back_img = resize(back_img, (256, 256), anti_aliasing=True)
    fore_img = resize(fore_img, (256, 256), anti_aliasing=True)
    groundtruth_img = resize(groundtruth_img, (256, 256), anti_aliasing=True)

    return back_img, fore_img, groundtruth_img

def preprocess_single_image(back_img, fore_img):
    # Check if images are of shape (256, 256, 3)
    if back_img.shape != (256, 256, 3) or fore_img.shape != (256, 256, 3):
        raise ValueError("Both back and fore images must be of shape (256, 256, 3).")

    # Convert to float and normalize (0-1)
    back_img = back_img.astype(np.float32) / 255.0
    fore_img = fore_img.astype(np.float32) / 255.0
    
    # Concatenate images along the channel axis
    input_data = np.concatenate((back_img, fore_img), axis=-1)  # Shape: (256, 256, 6)
    
    # Add batch dimension
    input_data = np.expand_dims(input_data, axis=0)  # Shape: (1, 256, 256, 6)
    
    return input_data
