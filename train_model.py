import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from cnn_model import create_unet_model, save_model_architecture, dice_coefficient, dice_coefficient_loss
from utils.helpers import load_images, preprocess_images, create_data_generator

def train_model():
    # Set your dataset path correctly
    dataset_path = 'dataset'
    
    print("Loading images...")
    back_images, fore_images, groundtruth_images = load_images(os.path.join(dataset_path, 'train'))

    print("Preprocessing images...")
    X, y = preprocess_images(back_images, fore_images, groundtruth_images)

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create U-Net model
    model = create_unet_model(input_shape=X_train.shape[1:])

    # Compile the model
    optimizer = Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer,
                  loss=['binary_crossentropy', dice_coefficient_loss],
                  loss_weights=[1, 1],
                  metrics=['accuracy', dice_coefficient])

    # Callbacks
    checkpoint_path = 'models/checkpoints/best_model.h5'
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    # Data generators
    train_generator = create_data_generator(X_train, y_train, batch_size=16)
    val_generator = create_data_generator(X_val, y_val, batch_size=16)

    # Train the model
    history = model.fit(train_generator,
                        steps_per_epoch=len(X_train) // 16,
                        validation_data=val_generator,
                        validation_steps=len(X_val) // 16,
                        epochs=100,
                        callbacks=[checkpoint, early_stopping, reduce_lr])

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Save the model architecture
    save_model_architecture(model)
    model.save_weights('models/saved_model/model_weights.h5')

    print("Model training completed!")

if __name__ == "__main__":
    train_model()