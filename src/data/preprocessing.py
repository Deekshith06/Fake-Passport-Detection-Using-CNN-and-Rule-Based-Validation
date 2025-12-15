"""
IMAGE PREPROCESSING MODULE
===========================
This module handles all image preprocessing tasks for passport images.

Student: [Your Name]
Project: Fake Passport Detection Using CNN
Purpose: Educational project demonstrating image preprocessing for ML

WHAT THIS FILE DOES:
- Loads passport images from folders
- Resizes images to standard size (224x224)
- Normalizes pixel values (0-1 range)
- Applies data augmentation (rotation, brightness, etc.)
"""

import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


# ===== CONSTANTS =====
# These are fixed values used throughout the code

IMG_HEIGHT = 224  # Standard height for EfficientNet input
IMG_WIDTH = 224   # Standard width for EfficientNet input
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
BATCH_SIZE = 16   # Number of images to process at once


# ===== FUNCTION 1: LOAD AND PREPROCESS SINGLE IMAGE =====
def load_and_preprocess_image(image_path):
    """
    Load a single image and prepare it for the CNN model.
    
    STEPS:
    1. Read image from file
    2. Convert BGR (OpenCV) to RGB (standard)
    3. Resize to 224x224 pixels
    4. Normalize pixel values from 0-255 to 0-1
    
    WHY NORMALIZE?
    Neural networks work better with small numbers (0-1) instead of large (0-255).
    This helps training converge faster and prevents numerical instability.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.array: Preprocessed image ready for CNN (224, 224, 3)
    """
    
    # Step 1: Read image using OpenCV
    img = cv2.imread(image_path)
    
    # Check if image loaded successfully
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Step 2: Convert BGR to RGB
    # OpenCV reads images in BGR format, but we need RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Step 3: Resize to standard size (224x224)
    # All images must be same size for CNN input
    img = cv2.resize(img, IMG_SIZE)
    
    # Step 4: Normalize pixel values to 0-1 range
    # Original: 0-255 (int) → After: 0.0-1.0 (float)
    img = img.astype('float32') / 255.0
    
    return img


# ===== FUNCTION 2: LOAD IMAGES FROM FOLDER =====
def load_images_from_folder(folder_path, label, max_images=None):
    """
    Load all images from a folder and assign them a label.
    
    EXAMPLE:
    folder_path = "data/train/real"
    label = 1 (for real passports)
    
    Args:
        folder_path (str): Path to folder containing images
        label (int): 0 for fake, 1 for real
        max_images (int): Maximum number of images to load (None = all)
        
    Returns:
        tuple: (images_array, labels_array)
    """
    
    images = []
    labels = []
    
    # Get list of all files in folder
    image_files = os.listdir(folder_path)
    
    # Filter only image files
    image_files = [f for f in image_files if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Limit number of images if specified
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"Loading {len(image_files)} images from {folder_path}...")
    
    # Load each image
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        
        try:
            # Load and preprocess image
            img = load_and_preprocess_image(img_path)
            images.append(img)
            labels.append(label)
            
        except Exception as e:
            print(f"Error loading {img_file}: {e}")
            continue
    
    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"Successfully loaded {len(images)} images")
    
    return images, labels


# ===== FUNCTION 3: CREATE DATA AUGMENTATION =====
def create_data_augmentation():
    """
    Create data augmentation generator for training data.
    
    WHAT IS DATA AUGMENTATION?
    Creating slightly modified versions of existing images to increase dataset size.
    Example: Original image → Rotated 5° → Brighter version → Flipped version
    
    WHY USE IT?
    1. Prevents overfitting (model memorizing training data)
    2. Makes model robust to variations (rotation, lighting, etc.)
    3. Increases effective dataset size without collecting more images
    
    AUGMENTATIONS APPLIED:
    - Small rotations (±5 degrees)
    - Width/height shifts (±10%)
    - Zoom in/out (±10%)
    - Brightness changes
    
    Returns:
        ImageDataGenerator: Keras data augmentation object
    """
    
    # Create augmentation generator
    train_datagen = ImageDataGenerator(
        rotation_range=5,           # Rotate images randomly ±5 degrees
        width_shift_range=0.1,      # Shift horizontally by 10%
        height_shift_range=0.1,     # Shift vertically by 10%
        zoom_range=0.1,             # Zoom in/out by 10%
        brightness_range=[0.8, 1.2], # Adjust brightness 80%-120%
        horizontal_flip=False,      # Don't flip horizontally (text would be backwards!)
        fill_mode='nearest'         # Fill empty pixels with nearest pixel value
    )
    
    return train_datagen


# ===== FUNCTION 4: CREATE DATA GENERATORS =====
def create_data_generators(train_dir, validation_dir):
    """
    Create data generators for loading images during training.
    
    WHAT IS A DATA GENERATOR?
    Instead of loading all images into memory at once (might crash!),
    generators load images in small batches as needed.
    
    DIRECTORY STRUCTURE EXPECTED:
    train_dir/
        ├── real/
        │   ├── passport1.jpg
        │   └── passport2.jpg
        └── fake/
            ├── fake1.jpg
            └── fake2.jpg
    
    Args:
        train_dir (str): Path to training data directory
        validation_dir (str): Path to validation data directory
        
    Returns:
        tuple: (train_generator, validation_generator)
    """
    
    # Augmentation for training data (to prevent overfitting)
    train_datagen = create_data_augmentation()
    
    # NO AUGMENTATION for validation data (we want to test on real images)
    # Only normalize pixel values
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create training data generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,        # Resize to 224x224
        batch_size=BATCH_SIZE,       # Load 16 images at a time
        class_mode='binary',         # Binary classification (real vs fake)
        shuffle=True                 # Shuffle data (important for training!)
    )
    
    # Create validation data generator
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False                # Don't shuffle validation (we want consistent results)
    )
    
    return train_generator, validation_generator


# ===== FUNCTION 5: VISUALIZE SAMPLE IMAGES =====
def visualize_samples(data_generator, num_samples=9):
    """
    Display sample images from the dataset.
    
    This helps verify that:
    1. Images are loading correctly
    2. Augmentation is working
    3. Images belong to correct classes
    
    Args:
        data_generator: Keras ImageDataGenerator
        num_samples (int): Number of images to display
    """
    
    # Get one batch of images
    images, labels = next(data_generator)
    
    # Calculate grid size (3x3 for 9 images)
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    # Create figure
    plt.figure(figsize=(12, 12))
    
    # Display each image
    for i in range(min(num_samples, len(images))):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(images[i])
        
        # Get class name
        class_name = "REAL" if labels[i] == 1 else "FAKE"
        plt.title(f"Class: {class_name}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


# ===== FUNCTION 6: CALCULATE CLASS WEIGHTS =====
def calculate_class_weights(train_generator):
    """
    Calculate class weights to handle imbalanced datasets.
    
    WHAT IS CLASS IMBALANCE?
    When you have more images of one class than another.
    Example: 300 real passports, but only 100 fake passports
    
    PROBLEM:
    Model might just learn to predict "REAL" all the time
    (it would be 75% accurate by always saying REAL!)
    
    SOLUTION:
    Give more importance (weight) to the minority class (fake).
    This forces the model to learn both classes equally.
    
    FORMULA:
    weight = total_samples / (num_classes * class_count)
    
    Args:
        train_generator: Keras data generator
        
    Returns:
        dict: Class weights {0: weight_fake, 1: weight_real}
    """
    
    # Get class distribution
    class_counts = train_generator.classes
    
    # Count samples in each class
    num_real = np.sum(class_counts == 1)
    num_fake = np.sum(class_counts == 0)
    total = len(class_counts)
    
    print(f"\nDataset Statistics:")
    print(f"  Real passports: {num_real}")
    print(f"  Fake passports: {num_fake}")
    print(f"  Total: {total}")
    
    # Calculate weights
    # More samples → lower weight, Fewer samples → higher weight
    weight_fake = total / (2 * num_fake)
    weight_real = total / (2 * num_real)
    
    class_weights = {
        0: weight_fake,  # Fake class
        1: weight_real   # Real class
    }
    
    print(f"\nCalculated Class Weights:")
    print(f"  Fake class weight: {weight_fake:.2f}")
    print(f"  Real class weight: {weight_real:.2f}")
    
    return class_weights


# ===== MAIN EXECUTION (for testing this module) =====
if __name__ == "__main__":
    """
    Test code to verify preprocessing functions work correctly.
    Run this file directly to test: python preprocessing.py
    """
    
    print("=" * 60)
    print("TESTING IMAGE PREPROCESSING MODULE")
    print("=" * 60)
    
    # Test single image loading
    print("\nTest 1: Loading single image...")
    test_image_path = "data/train/real/sample.jpg"  # Change to your image path
    
    if os.path.exists(test_image_path):
        img = load_and_preprocess_image(test_image_path)
        print(f"✓ Image shape: {img.shape}")
        print(f"✓ Pixel value range: {img.min():.2f} to {img.max():.2f}")
    else:
        print("⚠ Test image not found. Place a sample image to test.")
    
    print("\n" + "=" * 60)
    print("Preprocessing module is ready to use!")
    print("=" * 60)
