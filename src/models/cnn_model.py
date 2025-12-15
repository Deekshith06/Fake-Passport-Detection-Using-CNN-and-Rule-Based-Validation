"""
CNN MODEL FOR FAKE PASSPORT DETECTION
======================================
This module contains the CNN (Convolutional Neural Network) model.

Student: [Your Name]
Project: Fake Passport Detection Using CNN
Purpose: Educational project demonstrating transfer learning

WHAT IS A CNN?
Convolutional Neural Network - a type of neural network designed for images.
Instead of looking at individual pixels, CNN looks at patterns:
  - Low-level: edges, lines, corners
  - Mid-level: textures, shapes
  - High-level: objects, features

WHY USE CNN FOR PASSPORT DETECTION?
- Passports have visual security features (patterns, microtext, etc.)
- CNNs automatically learn what features matter
- Better than manually programming rules for every feature

WHAT IS TRANSFER LEARNING?
Using a model already trained on millions of images (ImageNet)
Then "fine-tuning" it for our specific task (passport detection)
Benefits: Faster training, needs less data, better accuracy
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt


# ===== MODEL CONFIGURATION =====
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3  # RGB color images
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

LEARNING_RATE = 0.0001  # How fast the model learns (small = careful learning)
EPOCHS = 20              # Number of times to go through entire dataset
BATCH_SIZE = 16          # Number of images to process at once


# ===== FUNCTION 1: CREATE CNN MODEL =====
def create_cnn_model():
    """
    Build the CNN model using transfer learning.
    
    ARCHITECTURE:
    
    1. BASE MODEL (EfficientNetB0)
       - Pre-trained on ImageNet (1.4 million images, 1000 categories)
       - Already knows general visual features
       - We freeze most layers (don't retrain them)
    
    2. CUSTOM HEAD (Our addition)
       - Global Average Pooling: Reduces dimensions
       - Dense(128): Learns passport-specific patterns
       - Dropout(0.5): Prevents overfitting by randomly turning off neurons
       - Dense(1, sigmoid): Final prediction (0-1 probability)
    
    WHY THIS ARCHITECTURE?
    - EfficientNetB0: Best balance of accuracy and speed
    - Sigmoid output: Gives probability (0=fake, 1=real)
    - Dropout: Prevents model from memorizing training data
    
    Returns:
        keras.Model: Compiled CNN model ready for training
    """
    
    print("Building CNN model...")
    print("=" * 60)
    
    # ===== STEP 1: LOAD PRE-TRAINED BASE MODEL =====
    print("\n1. Loading EfficientNetB0 (pre-trained on ImageNet)...")
    
    base_model = EfficientNetB0(
        include_top=False,      # Don't include original classification layer
        weights='imagenet',     # Use weights trained on ImageNet
        input_shape=IMG_SHAPE   # Our image size: 224x224x3
    )
    
    # ===== STEP 2: FREEZE BASE MODEL LAYERS =====
    # We don't want to retrain all layers (would take too long and need more data)
    # Only train the last few layers for passport-specific features
    
    print("2. Freezing early layers (keep general features)...")
    base_model.trainable = True  # Allow some layers to be trainable
    
    # Freeze all layers except the last 20
    # Early layers: general features (edges, textures) - keep frozen
    # Later layers: specific features - allow fine-tuning
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # Count trainable vs non-trainable layers
    trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
    frozen_count = len(base_model.layers) - trainable_count
    
    print(f"   ✓ Frozen layers: {frozen_count}")
    print(f"   ✓ Trainable layers: {trainable_count}")
    
    # ===== STEP 3: BUILD CUSTOM CLASSIFICATION HEAD =====
    print("\n3. Adding custom classification head...")
    
    # Create sequential model
    model = models.Sequential([
        
        # Input: 224x224x3 image
        layers.Input(shape=IMG_SHAPE),
        
        # Base model (EfficientNetB0)
        base_model,
        
        # Global Average Pooling
        # Converts feature maps to single vector
        # Example: (7, 7, 1280) → (1280,)
        layers.GlobalAveragePooling2D(name='global_avg_pool'),
        
        # Dense layer with 128 neurons
        # Learns passport-specific patterns
        layers.Dense(128, activation='relu', name='dense_128'),
        
        # Dropout layer
        # Randomly turns off 50% of neurons during training
        # WHY? Prevents overfitting (memorizing instead of learning)
        layers.Dropout(0.5, name='dropout_50'),
        
        # Output layer
        # Single neuron with sigmoid activation
        # Output: probability between 0 and 1
        #   Close to 0 = FAKE
        #   Close to 1 = REAL
        layers.Dense(1, activation='sigmoid', name='output')
    ])
    
    print("   ✓ Model architecture created")
    
    # ===== STEP 4: COMPILE MODEL =====
    print("\n4. Compiling model...")
    
    # LOSS FUNCTION: Binary Crossentropy
    # Used for binary classification (2 classes: real vs fake)
    # Measures how far prediction is from true label
    
    # OPTIMIZER: Adam
    # Algorithm that updates model weights during training
    # Adam is a good default choice (adaptive learning rate)
    
    # METRICS: Track accuracy during training
    # Accuracy = (correct predictions) / (total predictions)
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("   ✓ Model compiled")
    print(f"   ✓ Optimizer: Adam (lr={LEARNING_RATE})")
    print(f"   ✓ Loss: Binary Crossentropy")
    
    # ===== STEP 5: DISPLAY MODEL SUMMARY =====
    print("\n5. Model Summary:")
    print("=" * 60)
    model.summary()
    print("=" * 60)
    
    return model


# ===== FUNCTION 2: CREATE TRAINING CALLBACKS =====
def create_callbacks(model_save_path='models/passport_cnn.h5'):
    """
    Create callback functions for training.
    
    WHAT ARE CALLBACKS?
    Functions that execute during training at specific times.
    Like "alarm clocks" during training process.
    
    CALLBACKS USED:
    
    1. EARLY STOPPING
       - Monitors validation loss
       - If it doesn't improve for 5 epochs → stop training
       - WHY? Prevents wasting time and overfitting
    
    2. REDUCE LEARNING RATE ON PLATEAU
       - If validation loss stops improving → reduce learning rate
       - Helps model make smaller, more careful updates
       - Like slowing down when approaching a target
    
    3. MODEL CHECKPOINT
       - Saves best model during training
       - Only saves when validation loss improves
       - Ensures we keep the best version
    
    Args:
        model_save_path (str): Where to save the best model
        
    Returns:
        list: List of callback objects
    """
    
    # Callback 1: Early Stopping
    # Stop training if validation loss doesn't improve
    early_stopping = EarlyStopping(
        monitor='val_loss',         # Watch validation loss
        patience=5,                 # Wait 5 epochs before stopping
        restore_best_weights=True,  # Restore weights from best epoch
        verbose=1                   # Print messages
    )
    
    # Callback 2: Reduce Learning Rate
    # Make smaller updates if stuck
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',         # Watch validation loss
        factor=0.5,                 # Reduce LR by half
        patience=3,                 # Wait 3 epochs before reducing
        min_lr=1e-7,               # Don't go below this value
        verbose=1                   # Print messages
    )
    
    # Callback 3: Model Checkpoint
    # Save best model
    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        monitor='val_loss',         # Watch validation loss
        save_best_only=True,        # Only save when better than previous
        mode='min',                 # Lower validation loss is better
        verbose=1                   # Print when saving
    )
    
    return [early_stopping, reduce_lr, checkpoint]


# ===== FUNCTION 3: TRAIN MODEL =====
def train_model(model, train_generator, validation_generator, class_weights, epochs=EPOCHS):
    """
    Train the CNN model.
    
    TRAINING PROCESS:
    1. Model looks at batch of images
    2. Makes predictions
    3. Calculates how wrong it was (loss)
    4. Updates weights to do better
    5. Repeat for all batches = 1 epoch
    6. Repeat for multiple epochs
    
    WHAT ARE CLASS WEIGHTS?
    If we have more REAL passports than FAKE:
    - Model might just learn to always say "REAL"
    - Class weights make FAKE mistakes more costly
    - Forces model to learn both classes
    
    Args:
        model: CNN model to train
        train_generator: Training data generator
        validation_generator: Validation data generator
        class_weights: Dictionary of class weights {0: weight, 1: weight}
        epochs: Number of training epochs
        
    Returns:
        History: Training history (loss, accuracy per epoch)
    """
    
    print("\n" + "=" * 60)
    print("STARTING MODEL TRAINING")
    print("=" * 60)
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Class weights: {class_weights}")
    
    # Create callbacks
    callbacks = create_callbacks()
    
    # Calculate steps per epoch
    steps_per_epoch = train_generator.samples // BATCH_SIZE
    validation_steps = validation_generator.samples // BATCH_SIZE
    
    print(f"\nDataset Info:")
    print(f"  Training samples: {train_generator.samples}")
    print(f"  Validation samples: {validation_generator.samples}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    
    # Start training
    print("\n" + "=" * 60)
    print("Training in progress...")
    print("=" * 60 + "\n")
    
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        class_weight=class_weights,  # Handle imbalanced data
        callbacks=callbacks,          # Use early stopping, LR reduction, etc.
        verbose=1                     # Show progress bar
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    
    return history


# ===== FUNCTION 4: PLOT TRAINING HISTORY =====
def plot_training_history(history):
    """
    Visualize training progress.
    
    PLOTS:
    1. Training vs Validation Accuracy
       - Both should increase over time
       - Should be close to each other
    
    2. Training vs Validation Loss
       - Both should decrease over time
       - Should be close to each other
    
    WHAT TO LOOK FOR:
    
    GOOD SIGNS:
    - Both curves improve steadily
    - Validation follows training closely
    - Curves stabilize at the end
    
    BAD SIGNS (OVERFITTING):
    - Training accuracy keeps increasing
    - Validation accuracy stops or decreases
    - Large gap between training and validation
    - Solution: More data augmentation, more dropout
    
    Args:
        history: Training history object from model.fit()
    """
    
    # Extract metrics
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))
    
    # Create figure with 2 subplots
    plt.figure(figsize=(14, 5))
    
    # Plot 1: Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs_range, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs_range, val_loss, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
    print("\n✓ Training plots saved to 'models/training_history.png'")
    plt.show()


# ===== FUNCTION 5: LOAD SAVED MODEL =====
def load_model(model_path='models/passport_cnn.h5'):
    """
    Load a previously trained model.
    
    Args:
        model_path (str): Path to saved model file
        
    Returns:
        keras.Model: Loaded model ready for prediction
    """
    
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("✓ Model loaded successfully!")
    
    return model


# ===== FUNCTION 6: PREDICT SINGLE IMAGE =====
def predict_single_image(model, image_path, threshold=0.5):
    """
    Predict if a single passport image is REAL or FAKE.
    
    OUTPUT INTERPRETATION:
    - Model outputs probability between 0 and 1
    - Close to 1 = Likely REAL
    - Close to 0 = Likely FAKE
    - Around 0.5 = Uncertain
    
    Args:
        model: Trained CNN model
        image_path (str): Path to image
        threshold (float): Decision threshold (default 0.5)
        
    Returns:
        tuple: (prediction_label, confidence_score)
    """
    
    from src.data.preprocessing import load_and_preprocess_image
    
    # Load and preprocess image
    img = load_and_preprocess_image(image_path)
    
    # Add batch dimension
    # Model expects shape (batch_size, height, width, channels)
    # Our image is (height, width, channels)
    # So we add batch dimension: (1, height, width, channels)
    img = np.expand_dims(img, axis=0)
    
    # Make prediction
    prediction = model.predict(img, verbose=0)[0][0]
    
    # Convert probability to label
    if prediction >= threshold:
        label = "REAL"
        confidence = prediction
    else:
        label = "FAKE"
        confidence = 1 - prediction
    
    return label, confidence, prediction


# ===== MAIN EXECUTION (for testing) =====
if __name__ == "__main__":
    """
    Test code to verify model creation works.
    """
    
    print("=" * 60)
    print("TESTING CNN MODEL MODULE")
    print("=" * 60)
    
    # Test model creation
    model = create_cnn_model()
    
    print("\n✓ Model created successfully!")
    print(f"✓ Total parameters: {model.count_params():,}")
    
    # Count trainable parameters
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    print("\n" + "=" * 60)
    print("CNN model module is ready to use!")
    print("=" * 60)
