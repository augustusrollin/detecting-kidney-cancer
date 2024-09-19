from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def build_model(input_shape=(128, 128, 3), num_classes=4):
    """
    Build a Convolutional Neural Network model.

    Args:
        input_shape (tuple): Shape of the input images (width, height, channels).
        num_classes (int): Number of classes to predict.

    Returns:
        model (Sequential): Compiled CNN model.
    """
    model = Sequential([
        # First Convolutional Layer
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Second Convolutional Layer
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Third Convolutional Layer
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten the 3D output to 1D
        Flatten(),
        
        # Fully Connected Layer
        Dense(256, activation='relu'),
        Dropout(0.5),  # Dropout for regularization
        
        # Output Layer
        Dense(num_classes, activation='softmax')  # Number of classes (e.g., 4)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
