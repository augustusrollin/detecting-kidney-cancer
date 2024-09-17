import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from PIL import Image
import numpy as np

def load_data(csv_path, img_dir, target_size=(128, 128)):
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Initialize lists to store images and labels
    images = []
    labels = []

    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        # Construct the full image path
        img_path = os.path.join(img_dir, row['path'])
        
        # Load the image
        img = Image.open(img_path).resize(target_size)
        img_array = np.array(img)
        
        # Append the image and label to the respective lists
        images.append(img_array)
        labels.append(row['target'])  # Use the 'target' column as labels

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = to_categorical(np.array(labels))  # One-hot encode labels

    return images, labels

def preprocess_data(images, labels, test_size=0.2):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=42)
    
    # Initialize data generators for augmentation
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create data generators
    train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
    test_generator = test_datagen.flow(X_test, y_test, batch_size=32)

    return train_generator, test_generator

