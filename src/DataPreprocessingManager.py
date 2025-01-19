import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

class DataPreprocessingManager:
    def __init__(self, data_directory, image_size=(224, 224), normalize_range=(0, 1)):
        self.data_directory = data_directory
        self.image_size = image_size
        self.data = []
        self.labels = []
        self.normalize_range = normalize_range
        self.allowed_extensions = {'.jpg', '.jpeg', '.png'}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def load_data(self):
        """
        Loads images and corresponding labels from the dataset directory.
        Assumes data is organized into subdirectories where each subdirectory is a class label.
        """
        logging.info("Starting data loading...")
        categories = os.listdir(self.data_directory)
        
        for category in categories:
            category_path = os.path.join(self.data_directory, category)
            
            if os.path.isdir(category_path):
                for image_name in os.listdir(category_path):
                    if not any(image_name.lower().endswith(ext) for ext in self.allowed_extensions):
                        continue  # Skip non-image files
                    
                    image_path = os.path.join(category_path, image_name)
                    try:
                        image = cv2.imread(image_path)
                        if image is None:
                            logging.warning(f"Skipping unreadable image: {image_path}")
                            continue
                        
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
                        image = cv2.resize(image, self.image_size)
                        self.data.append(image)
                        self.labels.append(category)
                    except Exception as e:
                        logging.error(f"Error loading image {image_path}: {e}")
        
        # Convert lists to NumPy arrays
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels)
        
        logging.info(f"Data loading completed. Loaded {len(self.data)} images.")

    def preprocess(self):
        """
        Preprocess the data: normalizing images and encoding labels to integers.
        """
        logging.info("Starting data preprocessing...")
        
        # Normalize the image data to the specified range
        min_val, max_val = self.normalize_range
        self.data = (self.data / 255.0) * (max_val - min_val) + min_val
        
        # Encode labels to integers using LabelEncoder
        label_encoder = LabelEncoder()
        self.labels = label_encoder.fit_transform(self.labels)
        
        logging.info("Preprocessing completed.")
        return self.data, self.labels

    def split_data(self, test_size=0.2):
        """
        Split the dataset into training and test sets.
        """
        logging.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            self.data, self.labels, test_size=test_size, random_state=42, stratify=self.labels
        )
        logging.info(f"Data split: {len(X_train)} train samples, {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test