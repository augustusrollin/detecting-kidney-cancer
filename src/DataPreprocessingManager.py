import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DataPreprocessingManager:
    def __init__(self, data_directory, image_size=(224, 224)):
        self.data_directory = data_directory
        self.image_size = image_size
        self.data = []
        self.labels = []

    def load_data(self):
        """
        Loads images and corresponding labels from the dataset directory.
        Assumes data is organized into subdirectories where each subdirectory is a class label.
        """
        categories = os.listdir(self.data_directory)
        for category in categories:
            category_path = os.path.join(self.data_directory, category)
            if os.path.isdir(category_path):
                for image_name in os.listdir(category_path):
                    image_path = os.path.join(category_path, image_name)
                    image = cv2.imread(image_path)
                    image = cv2.resize(image, self.image_size)
                    self.data.append(image)
                    self.labels.append(category)

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def preprocess(self):
        """
        Preprocess the data: normalizing images and encoding labels to integers.
        """
        # Normalize the image data (scale pixel values to the range [0, 1])
        self.data = self.data / 255.0
        
        # Encode labels to integers using LabelEncoder
        label_encoder = LabelEncoder()
        self.labels = label_encoder.fit_transform(self.labels)
        
        return self.data, self.labels

    def split_data(self, test_size=0.2):
        """
        Split the dataset into training and test sets.
        """
        return train_test_split(self.data, self.labels, test_size=test_size, random_state=42)
