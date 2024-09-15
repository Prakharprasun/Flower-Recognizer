import pandas as pd
import tensorflow as tf
import os
import random
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

#Creating class ImageClassifier
class ImageClassifier:
    def __init__(self, image_url, img_size=(224, 224), test_size=0.1, random_state=2):
        self.image_url = image_url
        self.img_size = img_size
        self.test_size = test_size
        self.random_state = random_state
        self.all_image_paths = None
        self.all_image_labels = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.model = None
        self.history = None
        self.val_loss = None
        self.val_acc = None

    def download_and_prepare_data(self):
        image_path = tf.keras.utils.get_file(
            'flower_photos', self.image_url, untar=True)
        data_root = os.path.abspath(image_path)
        self.all_image_paths = list(tf.io.gfile.glob(data_root + r'/*/*'))
        label_names = sorted(
            name for name in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, name)))
        label_to_index = dict((name, index) for index, name in enumerate(label_names))

        self.all_image_labels = [
            label_to_index[os.path.basename(os.path.dirname(path))]
            for path in self.all_image_paths
        ]

        all_images = [
            cv2.imread(path)
            for path in self.all_image_paths
        ]

        #Resize all images to the specified size
        self.all_images = [cv2.resize(img, self.img_size) for img in all_images]

    def preprocess_data(self):
        #Convert to NumPy array and normalize
        self.X_train = np.array(self.all_images) / 255.0
        self.Y_train = np.array(self.all_image_labels)

        # Split into training and test data
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X_train, self.Y_train, test_size=self.test_size, random_state=self.random_state)

    def build_model(self):
        #Using EfficientNetB0 model
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        self.model = models.Sequential()
        self.model.add(base_model)
        self.model.add(layers.GlobalAveragePooling2D())
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(5, activation='softmax'))

        #Compile the model
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',  # For integer labels
                           metrics=['accuracy'])

    def train_model(self, epochs=10):
        #Fit the model
        self.history = self.model.fit(
            self.X_train, self.Y_train,
            epochs=epochs,
            validation_data=(self.X_test, self.Y_test)
        )

    def evaluate_model(self):
        #Evaluate on test data
        self.val_loss, self.val_acc = self.model.evaluate(self.X_test, self.Y_test)
        print(f"Validation Accuracy: {self.val_acc}")

    def categorize_val_acc(self):
        if self.val_acc >= 0.90:
            return "Excellent!"
        elif 0.80 <= self.val_acc < 0.90:
            return "Great!"
        elif 0.70 <= self.val_acc < 0.80:
            return "Good."
        elif 0.60 <= self.val_acc < 0.70:
            return "Fair."
        else:
            return "Needs improvement."

    def print_model_summary(self):
        #Print model architecture
        self.model.summary()

    def run(self):
        self.download_and_prepare_data()
        self.preprocess_data()
        self.build_model()
        self.train_model()
        self.evaluate_model()
        self.print_model_summary()
        print(f"Validation Accuracy appears to be {self.val_acc} , which can be considered {self.categorize_val_acc()}")

#Run the ImageClassifier
image_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
classifier = ImageClassifier(image_url)
classifier.run()