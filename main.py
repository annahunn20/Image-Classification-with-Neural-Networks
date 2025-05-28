import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

# Load CIFAR-10 dataset
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

# Normalize images
training_images = training_images / 255.0
testing_images = testing_images / 255.0

# Class names
class_name = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Reduce dataset size
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

# Evaluate and save
loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Save model using supported Keras format
model.save('image_classifier.keras')

# Load the model
model = models.load_model('image_classifier.keras')

# Load and preprocess image
img = cv.imread('horse.jpg')
if img is None:
    raise FileNotFoundError("Image 'horse.jpg' not found. Check the path.")

img = cv.resize(img, (32, 32))
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img)
plt.axis('off')
plt.show()

# Normalize and predict
img_array = np.array([img]) / 255.0
prediction = model.predict(img_array)
index = np.argmax(prediction)

print(f'Prediction is {class_name[index]}')
