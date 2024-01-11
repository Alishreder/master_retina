import os
import cv2
import numpy as np
import pywt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def apply_wavelet_transform(image, wavelet='db1', size=(224, 224)):
    image = cv2.resize(image, size)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coeffs = pywt.dwt2(image, wavelet)
    LL, (LH, HL, HH) = coeffs
    combined = np.vstack((np.hstack((LL, LH)), np.hstack((HL, HH))))
    combined_normalized = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX)
    return combined_normalized.astype(np.uint8)


def load_and_process_images(folder_path, wavelet='db1', size=(224, 224)):
    processed_images = []
    for i in range(20):
        file_name = f"preprocessed_image_{i}.png"
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            image = cv2.imread(file_path)
            processed_image = apply_wavelet_transform(image, wavelet, size)
            processed_images.append(processed_image)
        else:
            print(f"File not found: {file_path}")
    return processed_images


images_folder = 'res'
processed_images = load_and_process_images(images_folder)

if not processed_images:
    print("No images were processed. Exiting the script.")
    exit()

labels = np.array([0 if i < 10 else 1 for i in range(20)], dtype=np.int64)

processed_images = np.array(processed_images).reshape(-1, 224, 224, 1) / 255.0

X_train, X_test, y_train, y_test = train_test_split(processed_images, labels, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(datagen.flow(X_train, y_train, batch_size=32),
          epochs=10,
          validation_data=(X_test, y_test))

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")
