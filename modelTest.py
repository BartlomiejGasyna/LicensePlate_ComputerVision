import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import cv2
import numpy as np
import os

# Define image dimensions and batch size
image_height = 64
image_width = 48

# Create an instance of the ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)


dataset_path = 'ocr_dataset'

# Generate training dataset from the subfolders
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(image_height, image_width),
    class_mode='categorical',
    subset='training'
)

class_names = train_generator.class_indices


model = load_model("charModel0.h5")



def inferImg(path):
    img = cv2.imread(path)

    padding = 5
    new_height = img.shape[0] + 2 * padding
    new_width = img.shape[1] + 2 * padding
    padded_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    padded_image.fill(255)  # Fill with white color (255)

    padded_image[padding:padding + img.shape[0], padding:padding + img.shape[1]] = img


    img = cv2.resize(padded_image, (image_width, image_height))
    img_cpy = img.copy()
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)

    predicted_class_index = np.argmax(pred[0])
    predicted_class = list(class_names.keys())[list(class_names.values()).index(predicted_class_index)]

    print("Predicted class:", predicted_class)

    cv2.imshow(predicted_class, img_cpy)



    return str(predicted_class)


directory = 'letter_dataset'
for filename in os.listdir(directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Construct the full file path
        file_path = os.path.join(directory, filename)

        # Read the image using OpenCV
        image = cv2.imread(file_path)

        pred = inferImg(file_path)
        # Display the image
            # Wait for a key press to move to the next image
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key == ord('q'):
            break
        
        