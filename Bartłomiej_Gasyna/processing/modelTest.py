import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
import cv2
import numpy as np
# import os

def inferImg(img: np.ndarray, model):
    # Define image dimensions and batch size
    image_height = int(64*1.3)
    image_width = int(48*1.4)

    # Get a list of subfolders in the 'dir' directory
    # subfolders = sorted(os.listdir('ocr_dataset'))

    # Create an instance of LabelEncoder
    le = LabelEncoder()
    # le.fit(subfolders)

    # # save Label Encoder classes
    # np.save('classes.npy', le.classes_)

    # load Label Encoder classes
    le.classes_ = np.load('processing/classes.npy')

    
    
    
    # np.save('classes.npy', class_names)

    # model = load_model("charModel0.h5")




    padding = 10
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
    

    # print('index: ', predicted_class_index)

    predicted_class = le.inverse_transform([predicted_class_index])
    print("Predicted class:", *predicted_class)

    # cv2.imshow(predicted_class, img_cpy)

                                 

    return str(*predicted_class)

if __name__ == '__main__':
    test_img = cv2.imread('extras/5_623.jpg')
    inferImg(test_img)