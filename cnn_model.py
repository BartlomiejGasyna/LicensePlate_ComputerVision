import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

dataset_path = 'ocr_dataset'

# Define image dimensions and batch size
image_height = 64
image_width = 48
batch_size = 32

# Create an instance of the ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Generate training dataset from the subfolders
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Generate validation dataset from the subfolders
validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)


# # Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
model.add(Dense(36, activation='softmax'))

# # Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


epochs = 10

# # Train the model using the training dataset
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# # Evaluate the model using the validation dataset
loss, accuracy = model.evaluate(validation_generator)
print("Validation loss:", loss)
print("Validation accuracy:", accuracy)


model.save('charModel0.h5')



# model = load_model("charModel.h5")
# # Evaluate the model using the validation dataset
# loss, accuracy = model.evaluate(validation_generator)
# print("Validation loss:", loss)
# print("Validation accuracy:", accuracy)
