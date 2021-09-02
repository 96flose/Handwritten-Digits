# Project: Handwritten letter recognition in Python
# Author: Florian S.
# Creation: 11.10.2020 (modified 02.09.2021)
# Description: A simple project relying on the 'mnist' dataset by tenserflow keras. It 
#              checks for any existing models and proceeds to creating its one if none
#              is found. It then is tested by the samples provided in the 'samples' folder


# Imports
import os
import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf

file_list = os.listdir(os.getcwd()) # lists all file in the current directory

if 'handwritten.model' not in file_list: # checks if there is already a model
    print("No model found! Creating new one.")
    
    mnist = tf.keras.datasets.mnist

    (train_image, train_label), (x_test, y_test) = mnist.load_data() # Devides the training data into the actual image and the corrosponding label

    train_image = tf.keras.utils.normalize(train_image , axis=1) # normalize simplifies the data by adjusting the values to only range from 0 to 1
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    model.add(tf.keras.layers.Dense(128, activation='relu')) # relu = rectified linear unit
    model.add(tf.keras.layers.Dense(128, activation='relu')) # relu = rectified linear unit
    model.add(tf.keras.layers.Dense(10, activation='softmax')) # all the Neurons add up to 1 (repr. confidence)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_image, train_label, epochs=5) # The model is trained with 5 iterations

    model.save('handwritten.model') 
else:
    print("Model found!")

model = tf.keras.models.load_model('handwritten.model')

image_number = 0 
while os.path.isfile(f'samples/sample{image_number}.png'): # loops through all the samples
    try:
        img = cv2.imread(f'samples/sample{image_number}.png')[:,:,0] # since we don't require colors, only the first channel is stored
        img = np.invert(np.array([img]))
        prediction = model.predict(img)

        print(f'This digit is a {np.argmax(prediction)}') # printing the models prediction
        plt.imshow(img[0], cmap=plt.cm.binary) # Showing the corresponding image
        plt.show()
    
    except:
        print("Error!")
    finally:
        image_number += 1