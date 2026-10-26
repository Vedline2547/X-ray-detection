# Import necessary libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import random
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
import pickle
# Training path directory
DataDir = r"C:\Users\HP\Desktop\New ML\X-ray-detection\train"
# Initialize category of classification
CATEGORIES = ["NORMAL","PNEUMONIA"]
# Converting images to GRAYSCALE
for i in CATEGORIES:
    path = os.path.join(DataDir,i)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array,cmap='gray')
        plt.show()
        break
#  Initializing size of image
img_size = 100
# Image after resize and gray scale
new_array = cv2.resize(img_array,(img_size,img_size))
plt.imshow(new_array,cmap='gray')
plt.show()
# Creating training data
training_data = []
def create_training_data():
    for i in CATEGORIES:
        path = os.path.join(DataDir,i)
        class_num = CATEGORIES.index(i)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(img_size,img_size))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
create_training_data()
print(len(training_data))
# Shuffling training data to ensure randomness
random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample)
X = []
y = []
# Splitting the features and labels
for features,labels in training_data:
    X.append(features)
    y.append(labels)
# print(X[0].reshape(-1,img_size,img_size,1))
y = np.array(y)
# Reshaping feature to make it compatible with tensorflow
X = np.array(X).reshape(-1,img_size,img_size,1)
# validatation data
validation_data = []
DataDir_val = r"C:\Users\HP\Desktop\New ML\X-ray-detection\test"
def create_validation_data():
    for i in CATEGORIES:
            path = os.path.join(DataDir_val,i)
            class_num = CATEGORIES.index(i)
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array,(img_size,img_size))
                    validation_data.append([new_array,class_num])
                except Exception as e:
                    pass
create_validation_data()
print(len(validation_data))
# Shuffling validation data to ensure randomness
random.shuffle(validation_data)
for sample in validation_data[:10]:
    print(sample)
X_val = []
y_val = []
# Splitting the features and labels
for features,labels in validation_data:
    X_val.append(features)
    y_val.append(labels)
y_val = np.array(y_val)
# Reshaping feature to make it compatible with tensorflow
X_val = np.array(X_val).reshape(-1,img_size,img_size,1)
# Training model
X = X/255.0
X_val = X_val/255.0
print(X.shape)
print(X_val.shape)
# Initialize neural network layer for training the model
model = Sequential()
model.add(Conv2D(64,(3,3),input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.2))

model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.2))

model.add(Conv2D(256,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.2))

model.add(Flatten())
model.add(Dense(64))

model.add(Dropout(.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# # Fitting model with 10 epochs
model.fit(X,y,batch_size=4,epochs=10,validation_data=(X_val,y_val))
# Saving model
model.save(r"model_10.h5")
# Function for prweparing image for predicting the class
def prepare(image):
    img_size=100
    img = tf.keras.preprocessing.image.load_img(image,color_mode='grayscale',target_size=(img_size,img_size))
    new_array = tf.keras.preprocessing.image.img_to_array(img)
    return new_array.reshape(-1,img_size,img_size,1)
model = tf.keras.models.load_model(r"C:\Users\HP\Desktop\New ML\X-ray-detection\model_10.h5")
image = r"C:\Users\HP\Desktop\New ML\X-ray-detection\val\PNEUMONIA\person1946_bacteria_4875.jpeg"
prediction = model.predict([prepare(image)/255.0])
print(prediction)
print(round(prediction[0][0]))
print(CATEGORIES[int(round(prediction[0][0]))])
img = mpimg.imread(image)
imgplot = plt.imshow(img)
plt.title(CATEGORIES[int(prediction[0][0])])
plt.show()