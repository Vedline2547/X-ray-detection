# Importing necessary libraries
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Define image size
img_size = 100
# Load the pretrained model(VGG16)
base_model = VGG16(input_shape=(img_size,img_size,3),include_top=False,weights='imagenet')
# Freeze layers of pretrained model
for layer in base_model.layers:
    layer.trainable = False
# Add custom layers for classification
x = Flatten()(base_model.output)
x = Dense(256,activation='relu')(x)
x = Dense(128,activation='relu')(x)
x = Dense(64,activation='relu')(x)
predictions = Dense(1,activation='sigmoid')(x)
# Create a new model
model = Model(inputs=base_model.input,outputs=predictions)
# Compile model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# Define image generator for training and validation data
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
# Specify training and validation directories
train_dir = r"C:\Users\HP\Desktop\New ML\X-ray-detection\train"
test_dir = r"C:\Users\HP\Desktop\New ML\X-ray-detection\test"
# Create image generators from training and validation data
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(img_size,img_size),batch_size=32,class_mode='binary')
val_generator = val_datagen.flow_from_directory(test_dir,target_size=(img_size,img_size),batch_size=32,class_mode='binary')
# Train model
model.fit(train_generator,epochs=10,validation_data=val_generator)
# Evaluate model on test data
test_dir = r"C:\Users\HP\Desktop\New ML\X-ray-detection\val"
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir,target_size=(img_size,img_size),batch_size=32,class_mode='binary',shuffle=False)
test_loss,test_accuracy = model.evaluate(test_generator)
print("Test Accuracy: ",test_accuracy)
# # Saving pretrained model
model.save(r"pretrained_model.h5")
model = tf.keras.models.load_model(r"C:\Users\HP\Desktop\New ML\X-ray-detection\pretrained_model.h5")
model.summary()