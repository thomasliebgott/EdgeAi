import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import MobileNet
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

train_data_dir = r'D:\Hochschule\SS\VigilantIA\yolov5train\datasets\person\train'
valid_data_dir = r'D:\Hochschule\SS\VigilantIA\yolov5train\datasets\person\valid'

# parameters
input_shape = (160, 160, 3)
num_classes = 2

# Data processing
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='categorical')

valid_generator = valid_datagen.flow_from_directory(
    valid_data_dir,
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='categorical')

# Create the model 
base_model = MobileNet(input_shape=(224, 224, 3), include_top=False, weights='imagenet', alpha=0.25)
# include_top  select inclusion of fully connect - False because we add our fully connected layer
# weights  pre trained model so Imagenet
# alpha  model size 

x = base_model.output
# get output model

x = GlobalAveragePooling2D()(x)
# pooling operation to reduce the spatial dimention to a fixed vector 

x = Dense(128, activation='relu')(x)
# Add out layer 128 neurones with activate fonction Relu pediction more complex 

predictions = Dense(num_classes, activation='softmax')(x)
# adding a fully connected layer to make the final predict the 2 classes only

model = Model(inputs=base_model.input, outputs=predictions)
# creation of the model

# Model training
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# chose optimiser algo - which update the weights 
# loss fonction use to calculate the model errors 
# metrics choose to evaluate the model

history = model.fit(train_generator, epochs=1, validation_data=valid_generator)

# Model evaluation
valid_generator.reset()
y_pred = model.predict(valid_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = valid_generator.classes

# Confusion matrix
confusion_mtx = confusion_matrix(y_true, y_pred_classes)
print("Confusion matrix:")
print(confusion_mtx)

# Save the TFLite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()
output_dir = r'D:\Hochschule\SS\VigilantIA\mobileNet\output\model_quantized.tflite'
with open(output_dir, 'wb') as f:
    f.write(tflite_model)

print("model saved")
