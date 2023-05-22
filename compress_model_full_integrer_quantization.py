import tensorflow as tf
import numpy as np
import glob

def preprocess_image(image):
    image = tf.image.resize(image, (416, 416))
    image = image / 255.0
    return image

image_directory = 'D:/Hochschule/SS/VigilantIA/yolov5train/datasets/coin/resized_images'
image_paths = glob.glob(image_directory + '/*.jpg')
representative_images = []

for image_path in image_paths:
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = preprocess_image(image)
    representative_images.append(image)

def representative_dataset_gen():
    for image in representative_images:
        yield [image]

saved_model_path = 'D:/Hochschule/SS/VigilantIA/yolov5train/yolo/yolov5-coin/runs/train/exp13/weights/best_saved_model'
quantized_model_path = 'D:/Hochschule/SS/VigilantIA/yolov5train/yolo/yolov5-coin/runs/train/exp13/weights/integrer_quantized_model.tflite'

model = tf.saved_model.load(saved_model_path)

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
converter.experimental_new_converter = True  # Contourne l'opération de rembourrage
tflite_quant_model = converter.convert()

with open(quantized_model_path, 'wb') as f:
    f.write(tflite_quant_model)

print('Quantification en nombres entiers terminée. Modèle enregistré sous :', quantized_model_path)
