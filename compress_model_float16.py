import tensorflow as tf

model_path='D:\Hochschule\SS\VigilantIA\yolov5train\yolo\yolov5-coin\\runs\\train\exp12\weights\\best_saved_model'
output_path='D:\Hochschule\SS\VigilantIA\yolov5train\yolo\yolov5-coin\\runs\\train\exp12\weights\\float16_model.tflite'

converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_quant_model = converter.convert()

with open(output_path, 'wb') as f:
    f.write(tflite_quant_model)
