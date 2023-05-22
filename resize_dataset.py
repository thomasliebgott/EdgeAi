import os
import cv2

input_directory = 'D:/Hochschule/SS/VigilantIA/yolov5train/datasets/coin/representative_dataset_coin'
output_directory = 'D:/Hochschule/SS/VigilantIA/yolov5train/datasets/coin/resized_images'
target_size = (416, 416)

# Créer le répertoire de sortie s'il n'existe pas
os.makedirs(output_directory, exist_ok=True)

# Parcourir toutes les images du répertoire d'entrée
for filename in os.listdir(input_directory):
    if filename.endswith('.jpg'):
        # Chemin complet de l'image d'entrée
        input_path = os.path.join(input_directory, filename)
        
        # Charger l'image
        image = cv2.imread(input_path)
        
        # Redimensionner l'image
        resized_image = cv2.resize(image, target_size)
        
        # Chemin complet de l'image de sortie
        output_path = os.path.join(output_directory, filename)
        
        # Enregistrer l'image redimensionnée
        cv2.imwrite(output_path, resized_image)
        
        print(f"Image redimensionnée et enregistrée : {output_path}")
