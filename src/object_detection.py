import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Cargar el modelo preentrenado SSD MobileNet
model = tf.saved_model.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

def detect_objects(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis,...]
    detections = model(input_tensor)
    return detections

def visualize(image, detections):
    h, w, _ = image.shape
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(int)

    for i in range(len(boxes)):
        if scores[i] > 0.5:  # Mostrar solo objetos con confianza > 50%
            box = boxes[i]
            y_min, x_min, y_max, x_max = box
            start_point = (int(x_min * w), int(y_min * h))
            end_point = (int(x_max * w), int(y_max * h))
            cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    image_path = "../data/imagen1.jpg"  # Asegúrate de colocar la ruta correcta de la imagen
    image = cv2.imread(image_path)
    detections = detect_objects(image)
    visualize(image, detections)
