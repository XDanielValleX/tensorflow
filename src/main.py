import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# Cargar el modelo preentrenado SSD MobileNet para detección de objetos
def load_object_detection_model():
    model = tf.saved_model.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
    return model

# Función para detección de objetos
def detect_objects(model, image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(image_rgb)
    input_tensor = input_tensor[tf.newaxis,...]
    
    # Realizar la detección de objetos
    detections = model(input_tensor)
    
    return detections

# Visualización de detección de objetos
def visualize_objects(image, detections):
    h, w, _ = image.shape
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(int)

    for i in range(len(boxes)):
        if scores[i] > 0.5:  # Mostrar solo objetos con una confianza mayor al 50%
            box = boxes[i]
            y_min, x_min, y_max, x_max = box
            start_point = (int(x_min * w), int(y_min * h))
            end_point = (int(x_max * w), int(y_max * h))
            cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
            label = f"Objeto {classes[i]}, Confianza: {scores[i]:.2f}"
            cv2.putText(image, label, (start_point[0], start_point[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return image

# Cargar el clasificador de reconocimiento facial de OpenCV
def load_face_detection_model():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

# Función para detección de rostros
def detect_faces(face_cascade, image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Visualización de detección de rostros
def visualize_faces(image, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return image

# Simulación del movimiento del dron en un espacio 2D
def simulate_drone_movement(points, object_model, face_cascade):
    for point in points:
        print(f"Moviendo el dron a la posición {point}")
        
        # Aquí, en un dron real, se controlarían los motores para mover el dron. En esta simulación, esperamos 1 segundo por cada punto.
        time.sleep(1)
        
        # Simular la captura de una imagen en el punto actual (usamos una imagen de prueba)
        image = cv2.imread("ruta/a/una/imagen_de_prueba.jpg")

        # Detección de objetos y rostros en la imagen
        detections = detect_objects(object_model, image)
        faces = detect_faces(face_cascade, image)

        # Visualizar las detecciones en la imagen simulada
        image_with_objects = visualize_objects(image.copy(), detections)
        image_with_faces = visualize_faces(image_with_objects.copy(), faces)

        # Mostrar la imagen con objetos y rostros detectados
        cv2.imshow(f"Vista del Dron en {point}", image_with_faces)
        cv2.waitKey(1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Cargar los modelos de detección
    object_model = load_object_detection_model()
    face_cascade = load_face_detection_model()

    # Definir los puntos de interés que el dron debe recorrer (coordenadas simuladas en 2D)
    points_of_interest = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]

    # Simular el movimiento del dron a través de esos puntos
    simulate_drone_movement(points_of_interest, object_model, face_cascade)
