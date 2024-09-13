import cv2

from face_detection import detect_faces
from face_detection import visualize as visualize_faces
from object_detection import detect_objects
from object_detection import visualize as visualize_objects


def process_image(image_path):
    image = cv2.imread(image_path)

    # Detección de objetos
    detections = detect_objects(image)
    print("Detectando objetos...")
    visualize_objects(image.copy(), detections)

    # Detección de rostros
    faces = detect_faces(image)
    print("Detectando rostros...")
    visualize_faces(image.copy(), faces)

if __name__ == "__main__":
    image_path = "../data/imagen1.jpg"  # Cambia esto por tu imagen o video
    process_image(image_path)
