import cv2
import matplotlib.pyplot as plt

# Cargar el clasificador de OpenCV para la detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def visualize(image, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    image_path = "../data/imagen1.jpg"  # Cambia esto por tu imagen de prueba
    image = cv2.imread(image_path)
    faces = detect_faces(image)
    visualize(image, faces)
