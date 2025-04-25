import cv2
import os

def delete_if_no_face(image_path):
    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Check if no faces are detected
    if len(faces) == 0:
        # Delete the image file if no faces are detected
        os.remove(image_path)
        print(f"No face detected. Deleted: {image_path}")
    else:
        print(f"Face detected in: {image_path}")

def process_images_in_directory(directory_path):
    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Add other extensions if needed
            image_path = os.path.join(directory_path, filename)
            delete_if_no_face(image_path)

# Example usage
directory_path = r'C:\...'
process_images_in_directory(directory_path)
