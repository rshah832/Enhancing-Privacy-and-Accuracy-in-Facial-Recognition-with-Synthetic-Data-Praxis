import os
import insightface
from PIL import Image
import numpy as np

# Initialize the InsightFace model for age and gender estimation
def load_gender_classifier():
    # Load the pre-trained model for gender and age estimation
    model = insightface.app.FaceAnalysis()
    model.prepare(ctx_id=-1, det_size=(224, 224))  # Use CPU with det_size appropriate for your images
    return model

def delete_female_images(directory):
    # Initialize the model
    model = load_gender_classifier()
    
    # Iterate over all images in the directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                try:
                    # Load the image and convert it to a format InsightFace can analyze
                    image = Image.open(file_path).convert('RGB')
                    image_np = np.array(image)
                    
                    # Run gender prediction
                    faces = model.get(image_np)
                    if faces:
                        gender = faces[0].gender  # Get the gender of the first detected face

                        # Gender: 0 is Female, 1 is Male in InsightFace models
                        if gender == 0:
                            os.remove(file_path)
                            print(f"Deleted Female image: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    target_directory = r"C:\..."  # Replace with your directory path
    delete_female_images(target_directory)
