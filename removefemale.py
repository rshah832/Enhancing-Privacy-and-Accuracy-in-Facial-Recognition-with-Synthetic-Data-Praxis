import os
from deepface import DeepFace

# Directory where the images are stored
image_directory = r"C:\..."

# Iterate through all files in the directory
for filename in os.listdir(image_directory):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):  # Ensure case-insensitive file extensions
        image_path = os.path.join(image_directory, filename)
        
        try:
            # Analyze the gender of the image using DeepFace
            result = DeepFace.analyze(image_path, actions=['gender'])
            print(f"Analysis result for {filename}: {result}")  # Print full analysis result for debugging
            
            dominant_gender = result[0]['dominant_gender']
            
            # Remove the image if it's female
            if dominant_gender == 'Woman':
                os.remove(image_path)
                print(f"Removed female photo: {filename}")
            else:
                print(f"Kept male photo: {filename}")
        
        except ValueError as ve:
            # This catches the error where DeepFace cannot detect a face and removes the image
            if "Face could not be detected" in str(ve):
                os.remove(image_path)
                print(f"Removed undetectable face photo: {filename}")
            else:
                print(f"Error processing {filename}: {str(ve)}")

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
