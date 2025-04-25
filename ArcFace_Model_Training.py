import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score
)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import insightface
from insightface.app import FaceAnalysis

# Initialize FaceAnalysis
app = FaceAnalysis(name='buffalo_sc', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.03)

# Load CSV Labels
csv_path = r"C:\...\image_data.csv"
labels_df = pd.read_csv(csv_path)

# Convert categorical labels to numeric
labels_df['gender'] = labels_df['Gender'].map({'Male': 1, 'Female': 0})
labels_df['age'] = labels_df['Age_Group'].map({'Young': 0, 'Old': 1})
labels_df['race'] = labels_df['Race'].map({'Asian': 0, 'Black': 1, 'White': 2, 'Indian': 3})
labels_df['file_path'] = labels_df['File_Path']

# Function to extract face embeddings
def encode_face(image_path, apply_preprocessing=False):
    if not os.path.exists(image_path):
        return None

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Apply contrast enhancement (optional)
    if apply_preprocessing:
        image_rgb = cv2.convertScaleAbs(image_rgb, alpha=1.2, beta=20)

    resized_image = cv2.resize(image_rgb, (640, 640))
    faces = app.get(resized_image)
    
    return faces[0].embedding if len(faces) > 0 else None

# Apply encoding with preprocessing enabled
labels_df['encoding'] = labels_df['file_path'].apply(lambda x: encode_face(x, apply_preprocessing=False))
labels_df = labels_df[labels_df['encoding'].notnull()]

# Define function to build model
def build_multi_output_model():
    input_layer = Input(shape=(512,), name="arcface_embedding")
    x = Dense(256, activation='relu')(input_layer)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    gender_output = Dense(1, activation='sigmoid', name='gender_output')(x)
    age_output = Dense(1, activation='sigmoid', name='age_output')(x)
    race_output = Dense(4, activation='softmax', name='race_output')(x)
    return Model(inputs=input_layer, outputs=[gender_output, age_output, race_output])

results = []

for i in range(10):
    print(f"Iteration {i+1}/10")
    train_df, test_df = train_test_split(labels_df, test_size=0.2, random_state=i)
    X_train = np.stack(train_df['encoding'].values)
    y_train_gender, y_train_age, y_train_race = train_df['gender'].values, train_df['age'].values, train_df['race'].values
    X_test = np.stack(test_df['encoding'].values)
    y_test_gender, y_test_age, y_test_race = test_df['gender'].values, test_df['age'].values, test_df['race'].values
    
    model = build_multi_output_model()
    model.compile(optimizer='adam',
                  loss={'gender_output': 'binary_crossentropy', 
                        'age_output': 'binary_crossentropy', 
                        'race_output': 'sparse_categorical_crossentropy'},
                  metrics={'gender_output': ['accuracy'], 
                        'age_output': ['accuracy'], 
                        'race_output': ['accuracy']}
    )
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(X_train, {'gender_output': y_train_gender, 'age_output': y_train_age, 'race_output': y_train_race},
              validation_split=0.1, epochs=23, batch_size=64, callbacks=[early_stopping], verbose=0)
    
    y_pred = model.predict(X_test)
    y_pred_gender, y_pred_age = (y_pred[0] > 0.5).astype(int).flatten(), (y_pred[1] > 0.5).astype(int).flatten()
    y_pred_race = np.argmax(y_pred[2], axis=1)
    
    results.append({
        'Iteration': i+1,
        'Gender_Accuracy': accuracy_score(y_test_gender, y_pred_gender),
        'Gender_F1': f1_score(y_test_gender, y_pred_gender),
        'Gender_Precision': precision_score(y_test_gender, y_pred_gender),
        'Gender_Recall': recall_score(y_test_gender, y_pred_gender),
        'Age_Accuracy': accuracy_score(y_test_age, y_pred_age),
        'Age_F1': f1_score(y_test_age, y_pred_age),
        'Age_Precision': precision_score(y_test_age, y_pred_age),
        'Age_Recall': recall_score(y_test_age, y_pred_age),
        'Race_Accuracy': accuracy_score(y_test_race, y_pred_race),
        'Race_F1': f1_score(y_test_race, y_pred_race, average='weighted'),
        'Race_Precision': precision_score(y_test_race, y_pred_race, average='weighted'),
        'Race_Recall': recall_score(y_test_race, y_pred_race, average='weighted')
    })

results_df = pd.DataFrame(results)
results_df.to_csv(r"C:\...\real_data_training_results.csv", index=False)
print("Training completed and results saved to training_results.csv")
