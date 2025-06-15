import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import joblib

EMBEDDING_DIR = "data/embeddings/"
MODEL_DIR = "models/face_recognition/"
os.makedirs(MODEL_DIR, exist_ok=True)

embeddings = np.load(os.path.join(EMBEDDING_DIR, "embeddings.npy"))
labels = np.load(os.path.join(EMBEDDING_DIR, "labels.npy"))

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

model = keras.Sequential([
    keras.layers.Dense(724, activation="relu", input_shape=(embeddings.shape[1],)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),

    keras.layers.Dense(428, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(100, activation="relu"),
    keras.layers.BatchNormalization(),

    keras.layers.Dense(num_classes, activation="softmax") 
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

print("Training Neural Network face recognition model...")
model.fit(embeddings, encoded_labels, epochs=15, batch_size=16, validation_split=0.2)

model.save(os.path.join(MODEL_DIR, "face_recognition_model.h5"))
print("Face recognition model trained and saved successfully!")