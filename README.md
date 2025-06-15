# DeepTrack
Face Recognition Attendance System is a Python-based solution that automates attendance tracking using deep learning and facial recognition. It captures face images of students, extracts facial embeddings using the DeepFace library and Facenet, trains a neural network model for identity recognition, and logs attendance in real-time using a webcam.

---

## 📁 Project Structure
``` DeepTrack/ ├── 
├── capture_faces.py # Capture and save student face images
├── detect_faces.py # Real-time face recognition and attendance marking
├── extract_embeddings.py # Generate facial embeddings from saved images
├── train_model.py # Train a classification model on embeddings
├── data/
│ ├── images/ # Folder where captured face images are stored
│ ├── embeddings/ # Stores embeddings.npy and labels.npy
│ └── csv/
│ └── attendance.csv # Attendance log file
├── models/
│ └── face_recognition/ # Trained model and label encoder
``` 
# How It Works
1. 📸 Capture Faces
  Run to capture 100 face images of a new student:

  
  ```bash
  python capture_faces.py
  ```
  * Prompts for Student ID (e.g., STU001)
  * Saves processed grayscale images using MTCNN
  * Creates a new folder in data/images/
  2. 🔍 Extract Embeddings
    Run after collecting images:
  ```bash
  python extract_embeddings.py
  ```
  * Uses DeepFace (Facenet) to extract 128D embeddings
  * Saves embeddings.npy and labels.npy in data/embeddings/
3. 🧠 Train the Model
  * Train the neural network on embeddings:
  ```bash
  python train_model.py
  ```
  * Loads the saved embeddings and labels
  * Trains a neural network using TensorFlow/Keras
  * Saves:
    * face_recognition_model.h5 (model)
    * label_encoder.pkl (for decoding predictions)

4. 🎯 Real-Time Face Recognition
  * Start webcam-based face detection and attendance logging:

  ```bash
  python detect_faces.py
  ```
  * Captures webcam frames
  * Logs attendance to data/csv/attendance.csv
  * Uses face_recognition for locating faces
  * Generates embeddings using DeepFace
  * Predicts student ID using the trained model
  * Confidence threshold: 0.95
  * Press q to quit

## 📈 Future Improvements
  * Add liveness detection (e.g., blink detection)
  * Export attendance to Excel with timestamps
  * Web dashboard using Flask/Streamlit
  * Cloud upload (Google Sheets, Firebase, etc.)
