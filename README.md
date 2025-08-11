# 🎭 Real-Time Emotion Detection Using AI

This project detects human emotions **in real-time** using your webcam.  
It uses **OpenCV** to detect faces, then a **pre-trained deep learning model** (`Emotion_Detection.h5`) to classify the facial expression into one of the following emotions:

- Angry 😡
- Happy 😄
- Neutral 😐
- Sad 😢
- Surprise 😲

---

## 📸 How It Works

1. **Face Detection**  
   - Uses **Haar Cascade Classifier** (`haarcascade_frontalface_default.xml`) to detect faces in the video stream.
   - Haar Cascade is a traditional computer vision method trained to quickly find objects (faces in this case) in an image.

2. **Preprocessing the Face**  
   - The detected face is **cropped** from the frame.
   - Converted to **grayscale** (1 channel instead of 3).
   - Resized to **48×48 pixels** (the size expected by the model).
   - Pixel values normalized to `[0,1]` for better model performance.

3. **Emotion Classification**  
   - The preprocessed face is passed to the **Keras/TensorFlow** model.
   - The model outputs probabilities for each emotion category.
   - The emotion with the highest probability is chosen as the prediction.

4. **Displaying Results**  
   - The predicted emotion label is drawn above the face in the live video feed.
   - Press **`q`** to exit.

---

## 🛠 Technologies & Libraries Used

- **Python 3.10**
- **OpenCV** → Face detection, video capture, image processing
- **TensorFlow / Keras** → Model loading and inference
- **NumPy** → Array operations
- **Haar Cascade** → Pretrained XML file for detecting faces
- **Pretrained Model** (`Emotion_Detection.h5`) → CNN trained on FER2013 dataset

---

## 📊 How the Model Was Trained (Summary)

- **Dataset:** FER2013 (Facial Expression Recognition 2013)  
- **Model Type:** Convolutional Neural Network (CNN)  
- **Input Shape:** 48×48 grayscale images  
- **Output Classes:** Angry, Happy, Neutral, Sad, Surprise  
- **Loss Function:** Categorical Crossentropy  
- **Optimizer:** Adam  
- **Training:** The model learns to identify patterns in facial features (eyes, eyebrows, mouth shape) associated with each emotion.


---

## 🛠️ Technologies Used
- Python 3.x
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib

---

## 📥 Download Pre-trained Model
Download the pre-trained model file (`Emotion_Detection.h5`) from Google Drive:  
[📌 Click here to download](https://drive.google.com/uc?export=download&id=1_sy6WGNcqmBOVKH2HOd7-Rd8_DwPB9RA)  

Place it inside the `Emotion-Detection/` directory.

---

## ⚙️ Installation & Setup


### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/emotion-detection-opencv-tf.git
cd emotion-detection-opencv-tf

python -m venv .venv

.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt




