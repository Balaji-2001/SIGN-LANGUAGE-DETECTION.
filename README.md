# SIGN-LANGUAGE-DEECTION.

A real-time gesture recognition system for American Sign Language (ASL) alphabet detection using deep learning and computer vision.

## 🚀 Project Overview

This project leverages a Convolutional Neural Network (CNN) trained on the [Sign Language MNIST dataset](https://www.kaggle.com/datamunge/sign-language-mnist) to recognize ASL hand signs (A-Y, excluding J and Z) from webcam input. It combines TensorFlow/Keras for deep learning, OpenCV for real-time video capture, and MediaPipe for hand landmark detection, enabling robust and accurate sign recognition.

## 🎯 Features

- *High Accuracy:* Achieves ~90% accuracy on test data.
- *Real-Time Detection:* Uses webcam input for live sign language recognition.
- *Hand ROI Extraction:* Utilizes MediaPipe for accurate hand localization.
- *Top-3 Prediction Display:* Shows the most probable ASL letters with confidence scores.
- *User-Friendly:* Simple interface—press SPACE to predict, ESC to exit.

## 📂 Dataset

- *Source:* [Sign Language MNIST (Kaggle)](https://www.kaggle.com/datamunge/sign-language-mnist)
- *Description:* 28x28 grayscale images of hand signs for the ASL alphabet (24 classes, A-Y excluding J, Z).

## 🛠 Installation

1. *Clone the repository:*
   bash
   git clone https://github.com/Balaji-2001/sign-language-detection.git
   cd sign-language-detection
   

2. *Install dependencies:*
   bash
   pip install numpy pandas matplotlib seaborn opencv-python mediapipe tensorflow keras
   

3. *Download the dataset:*
   - Place sign_mnist_train.csv and sign_mnist_test.csv in the project directory.

## 🏃 Usage

### 1. Train the Model

bash
python model.py

- Trains the CNN and saves the model as smnist.h5.

### 2. Real-Time Detection

bash
python camera.py

- Opens your webcam for real-time sign recognition.
- *Controls:*  
  - Press SPACE to predict the current hand sign  
  - Press ESC to exit

## 🧠 Model Architecture

- 3 Convolutional layers with Batch Normalization and MaxPooling
- Dropout layers to prevent overfitting
- Dense layers for classification (Softmax activation)
- Data augmentation for improved generalization

## 📊 Results

- *Test Accuracy:* ~90%
- *Real-time Prediction:* Top-3 most likely ASL letters with confidence scores

## 🖥 Technologies Used

- Python
- TensorFlow & Keras
- OpenCV
- MediaPipe
- Pandas, NumPy, Matplotlib, Seaborn

## 🙏 Credits

- Dataset: [Sign Language MNIST (Kaggle)](https://www.kaggle.com/datamunge/sign-language-mnist)
- Hand landmark detection: [MediaPipe](https://mediapipe.dev/)
- Developed by [Balaji V](https://github.com/Balaji-2001)

*Feel free to fork, contribute, or use this project for learning and research!*

Let me know if you want to include a sample output image, GIF, or further customization for your GitHub!
