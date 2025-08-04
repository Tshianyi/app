import cv2
import numpy as np
import pywt
import joblib
import os

def extract_features(recto_path, verso_path):
    # Charge les images en niveaux de gris
    img1 = cv2.imread(recto_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(verso_path, cv2.IMREAD_GRAYSCALE)

    # Combine les deux images (option simple : moyenne)
    combined = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

    # Applique la transformation par ondelettes
    coeffs = pywt.wavedec2(combined, 'db1', level=1)
    cA, (cH, cV, cD) = coeffs

    # Calcule les 4 caractéristiques
    Variance = np.var(cA)
    Skewness = np.mean((cA - np.mean(cA))**3) / (np.std(cA)**3)
    Curtosis = np.mean((cA - np.mean(cA))**4) / (np.std(cA)**4)
    Entropy = -np.sum(cA * np.log2(np.abs(cA) + 1e-10))

    return [[Variance, Skewness, Curtosis, Entropy]]

def load_model():
    path = os.path.join('app', 'model', 'banknoteMd.pkl')
    return joblib.load(path)

def predict(model, features):
    prediction = model.predict(features)
    return "Billet authentique" if prediction[0] == 0 else "Faux billet"
