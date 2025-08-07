import cv2
import numpy as np
import pywt
from scipy.stats import kurtosis, skew, entropy

def preprocess_image(image_path):
    # Charger l'image en niveaux de gris
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image introuvable : {image_path}")

    # Étape 1 : Denoising + Redimensionnement
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.resize(img, (256, 256))

    # Étape 2 : Seuil adaptatif pour simuler un scan (binarisation nette)
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Étape 3 : Amélioration du contraste (Histogramme equalization)
    img = cv2.equalizeHist(img)

    return img

def extract_features_from_image(image_path):
    img = preprocess_image(image_path)

    # Appliquer une transformée en ondelettes
    coeffs = pywt.wavedec2(img, 'haar', level=1)
    cA, (cH, cV, cD) = coeffs

    features = np.concatenate([cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()])

    # Calcul des 4 statistiques
    var = np.var(features)
    skewness = skew(features)
    kurt = kurtosis(features)
    entr = entropy(np.abs(features) + 1e-10)  # +1e-10 pour éviter log(0)

    return [var, skewness, kurt, entr]

def transform_two_images(recto_path, verso_path):
    features_recto = extract_features_from_image(recto_path)
    features_verso = extract_features_from_image(verso_path)

    # Moyenne des deux vecteurs
    features_final = np.mean([features_recto, features_verso], axis=0)
    return np.array(features_final).reshape(1, -1)
