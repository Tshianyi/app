import cv2
import numpy as np
import pywt
from scipy.stats import kurtosis, skew, entropy

def preprocess_image(image_path):
    # Lire l’image en niveaux de gris
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image introuvable : {image_path}")

    # Redimensionner uniformément
    img = cv2.resize(img, (256, 256))

    # Appliquer un flou gaussien pour réduire le bruit
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Normalisation des pixels (0–1)
    img = img / 255.0

    # Améliorer le contraste (histogramme equalization)
    img = cv2.equalizeHist((img * 255).astype(np.uint8))

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
