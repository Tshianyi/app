import cv2
import numpy as np
import pywt
from scipy.stats import kurtosis, skew, entropy

def extract_features_from_image(image_path):
    # Lire l’image en niveau de gris
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Impossible de lire l'image à l’emplacement : {image_path}")

    # Redimensionner l’image
    img = cv2.resize(img, (100,100))

    # Appliquer la transformée en ondelettes (Haar)
    coeffs = pywt.wavedec2(img, 'haar', level=1)
    cA, (cH, cV, cD) = coeffs

    # Fusionner tous les coefficients en un seul vecteur
    features = np.concatenate([cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()])

    # Calcul des statistiques
    var = np.var(features)
    skewness = skew(features)
    kurt = kurtosis(features)
    entr = entropy(np.abs(features))

    return [var, skewness, kurt, entr]

def transform_two_images(recto_path, verso_path):
    # Extraire les caractéristiques de chaque image
    features_recto = extract_features_from_image(recto_path)
    features_verso = extract_features_from_image(verso_path)

    # Calcul de la moyenne des deux vecteurs
    features_final = np.mean([features_recto, features_verso], axis=0)

    return np.array(features_final).reshape(1, -1)
