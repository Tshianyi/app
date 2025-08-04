import cv2
import numpy as np
import pywt
from scipy.stats import kurtosis, skew, entropy

def extract_features_from_image(image_path):
    # Lire et prétraiter l’image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100, 100))

    # Appliquer une transformée en ondelette (ex: Haar)
    coeffs = pywt.wavedec2(img, 'haar', level=1)
    cA, (cH, cV, cD) = coeffs

    features = np.concatenate([cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()])

    # Calcul des 4 statistiques demandées
    var = np.var(features)
    skewness = skew(features)
    kurt = kurtosis(features)
    entr = entropy(np.abs(features))

    return [var, skewness, kurt, entr]

def transform_two_images(recto_path, verso_path):
    # Extraire les features des deux images
    features_recto = extract_features_from_image(recto_path)
    features_verso = extract_features_from_image(verso_path)

    # Moyenne des deux vecteurs
    features_final = np.mean([features_recto, features_verso], axis=0)
    return features_final.reshape(1, -1)

