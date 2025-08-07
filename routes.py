from flask import Blueprint, render_template, request, jsonify
import os
import pathlib
from joblib import load
from ETL import transform_two_images

routes = Blueprint('routes', __name__)
base_dir = pathlib.Path(__file__).resolve().parent.parent

# Charger le modèle
model_path = os.path.join(os.path.dirname(__file__), "model", "banknoteMd.pkl")
model = load(model_path)

@routes.route("/", methods=["GET", "POST"])
def index():
    resultat = None
    if request.method == "POST":
        recto = request.files.get("recto")
        verso = request.files.get("verso")

        if recto and verso:
            upload_folder = base_dir / "app" / "static" / "uploads"
            os.makedirs(upload_folder, exist_ok=True)

            recto_path = upload_folder / recto.filename
            verso_path = upload_folder / verso.filename

            recto.save(recto_path)
            verso.save(verso_path)

            try:
    features = transform_two_images(str(recto_path), str(verso_path))
    prediction = model.predict(features)
    resultat = "✅ VRAI BILLET" if prediction[0] == 0 else "❌ FAUX BILLET"
except Exception as e:
    # Forcer le retour à VRAI BILLET même en cas d'erreur
    print(f"[ERREUR] Analyse échouée : {e}")
    resultat = "✅ VRAI BILLET (par défaut - erreur d’analyse)"

    return render_template("index.html", result=resultat)
