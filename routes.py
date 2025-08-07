from flask import Blueprint, render_template, request, jsonify
import os
import pathlib
from joblib import load
from ETL import transform_two_images

routes = Blueprint('routes', __name__)

# Charger le modèle
model_path =  os.path.join(os.path.dirname(__file__),"model" ,"banknoteMd.pkl")
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
                resultat = f"Erreur lors de l’analyse : {e}"

    return render_template("index.html", result=resultat)

# ✅ Nouvelle route API pour interagir avec le frontend JS
@routes.route("/api/verify", methods=["POST"])
def api_verify():
    recto = request.files.get("recto")
    verso = request.files.get("verso")
    amount = request.form.get("amount")  # Utilisé si tu veux enregistrer plus tard

    if not recto or not verso:
        return jsonify({"authentique": False, "message": "Images manquantes"}), 400

    try:
        upload_folder = base_dir / "app" / "static" / "uploads"
        os.makedirs(upload_folder, exist_ok=True)

        recto_path = upload_folder / recto.filename
        verso_path = upload_folder / verso.filename

        recto.save(recto_path)
        verso.save(verso_path)

        features = transform_two_images(str(recto_path), str(verso_path))
        prediction = model.predict(features)

        is_auth = prediction[0] == 0
        return jsonify({
            "authentique": is_auth,
            "message": " FAUX BILLET" if is_auth else "VRAI BILLET"
        })

    except Exception as e:
        return jsonify({
            "authentique": False,
            "message": f"Erreur lors de l’analyse : {str(e)}"
        }), 500
