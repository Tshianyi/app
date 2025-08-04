from flask import Blueprint, render_template, request
import os
from .utils import extract_features, load_model, predict
import uuid

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def prediction():
    if 'recto' not in request.files or 'verso' not in request.files:
        return 'Veuillez uploader les deux images recto et verso.'

    recto = request.files['recto']
    verso = request.files['verso']

    recto_path = os.path.join('uploads', str(uuid.uuid4()) + '_recto.jpg')
    verso_path = os.path.join('uploads', str(uuid.uuid4()) + '_verso.jpg')
    recto.save(recto_path)
    verso.save(verso_path)

    # Extraction des features à partir des deux images
    features = extract_features(recto_path, verso_path)
    model = load_model()
    result = predict(model, features)

    return render_template('result.html', result=result)