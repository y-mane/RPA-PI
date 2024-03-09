from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import librosa
import os
import logging

app = Flask(__name__)

# Charger le modèle
model = pickle.load(open("C:/Users/benji/Downloads/Compressed/Speech_Emotion_Detection-master/Speech_Emotion_Detection-master/Emotion_Voice_Detection_Model.pkl", "rb"))

# Dossier pour stocker les fichiers audio téléchargés
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Émotions possibles
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

# Route principale
@app.route('/')
def index():
    return render_template('index1.html')

# Route pour la prédiction d'émotion
@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    logging.error("dqns la fonction")
    if 'file' not in request.files:
        logging.error("No file part")
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        logging.error("No selected file")
        return jsonify({'error': 'No selected file'})

    if file:
        # Enregistrez le fichier téléchargé
        logging.error("le fichier audio est bon")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        logging.error("enregistrer le chemin du fichier")
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        file.save(file_path)
        logging.error("file save")

        # Effectuez la prédiction
        logging.error("before prediction")
        emotion = predict_emotion_from_file(file_path)
        logging.error("after prediction")

        # Supprimez le fichier téléchargé après la prédiction
        os.remove(file_path)
        logging.error("chemin supprime")

        logging.error("return prediction")
        return jsonify({'emotion': emotion})

# Fonction pour prédire l'émotion à partir du fichier audio
def predict_emotion_from_file(file_path):
    try:
        
        # Chargez le fichier audio
        y, sr = librosa.load(file_path, duration=2.5)

        # Extrayez les caractéristiques MFCC
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, hop_length=1024, htk=True, n_mfcc=40), axis=1)
        
        # Remodeler les MFCC en 1D
        mfccs = mfccs.reshape(1, -1)
        logging.error(f"Shape of input data: {mfccs.shape}")
        
        # Vérifier si le nombre de caractéristiques est correct
        if mfccs.shape[1] <= 180:
            logging.error("moins de 180")
            raise ValueError(f"Expected 180 features, but got {mfccs.shape[1]} features.")
        else:
            logging.error("plus de 180")
        
        prediction_result = model.predict(mfccs)
        # Assurez-vous que le résultat de la prédiction est indexable
        if not hasattr(prediction_result, '__getitem__'):
            logging.error("Prediction result is not indexable: {prediction_result}")
            raise ValueError(f"Prediction result is not indexable: {prediction_result}")
        
        # Effectuez la prédiction avec le modèle
        emotion_id = prediction_result[0]
        emotion = observed_emotions[emotion_id]

        return emotion

    except Exception as e:
        return f'Error: {str(e)}'

if __name__ == '__main__':
    app.run(debug=True)
