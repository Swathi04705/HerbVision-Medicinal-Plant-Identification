import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained plant classification model
#model_species = load_model('leaf.h5', compile=False)
model_species = None

# Build and load the leaf model with weights
def build_leaf_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(30, activation='softmax'))
    return model

model_segment = build_leaf_model()
model_segment.load_weights('trained_leaf_weights.h5') 

# Replace with your actual class names for the 30 classes
class_names = {
    0: "Ocimum tenuiflorum (Tulsi)",
    1: "Azadirachta indica (Neem)",
    2: "Aloe vera",
    3: "Mentha (Mint)",
    4: "Phyllanthus emblica (Amla)",
    5: "Zingiber officinale (Ginger)",
    6: "Withania somnifera (Ashwagandha)",
    7: "Tinospora cordifolia (Giloy)",
    8: "Terminalia arjuna",
    9: "Piper nigrum (Black Pepper)",
    10: "Centella asiatica (Gotu Kola)",
    11: "Cinnamomum verum (Cinnamon)",
    12: "Elettaria cardamomum (Cardamom)",
    13: "Eclipta prostrata (Bhringraj)",
    14: "Ocimum gratissimum (Ram Tulsi)",
    15: "Adhatoda vasica (Vasaka)",
    16: "Boerhavia diffusa (Punarnava)",
    17: "Alstonia scholaris (Saptaparni)",
    18: "Cassia fistula (Amaltas)",
    19: "Curcuma longa (Turmeric)",
    20: "Embelia ribes (Vidanga)",
    21: "Moringa oleifera (Moringa)",
    22: "Ricinus communis (Castor)",
    23: "Saraca asoca (Ashoka)",
    24: "Sida cordifolia (Bala)",
    25: "Solanum trilobatum",
    26: "Tephrosia purpurea (Sharpunkha)",
    27: "Tribulus terrestris (Gokshura)",
    28: "Vernonia cinerea (Sahadevi)",
    29: "Wrightia tinctoria (Indrajao)"
}

def process_image(file_path):
    img = image.load_img(file_path, target_size=(256, 256))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def predict_crop_species(img):
    prediction = model_segment.predict(img)

    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction)) * 100

    print("Predictions:", prediction)
    print("Confidence:", confidence)

    return class_names[predicted_class], round(confidence, 2)

def predict_leaf_species(img):
    preds = model_segment.predict(img)
    class_index = np.argmax(preds)
    return class_names[class_index]

def fetch_plant_description(plant_species):

    plant_info = {
        "Aloe vera": {
            "action": "Skin healing and digestive support",
            "ayurvedic_name": "Kumari",
            "dosage": "10-20 ml juice daily",
            "family": "Asphodelaceae",
            "habitat": "Dry and tropical regions",
            "siddha_tamil": "Katrazhai",
            "unani": "Musabbar"
        },

        "Ocimum tenuiflorum (Tulsi)": {
            "action": "Boosts immunity",
            "ayurvedic_name": "Tulasi",
            "dosage": "5-10 leaves daily",
            "family": "Lamiaceae",
            "habitat": "Tropical climates",
            "siddha_tamil": "Thulasi",
            "unani": "Rehan"
        },

        "Azadirachta indica (Neem)": {
            "action": "Antibacterial and antifungal",
            "ayurvedic_name": "Nimba",
            "dosage": "5-10 ml juice",
            "family": "Meliaceae",
            "habitat": "India and Southeast Asia",
            "siddha_tamil": "Veppu",
            "unani": "Neem"
        }
    }

    return plant_info.get(
        plant_species,
        {
            "action": "Information not available",
            "ayurvedic_name": "-",
            "dosage": "-",
            "family": "-",
            "habitat": "-",
            "siddha_tamil": "-",
            "unani": "-"
        }
    )

def fetch_leaf_description(leaf_species):
    return {
        'description': f"This is a placeholder description for {leaf_species}.",
        'uses': "Medicinal uses will be shown here.",
        'warnings': "Any warnings about the leaf go here."
    }

@app.route('/')
def index():
    return render_template('login_page.html')

@app.route('/signup')
def signup():
    return render_template('signup_page.html')

@app.route('/forgot')
def forgot():
    return render_template('forgot.html')

@app.route('/verification')
def verification():
    return render_template('verification.html')

@app.route('/home')
def home():
    return render_template('home_page.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/plant_species', methods=['POST'])
def prediction():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    img = process_image(file_path)

    prediction, confidence = predict_crop_species(img)

    plant_description = fetch_plant_description(prediction)

    return render_template(
        'prediction.html',
        prediction=prediction,
        confidence=confidence,
        plant_description=plant_description
    )

@app.route('/plant_segment', methods=['POST'])
def plant_segment():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    img = process_image(file_path)

    prediction, confidence = predict_crop_species(img)

    plant_description = fetch_leaf_description(prediction)

    return render_template(
        'prediction.html',
        prediction=prediction,
        confidence=confidence,
        plant_description=plant_description
    )

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001)
