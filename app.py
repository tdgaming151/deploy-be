# Danh sách các disease/triệu chứng bạn đưa
disease_list = [
    "anxiety and nervousness",
    "depression",
    "shortness of breath",
    "depressive or psychotic symptoms",
    "sharp chest pain",
    "dizziness",
    "insomnia",
    "abnormal involuntary movements",
    "chest tightness",
    "palpitations",
    "irregular heartbeat",
    "breathing fast",
    "hoarse voice",
    "sore throat",
    "difficulty speaking",
    "cough",
    "nasal congestion",
    "throat swelling",
    "diminished hearing",
    "lump in throat",
    "throat feels tight",
    "difficulty in swallowing",
    "skin swelling",
    "retention of urine",
    "groin mass",
    "leg pain",
    "hip pain",
    "suprapubic pain",
    "blood in stool",
    "lack of growth",
    "emotional symptoms",
    "elbow weakness",
    "back weakness",
    "pus in sputum",
    "symptoms of the scrotum and testes",
    "swelling of scrotum",
    "pain in testicles",
    "flatulence",
    "pus draining from ear",
    "jaundice",
    "mass in scrotum",
    "white discharge from eye",
    "irritable infant",
    "abusing alcohol",
    "fainting",
    "hostile behavior",
    "drug abuse",
    "sharp abdominal pain",
    "feeling ill",
    "vomiting",
    "headache",
    "nausea",
    "diarrhea",
    "vaginal itching",
    "vaginal dryness",
    "painful urination",
    "involuntary urination",
    "pain during intercourse",
    "frequent urination",
    "lower abdominal pain",
    "vaginal discharge",
    "blood in urine",
    "hot flashes",
    "intermenstrual bleeding",
    "hand or finger pain",
    "wrist pain",
    "hand or finger swelling",
    "arm pain",
    "wrist swelling",
    "arm stiffness or tightness",
    "arm swelling",
    "hand or finger stiffness or tightness",
    "wrist stiffness or tightness",
    "lip swelling",
    "toothache",
    "abnormal appearing skin",
    "skin lesion",
    "acne or pimples",
    "dry lips",
    "facial pain",
    "mouth ulcer",
    "skin growth",
    "eye deviation",
    "diminished vision",
    "double vision",
    "cross-eyed",
    "symptoms of eye",
    "pain in eye",
    "eye moves abnormally",
    "abnormal movement of eyelid",
    "foreign body sensation in eye",
    "irregular appearing scalp",
    "swollen lymph nodes",
    "back pain",
    "neck pain",
    "low back pain",
    "pain of the anus",
    "pain during pregnancy",
    "pelvic pain",
    "impotence",
    "infant spitting up",
    "vomiting blood",
    "regurgitation",
    "burning abdominal pain",
    "restlessness",
    "symptoms of infants",
    "wheezing",
    "peripheral edema",
    "neck mass",
    "ear pain",
    "jaw swelling",
    "mouth dryness",
    "neck swelling",
    "knee pain",
    "foot or toe pain",
    "bowlegged or knock-kneed",
    "ankle pain",
    "bones are painful",
    "knee weakness",
    "elbow pain",
    "knee swelling",
    "skin moles",
    "knee lump or mass",
    "weight gain",
    "problems with movement",
    "knee stiffness or tightness",
    "leg swelling",
    "foot or toe swelling",
    "heartburn",
    "smoking problems",
    "muscle pain",
    "infant feeding problem",
    "recent weight loss",
    "problems with shape or size of breast",
    "underweight",
    "difficulty eating",
    "scanty menstrual flow",
    "vaginal pain",
    "vaginal redness",
    "vulvar irritation",
    "weakness",
    "decreased heart rate",
    "increased heart rate",
    "bleeding or discharge from nipple",
    "ringing in ear",
    "plugged feeling in ear",
    "itchy ear(s)",
    "frontal headache",
    "fluid in ear",
    "neck stiffness or tightness",
    "spots or clouds in vision",
    "eye redness",
    "lacrimation",
    "itchiness of eye",
    "blindness",
    "eye burns or stings",
    "itchy eyelid",
    "feeling cold",
    "decreased appetite",
    "excessive appetite",
    "excessive anger",
    "loss of sensation",
    "focal weakness",
    "slurring words",
    "symptoms of the face",
    "disturbance of memory",
    "paresthesia",
    "side pain",
    "fever",
    "shoulder pain",
    "shoulder stiffness or tightness",
    "shoulder weakness",
    "arm cramps or spasms",
    "shoulder swelling",
    "tongue lesions",
    "leg cramps or spasms",
    "abnormal appearing tongue",
    "ache all over",
    "lower body pain",
    "problems during pregnancy",
    "spotting or bleeding during pregnancy",
    "cramps and spasms",
    "upper abdominal pain",
    "stomach bloating",
    "changes in stool appearance",
    "unusual color or odor to urine",
    "kidney mass",
    "swollen abdomen",
    "symptoms of prostate",
    "leg stiffness or tightness",
    "difficulty breathing",
    "rib pain",
    "joint pain",
    "muscle stiffness or tightness",
    "pallor",
    "hand or finger lump or mass",
    "chills",
    "groin pain",
    "fatigue",
    "abdominal distention",
    "regurgitation.1",
    "symptoms of the kidneys",
    "melena",
    "flushing",
    "coughing up sputum",
    "seizures",
    "delusions or hallucinations",
    "shoulder cramps or spasms",
    "joint stiffness or tightness",
    "pain or soreness of breast",
    "excessive urination at night",
    "bleeding from eye",
    "rectal bleeding",
    "constipation",
    "temper problems",
    "coryza",
    "wrist weakness",
    "eye strain",
    "hemoptysis",
    "lymphedema",
    "skin on leg or foot looks infected",
    "allergic reaction",
    "congestion in chest",
    "muscle swelling",
    "pus in urine",
    "abnormal size or shape of ear",
    "low back weakness",
    "sleepiness",
    "apnea",
    "abnormal breathing sounds",
    "excessive growth",
    "elbow cramps or spasms",
    "feeling hot and cold",
    "blood clots during menstrual periods",
    "absence of menstruation",
    "pulling at ears",
    "gum pain",
    "redness in ear",
    "fluid retention",
    "flu-like syndrome",
    "sinus congestion",
    "painful sinuses",
    "fears and phobias",
    "recent pregnancy",
    "uterine contractions",
    "burning chest pain",
    "back cramps or spasms",
    "stiffness all over",
    "muscle cramps, contractures, or spasms",
    "low back cramps or spasms",
    "back mass or lump",
    "nosebleed",
    "long menstrual periods",
    "heavy menstrual flow",
    "unpredictable menstruation",
    "painful menstruation",
    "infertility",
    "frequent menstruation",
    "sweating",
    "mass on eyelid",
    "swollen eye",
    "eyelid swelling",
    "eyelid lesion or rash",
    "unwanted hair",
    "symptoms of bladder",
    "irregular appearing nails",
    "itching of skin",
    "hurts to breath",
    "nailbiting",
    "skin dryness, peeling, scaliness, or roughness",
    "skin on arm or hand looks infected",
    "skin irritation",
    "itchy scalp",
    "hip swelling",
    "incontinence of stool",
    "foot or toe cramps or spasms",
    "warts",
    "bumps on penis",
    "too little hair",
    "foot or toe lump or mass",
    "skin rash",
    "mass or swelling around the anus",
    "low back swelling",
    "ankle swelling",
    "hip lump or mass",
    "drainage in throat",
    "dry or flaky scalp",
    "premenstrual tension or irritability",
    "feeling hot",
    "feet turned in",
    "foot or toe stiffness or tightness",
    "pelvic pressure",
    "elbow swelling",
    "elbow stiffness or tightness",
    "early or late onset of menopause",
    "mass on ear",
    "bleeding from ear",
    "hand or finger weakness",
    "low self-esteem",
    "throat irritation",
    "itching of the anus",
    "swollen or red tonsils",
    "irregular belly button",
    "swollen tongue",
    "lip sore",
    "vulvar sore",
    "hip stiffness or tightness",
    "mouth pain",
    "arm weakness",
    "leg lump or mass",
    "disturbance of smell or taste",
    "discharge in stools",
    "penis pain",
    "loss of sex drive",
    "obsessions and compulsions",
    "antisocial behavior",
    "neck cramps or spasms",
    "pupils unequal",
    "poor circulation",
    "thirst",
    "sleepwalking",
    "skin oiliness",
    "sneezing",
    "bladder mass",
    "knee cramps or spasms",
    "premature ejaculation",
    "leg weakness",
    "posture problems",
    "bleeding in mouth",
    "tongue bleeding",
    "change in skin mole size or color",
    "penis redness",
    "penile discharge",
    "shoulder lump or mass",
    "polyuria",
    "cloudy eye",
    "hysterical behavior",
    "arm lump or mass",
    "nightmares",
    "bleeding gums",
    "pain in gums",
    "bedwetting",
    "diaper rash",
    "lump or mass of breast",
    "vaginal bleeding after menopause",
    "infrequent menstruation",
    "mass on vulva",
    "jaw pain",
    "itching of scrotum",
    "postpartum problems of the breast",
    "eyelid retracted",
    "hesitancy",
    "elbow lump or mass",
    "muscle weakness",
    "throat redness",
    "joint swelling",
    "tongue pain",
    "redness in or around nose",
    "wrinkles on skin",
    "foot or toe weakness",
    "hand or finger cramps or spasms",
    "back stiffness or tightness",
    "wrist lump or mass",
    "skin pain",
    "low back stiffness or tightness",
    "low urine output",
    "skin on head or neck looks infected",
    "stuttering or stammering",
    "problems with orgasm",
    "nose deformity",
    "lump over jaw",
    "sore in nose",
    "hip weakness",
    "back swelling",
    "ankle stiffness or tightness",
    "ankle weakness",
    "neck weakness"
]

import os
import pickle
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from bson import ObjectId
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import io

# ─── Load ENV ────────────────────────────────────────────────────────────────
load_dotenv()
MONGO_URI = os.getenv('MONGO_URI')
FLASK_SECRET_KEY = os.getenv('FLASK_SECRET_KEY')
if not MONGO_URI or not FLASK_SECRET_KEY:
    raise RuntimeError("Missing MONGO_URI or FLASK_SECRET_KEY")

# ─── Flask + Mongo Setup ─────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = FLASK_SECRET_KEY
CORS(app, origins=["*"])
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
except ServerSelectionTimeoutError:
    raise RuntimeError("Failed to connect to MongoDB")
db = client['meddb']
drugs_collection = db.drugs

# ─── Load Your ML Artifacts ───────────────────────────────────────────────────
# Text-based disease prediction
text_model = load_model('disease_prediction_dnn.h5')
with open('scaler.pkl', 'rb') as f_scaler:
    scaler = pickle.load(f_scaler)
with open('label_encoder.pkl', 'rb') as f_le:
    label_encoder = pickle.load(f_le)

# Image-based medicine recognition
image_model = load_model('medicine_recognition.h5')
# Define class labels used in training (update accordingly)
class_labels = ['Alaxan', 'Bactidol', 'Bioflu', 'Biogesic', 'DayZinc', 'Decolgen', 'Fish Oil', 'Kremil S', 'Medicol', 'Neozep']

# ─── Load spaCy và chuẩn bị Matcher cho disease extraction ─────────────────
import spacy
from spacy.matcher import Matcher

# Load English model
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)
# Build patterns from disease_list
for d in disease_list:
    tokens = [tok.lower() for tok in d.split()]
    pattern = [{"LOWER": t} for t in tokens]
    matcher.add("DISEASE", [pattern])

# ─── Helper for image preprocessing

def preprocess_image(image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    arr = img_to_array(image) / 255.0
    return np.expand_dims(arr, axis=0)

# ─── New Medicine Endpoint ────────────────────────────────────────────────────
@app.route('/drugs/<string:name>', methods=['GET'])
def get_drug_by_name(name):
    # Case‐insensitive match; you can adjust to exact or regex
    query = { 'name': { '$regex': f'^{name}$', '$options': 'i' } }
    drug = drugs_collection.find_one(query)
    if not drug:
        abort(404, f"Drug '{name}' not found")
    # Convert ObjectId to string for JSON serialization
    drug['_id'] = str(drug['_id'])
    return jsonify(drug), 200

# ─── New Text Prediction Endpoint ────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict_disease():
    data = request.get_json(force=True)
    text = data.get("text")
    if not text:
        abort(400, "Missing 'text' field in JSON payload")

    doc = nlp(text)
    matches = matcher(doc)
    found = {doc[start:end].text for _, start, end in matches}
    extracted = list(found)
    vector = [1 if phrase in found else 0 for phrase in disease_list]

    try:
        X_new = scaler.transform([vector])
    except Exception as e:
        abort(400, f"Error scaling input data: {e}")

    preds = text_model.predict(X_new)
    idx = int(preds.argmax(axis=1)[0])
    predicted = label_encoder.inverse_transform([idx])[0]
    confidence = float(preds[0, idx])

    return jsonify({
        "extracted": extracted,
        "symptoms": vector,
        "predicted_disease": predicted,
        "confidence": confidence
    }), 200

# ─── Image Recognition Endpoint ─────────────────────────────────────────────
@app.route('/recognize_image', methods=['POST'])
def recognize_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    try:
        image = Image.open(io.BytesIO(file.read()))
        processed = preprocess_image(image)
        preds = image_model.predict(processed)

        idx = int(np.argmax(preds[0]))
        label = class_labels[idx]
        confidence = float(preds[0, idx])

        return jsonify({
            "prediction": label,
            "confidence": round(confidence, 4)
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
