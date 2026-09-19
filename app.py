import json
import numpy as np
import torch
import torch.nn as nn
import mysql.connector
from flask import Flask, request, jsonify
from flask_cors import CORS

DB_HOST = "127.0.0.1"
DB_PORT = 3306
DB_USER = "root"
DB_PASSWORD = ""
DB_NAME = "federated_health"

SYMPTOM_VOCAB = [
    "fever", "cough", "fatigue", "headache", "nausea", "chest_pain",
    "shortness_of_breath", "joint_pain", "rash", "dizziness",
    "vomiting", "sore_throat",
]

def get_conn():
    return mysql.connector.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, database=DB_NAME
    )

class DiseaseNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)

def load_latest_model():
    conn = get_conn()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM model_weights ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    if row is None:
        return None, None
    meta = json.loads(row["meta_json"])
    weights = json.loads(row["weights_json"])
    model = DiseaseNet(meta["input_dim"], meta["hidden_dim"], meta["num_classes"])
    state_dict = {k: torch.tensor(v) for k, v in weights.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, meta

app = Flask(__name__)
CORS(app)

@app.route("/symptoms", methods=["GET"])
def get_symptoms():
    return jsonify({"symptoms": SYMPTOM_VOCAB})

@app.route("/register", methods=["POST"])
def register_patient():
    data = request.get_json(force=True)
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO patients
        (name, age, gender, birthdate, weight_kg, height_cm, phone_number, address,
         emergency_contact_name, emergency_contact_relation, emergency_contact_phone,
         symptoms, past_conditions, family_history, diagnosis)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            data.get("name"),
            data.get("age"),
            data.get("gender"),
            data.get("birthdate"),
            data.get("weight_kg"),
            data.get("height_cm"),
            data.get("phone_number"),
            data.get("address"),
            data.get("emergency_contact_name"),
            data.get("emergency_contact_relation"),
            data.get("emergency_contact_phone"),
            ",".join(data.get("symptoms", [])),
            data.get("past_conditions"),
            data.get("family_history"),
            data.get("diagnosis"),
        ),
    )
    conn.commit()
    new_id = cursor.lastrowid
    cursor.close()
    conn.close()
    return jsonify({"patient_id": new_id})

@app.route("/predict", methods=["POST"])
def predict():
    model, meta = load_latest_model()
    if model is None:
        return jsonify({"error": "No trained model found in database yet"}), 400

    data = request.get_json(force=True)
    selected_symptoms = set(data.get("symptoms", []))
    gender = data.get("gender", "Other")
    age = float(data.get("age", 0))
    weight_kg = float(data.get("weight_kg", 0))
    height_cm = float(data.get("height_cm", 0))

    vector = []
    for col in meta["feature_cols"]:
        if col == "age":
            vector.append(age)
        elif col == "weight_kg":
            vector.append(weight_kg)
        elif col == "height_cm":
            vector.append(height_cm)
        elif col.startswith("gender_"):
            vector.append(1.0 if col == f"gender_{gender}" else 0.0)
        elif col in meta["symptom_vocab"]:
            vector.append(1.0 if col in selected_symptoms else 0.0)
        else:
            vector.append(0.0)

    x = torch.tensor(np.array([vector], dtype=np.float32))
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).numpy()[0]

    order = np.argsort(probs)[::-1]
    classes = meta["classes"]
    top_k = [{"disease": classes[i], "confidence": float(probs[i])} for i in order[:5]]

    return jsonify({
        "prediction": top_k[0]["disease"],
        "confidence": top_k[0]["confidence"],
        "top_k": top_k,
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
