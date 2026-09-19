!pip install mysql-connector-python pandas scikit-learn torch -q

import json
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import mysql.connector
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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

conn = mysql.connector.connect(
    host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, database=DB_NAME
)
df = pd.read_sql("SELECT * FROM patients WHERE diagnosis IS NOT NULL AND diagnosis <> ''", conn)
conn.close()

df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(df["age"].median())
df["weight_kg"] = pd.to_numeric(df["weight_kg"], errors="coerce").fillna(df["weight_kg"].median())
df["height_cm"] = pd.to_numeric(df["height_cm"], errors="coerce").fillna(df["height_cm"].median())
df["gender"] = df["gender"].fillna("Other")
df["symptoms"] = df["symptoms"].fillna("")

gender_dummies = pd.get_dummies(df["gender"], prefix="gender")

symptom_matrix = pd.DataFrame(
    {
        symptom: df["symptoms"].apply(lambda s: 1 if symptom in [x.strip() for x in s.split(",")] else 0)
        for symptom in SYMPTOM_VOCAB
    }
)

numeric_features = df[["age", "weight_kg", "height_cm"]].reset_index(drop=True)

X_df = pd.concat(
    [numeric_features, gender_dummies.reset_index(drop=True), symptom_matrix.reset_index(drop=True)],
    axis=1,
)
feature_cols = X_df.columns.tolist()
X = X_df.values.astype(np.float32)

encoder = LabelEncoder()
y = encoder.fit_transform(df["diagnosis"].astype(str).values)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
)

NUM_CLIENTS = 3
ROUNDS = 15
LOCAL_EPOCHS = 3
LOCAL_LR = 0.01
BATCH_SIZE = 16
USE_DP = True
DP_CLIP_NORM = 1.0
DP_NOISE_STD = 0.01

client_indices = np.array_split(np.random.permutation(len(X_train)), NUM_CLIENTS)
client_data = [(X_train[idx], y_train[idx]) for idx in client_indices]

INPUT_DIM = X.shape[1]
HIDDEN_DIM = 128
NUM_CLASSES = len(encoder.classes_)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_model = DiseaseNet(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES).to(device)

def clip_and_noise(state_dict, ref_state_dict):
    clipped = {}
    for k in state_dict:
        delta = state_dict[k] - ref_state_dict[k]
        norm = torch.norm(delta)
        factor = min(1.0, DP_CLIP_NORM / (norm + 1e-6))
        delta = delta * factor
        noise = torch.randn_like(delta) * DP_NOISE_STD
        clipped[k] = ref_state_dict[k] + delta + noise
    return clipped

def local_train(model, X_c, y_c):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=LOCAL_LR, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    X_t = torch.tensor(X_c).to(device)
    y_t = torch.tensor(y_c, dtype=torch.long).to(device)
    n = len(X_t)
    for _ in range(LOCAL_EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = X_t[idx], y_t[idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
    return model.state_dict()

def average_weights(weight_list):
    avg = copy.deepcopy(weight_list[0])
    for k in avg:
        for i in range(1, len(weight_list)):
            avg[k] += weight_list[i][k]
        avg[k] = avg[k] / len(weight_list)
    return avg

def evaluate(model, X_e, y_e):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_e).to(device)
        y_t = torch.tensor(y_e, dtype=torch.long).to(device)
        out = model(X_t)
        preds = torch.argmax(out, dim=1)
        acc = (preds == y_t).float().mean().item()
    return acc

final_acc = 0.0
for rnd in range(1, ROUNDS + 1):
    global_state = global_model.state_dict()
    local_states = []
    for X_c, y_c in client_data:
        local_model = DiseaseNet(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES).to(device)
        local_model.load_state_dict(global_state)
        new_state = local_train(local_model, X_c, y_c)
        if USE_DP:
            new_state = clip_and_noise(new_state, global_state)
        local_states.append(new_state)
    global_model.load_state_dict(average_weights(local_states))
    final_acc = evaluate(global_model, X_test, y_test)
    print(f"round {rnd}/{ROUNDS} accuracy {final_acc:.4f}")

weights_serializable = {k: v.cpu().tolist() for k, v in global_model.state_dict().items()}
meta = {
    "feature_cols": feature_cols,
    "classes": encoder.classes_.tolist(),
    "input_dim": INPUT_DIM,
    "hidden_dim": HIDDEN_DIM,
    "num_classes": NUM_CLASSES,
    "symptom_vocab": SYMPTOM_VOCAB,
}

conn = mysql.connector.connect(
    host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, database=DB_NAME
)
cursor = conn.cursor()
cursor.execute(
    "INSERT INTO model_weights (round_number, weights_json, meta_json, accuracy) VALUES (%s, %s, %s, %s)",
    (ROUNDS, json.dumps(weights_serializable), json.dumps(meta), float(final_acc)),
)
conn.commit()
cursor.close()
conn.close()
