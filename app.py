from fastapi import FastAPI
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

app = FastAPI()

# Load dataset and model
dataset_path = "Disease_symptom_and_patient_profile_dataset.csv"
df = pd.read_csv(dataset_path)
disease_labels = df['Disease'].unique()
num_labels = len(disease_labels)
label_to_id = {label: idx for idx, label in enumerate(disease_labels)}
id_to_label = {idx: label for label, idx in label_to_id.items()}

valid_symptoms = ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]

tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
model = BertForSequenceClassification.from_pretrained("dmis-lab/biobert-v1.1", num_labels=num_labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.get("/")
def home():
    return {"message": "Disease Prediction API is Running"}

@app.post("/predict/")
def predict(symptoms: str):
    symptom_list = [s.strip().title() for s in symptoms.split(",")]
    if not all(symptom in valid_symptoms for symptom in symptom_list):
        return {"error": "Invalid symptom(s). Use: " + ", ".join(valid_symptoms)}

    symptom_text = " ".join(symptom_list)
    tokens = tokenizer(symptom_text, padding=True, truncation=True, return_tensors="pt")
    tokens = {key: value.to(device) for key, value in tokens.items()}

    with torch.no_grad():
        outputs = model(**tokens)
    
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    predicted_disease = id_to_label.get(predicted_class, "Unknown Disease")

    return {"Predicted Disease": predicted_disease}

