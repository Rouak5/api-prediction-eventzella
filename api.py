from fastapi import FastAPI, Request
import joblib
import pandas as pd
import json

app = FastAPI()

model = joblib.load("eventzella_best_model_Random_Forest.pkl")

@app.get("/")
def read_root():
    return {"message": "✅ L'API fonctionne parfaitement !"}

@app.post("/predict")
async def predict(request: Request):
    try:
        # 1. Vérifier si l'enveloppe envoyée par Power Automate est vide
        body_bytes = await request.body()
        if not body_bytes:
            return ["Erreur : Aucune donnée reçue par l'API (Le Corps HTTP est vide)"]
        
        # 2. Lire les données envoyées
        data = json.loads(body_bytes)
        
        # Sécurité : si les données arrivent sous forme de texte, on les re-décode
        if isinstance(data, str):
            data = json.loads(data)
        
        df = pd.DataFrame(data)
        
        # 3. Dictionnaire pour traduire le langage brut de Power Automate pour l'IA
        colonnes_mapping = {
            'Valeur budget 2': 'budget',
            'Valeur marketing_spend': 'marketing_spend',
            'Valeur new_beneficiaries': 'new_beneficiaries',
            'Valeur rating': 'rating',
            'Valeur event_duration_days': 'event_duration_days',
            'Valeur is_weekend': 'is_weekend',
            'Valeur Month': 'month',
            'Valeur day_of_week': 'day_of_week',
            'Valeur week_of_year': 'week_of_year',
            'Valeur quarter': 'quarter',
            'Selected_event_type': 'event_type',
            'Selected_provider_service_type': 'provider_service_type',
            'Selected_category_name': 'category_name',
            'Selected_region': 'region'
        }
        
        # Renommer les colonnes
        df = df.rename(columns=colonnes_mapping)
        
        # Nettoyer les surplus
        if 'ID' in df.columns:
            df = df.drop(columns=['ID'])
        df = df.drop_duplicates().head(1)
        
        # 4. Faire la prédiction
        predictions = model.predict(df)
        return predictions.tolist()
        
    except Exception as e:
        return [f"Erreur technique de l'API : {str(e)}"]
