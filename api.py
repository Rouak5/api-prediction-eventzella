from fastapi import FastAPI, Request
import joblib
import pandas as pd
import json

app = FastAPI()

# Chargement du modèle
model = joblib.load("eventzella_best_model_Random_Forest.pkl")

# 1. Route GET pour vérifier que l'API est allumée (Testable dans Chrome)
@app.get("/")
def read_root():
    return {"message": "✅ L'API fonctionne parfaitement ! Retournez sur Power BI."}

# 2. Route POST pour la prédiction (Uniquement pour Power BI)
@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()
        df = pd.DataFrame(json.loads(data))
        predictions = model.predict(df)
        return predictions.tolist()
    except Exception as e:
        # En cas de problème de données, l'API ne plante pas mais renvoie l'erreur
        return [f"Erreur API: {str(e)}"]