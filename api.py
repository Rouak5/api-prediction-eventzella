from fastapi import FastAPI, Request
import joblib
import pandas as pd

app = FastAPI()

# Chargement de votre modèle IA
model = joblib.load("eventzella_best_model_Random_Forest.pkl")

@app.get("/")
def read_root():
    return {"message": "✅ L'API fonctionne parfaitement !"}

@app.post("/predict")
async def predict(request: Request):
    try:
        # 1. On récupère directement les données (FastAPI les a déjà traduites en liste)
        data = await request.json()
        
        # 2. On transforme la liste en tableau de données (DataFrame)
        df = pd.DataFrame(data)
        
        # 3. Nettoyage magique : on supprime les espaces invisibles avant et après les noms de colonnes
        df.columns = df.columns.str.strip()
        
        # 4. On supprime la colonne "Somme de ID" ou "ID" si Power BI l'a envoyée par erreur
        colonnes_a_supprimer = [col for col in df.columns if "ID" in col]
        df = df.drop(columns=colonnes_a_supprimer)
        
        # 5. On fait la prédiction avec le modèle !
        prediction = model.predict(df)
        
        # 6. On renvoie le chiffre d'affaires (arrondi à 2 chiffres après la virgule pour faire propre)
        resultat_ca = round(prediction[0], 2)
        
        return [f"{resultat_ca} €"]
        
    except Exception as e:
        return [f"Erreur API : {str(e)}"]
