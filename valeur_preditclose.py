# Charger le modèle et préparer les données
from model_pipeline import load_model, prepare_data
import pandas as pd
import numpy as np

# 1️⃣ Charger le dataset original
data_path = "insurance.csv"
target_col = "charges"
df = pd.read_csv(data_path)

# 2️⃣ Préparer les données (encode + scale)
X_train, X_test, y_train, y_test = prepare_data(data_path, target=target_col)

# 3️⃣ Charger le modèle
model = load_model("model.pkl")

# 4️⃣ Faire les prédictions
y_pred = model.predict(X_test)

# 5️⃣ Calculer les erreurs absolues
errors = np.abs(y_test - y_pred)

# 6️⃣ Trouver l'indice de la meilleure prédiction
best_idx = np.argmin(errors)

# 7️⃣ Retrouver l'index correspondant dans le dataset original
original_index = y_test.index[best_idx]

# 8️⃣ Afficher la ligne du dataset, la valeur prédite et l'erreur
print(f"Index dans le dataset original : {original_index}")
print("Ligne du dataset :")
print(df.loc[original_index])
print(f"Valeur prédite : {y_pred[best_idx]}")
print(f"Erreur absolue : {errors.iloc[best_idx]}")

