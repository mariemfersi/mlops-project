# -*- coding: utf-8 -*-
import pandas as pd
from model_pipeline import prepare_data, select_and_train_model
import matplotlib.pyplot as plt

# Charger et préparer les données
X_train, X_test, y_train, y_test = prepare_data("insurance.csv", target="charges")

# Entraîner le modèle
model = select_and_train_model(X_train, y_train)

# Faire les prédictions
y_pred = model.predict(X_test)

# Créer un DataFrame pour comparer
comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred,
    "Error": y_test - y_pred
})

# Trier par erreur absolue décroissante
comparison["AbsError"] = comparison["Error"].abs()
comparison_sorted = comparison.sort_values(by="AbsError", ascending=False)

# Afficher les 10 plus grandes erreurs
print("Top 10 erreurs les plus importantes :")
print(comparison_sorted.head(10))

# Optionnel : visualiser Actual vs Predicted
plt.figure(figsize=(6,6))
plt.scatter(comparison["Actual"], comparison["Predicted"], alpha=0.7)
plt.plot([comparison["Actual"].min(), comparison["Actual"].max()],
         [comparison["Actual"].min(), comparison["Actual"].max()],
         "r--")
plt.xlabel("Actual charges")
plt.ylabel("Predicted charges")
plt.title("Actual vs Predicted")
plt.show()

