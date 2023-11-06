import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# charger le dataset :
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
         'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
data = pd.read_csv(url, names=names)

# Convert non-numeric columns to numeric
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Remplacer les valeurs manquantes avec la moyenne de la colonne correspondante
data = data.apply(lambda x: x.fillna(x.mean()), axis=0)

# Préparer les données pour l'entraînement
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Centrer et réduire les données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entraîner le modèle de forêt aléatoire
clf = RandomForestClassifier(random_state=100)
clf.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = clf.predict(X_test)

# Calculer l'exactitude des prédictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))

# Définir une nouvelle entrée de données pour tester le modèle
new_data = pd.DataFrame({
    'age': [50],
    'sex': [1],
    'cp': [2],
    'trestbps': [120],
    'chol': [240],
    'fbs': [0],
    'restecg': [1],
    'thalach': [160],
    'exang': [0],
    'oldpeak': [1.5],
    'slope': [2],
    'ca': [0],
    'thal': [2]
})

# Centrer et réduire les données de la nouvelle entrée
new_data_scaled = scaler.transform(new_data)

# Faire des prédictions sur la nouvelle entrée de données
prediction = clf.predict(new_data_scaled)

# Afficher le pourcentage que la personne a une maladie cardiaque
if prediction[0] == 1:
    print("La personne a une maladie cardiaque avec une probabilité de", round(clf.predict_proba(new_data_scaled)[0][1]*100,2), "%")
    b=round(clf.predict_proba(new_data_scaled)[0][1]*100,2)
    print(b)
    a=100-b
    print(a)
else:
    print("La personne n'a pas de maladie cardiaque avec une probabilité de", round(clf.predict_proba(new_data_scaled)[0][0]*100,2), "%")
    a=round(clf.predict_proba(new_data_scaled)[0][0]*100,2)
    print(a)
    b=100-a
    print(b)

# Définir les données pour le graphique circulaire
#proba = clf.predict_proba(new_data_scaled)[0]
labels = ["Pas de maladie", "Maladie"]
proba_labels = [a, b]
labels = ["Pas de maladie", "Maladie"]

colors = ['green', 'red']

# Créer le graphique circulaire
plt.pie(proba_labels, labels=labels, colors=colors, autopct='%1.1f%%')

# Ajouter un titre
plt.title('Probabilité de maladie cardiaque')

# Afficher le graphique
plt.show()