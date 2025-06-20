
import streamlit as st
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
# Charger le modèle sérialisé
with open('modele_regression.pkl', 'rb') as file:
    model = pickle.load(file)

# Titre de l'application
st.title("Prédiction avec Régression Linéaire")

# Entrées utilisateur
st.header("Entrez deux caractéristiques")
feature1 = st.number_input("Caractéristique 1", value=0.0, step=1.0)
feature2 = st.number_input("Caractéristique 2", value=0.0, step=1.0)

# Préparer les données pour la prédiction
input_data = np.array([[feature1, feature2]])

# Faire une prédiction
if st.button("Prédire"):
    prediction = model.predict(input_data)
    st.success(f"Prédiction : {prediction[0]:.2f}")
