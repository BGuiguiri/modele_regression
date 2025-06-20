import streamlit as st

# Titre de l'application
st.title("Table de Multiplication")

# Description
st.write("Choisissez un nombre pour afficher sa table de multiplication de 1 à 10.")

# Entrée utilisateur
number = st.number_input("Entrez un nombre", min_value=1, max_value=100, value=1, step=1)

# Afficher la table de multiplication
st.header(f"Table de multiplication de {number}")
for i in range(1, 11):
    result = number * i
    st.write(f"{number} × {i} = {result}")
