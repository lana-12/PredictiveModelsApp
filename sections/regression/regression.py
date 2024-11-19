import pandas as pd
import streamlit as st
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, '../../data/diabete.csv')

def load_data():
    """Load data and delete col"""
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
        if 'Unnamed: 0' in data.columns:
            data = data.drop(columns=['Unnamed: 0'])
        st.session_state.data = data
        return data
    else:
        st.error("Le fichier est introuvable.")
        return None

# def traitement_dataframe():
#     """Display and clean data."""
#     data = load_data()
#     if data is not None:
#         st.write("Données après suppression de 'Unnamed: 0':")
#         st.dataframe(data.head())
#         st.session_state.data = data
        
def display_dataframe_info():
    """Display basic information about the DataFrame"""
    st.header("Page Info DataFrame")
    st.caption("Informations sur le fichier diabète")
    data = load_data() 
    if data is not None:
        st.write("Voici les 5 premières lignes du fichier :")
        st.dataframe(data.head())
        st.write(f"Nombre de colonnes : {data.shape[1]}")
        st.write(f"Nombre de lignes : {data.shape[0]}")
        st.write("Résumé des statistiques :")
        st.dataframe(data.describe())
        st.write("Valeur manquante ou null :")
        st.dataframe(data.isna().sum())
        
        # Save data => state de session for access global
        st.session_state.data = data
    else:
        st.error("Le fichier est introuvable.")

def analyse_dataframe():
    """Analyze data, show correlations, and select features"""
    data = load_data()
    if data is not None:
        st.write("Analyse des corrélations")
        correlation_matrix = data.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

        target_corr = correlation_matrix["target"].sort_values(ascending=False)
        st.write("Corrélation avec la cible (target):")
        st.write(target_corr)

        selected_features = target_corr[target_corr >= 0.40].index.tolist()
        selected_features.remove('target')
        st.write("Features sélectionnées avec une corrélation >= 0.40 avec la cible :")
        st.write(selected_features)
        
        st.session_state.selected_features = selected_features

def train_model(model, model_name):
    """Generic function to train and evaluate a model"""
    if 'data' in st.session_state:
        data = st.session_state.data
        selected_features = st.session_state.selected_features if 'selected_features' in st.session_state else data.columns.tolist()
        
        X = data[selected_features]
        y = data['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.write(f"{model_name} - Mean Squared Error: {mse}")
        st.write(f"{model_name} - R²: {r2}")
        st.write(f"{model_name} - Train score: {model.score(X_train, y_train)}")
        st.write(f"{model_name} - Test score: {model.score(X_test, y_test)}")
    else:
        st.error("Les données traitées sont introuvables. Veuillez d'abord les traiter et analyser.")

def linear_dataframe():
    """Run linear regression."""
    analyse_dataframe()
    model = LinearRegression()
    train_model(model, "Régression Linéaire")

def decision_tree_dataframe():
    """Run decision tree regression."""
    analyse_dataframe()
    model = DecisionTreeRegressor(random_state=1000)
    train_model(model, "Arbre de Décision")

def regression_page():
    """Main page layout."""
    st.header("Bienvenue")
    st.caption("Bienvenue dans le Playground de Régression")

    if st.button("Afficher les informations sur le DataFrame", use_container_width=True):
        display_dataframe_info()

    options = st.selectbox(
        "Veuillez choisir un modèle",
        ["", "Régression Linéaire", "Arbre de Décision"],
        format_func=lambda x: "Sélectionnez un modèle" if x == "" else x
    )

    if options == "Régression Linéaire":
        st.header("Régression Linéaire")
        linear_dataframe()
    elif options == "Arbre de Décision":
        st.header("Arbre de Décision")
        decision_tree_dataframe()
