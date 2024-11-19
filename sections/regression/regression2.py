import pandas as pd
import streamlit as st
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
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
        ax.set_title("Matrice de corrélations")
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

        target_corr = correlation_matrix["target"].sort_values(ascending=False)
        st.write("Corrélation avec la cible (target):")
        st.write(target_corr)
        threshold = st.slider(
            "Veuillez entrer un seuil de corrélation pour sélectionner les features", 
            min_value=-0.6, max_value=1.0, value=0.0, step=0.01)

        selected_features = target_corr[target_corr >= threshold].index.tolist()
        selected_features.remove('target')
        st.write(f"Features sélectionnées avec une corrélation >= {threshold} avec la cible :")
        st.write(selected_features)
        
        st.session_state.selected_features = selected_features

def train_model(model, model_name):
    """Generic function to train, evaluate, and plot model performance."""
    if 'data' in st.session_state:
        data = st.session_state.data
        selected_features = st.session_state.selected_features if 'selected_features' in st.session_state else data.columns.tolist()
        
        X = data[selected_features]
        y = data['target']
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
       
        #Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        #Evaluate model
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        col_test, col_train = st.columns(2)
        with col_test:
            st.header('Test results')
            st.write(f"{model_name} - Mean Squared Error: {test_mse}")
            st.write(f"{model_name} - R²: {test_r2}")
            st.write(f"{model_name} - Test score: {model.score(X_test, y_test)}")
            
        with col_train: 
            st.header('Train results')  
            st.write(f"{model_name} - Mean Squared Error: {train_mse}")
            st.write(f"{model_name} - R²: {train_r2}")
            st.write(f"{model_name} - Train score: {model.score(X_train, y_train)}")
        
        
        # Plot predictions and actual values
        fig, ax = plt.subplots(figsize=(10, 6))
        # Value reel en bleu
        ax.scatter(np.arange(len(y_test)), y_test, color='blue', label="Valeurs réelles")
        # Value predict en rouge
        ax.scatter(np.arange(len( y_test_pred)),  y_test_pred, color='red', label="Valeurs prédites")
        ax.set_title("Valeurs réelles et prédites")
        ax.set_xlabel("Index des observations")
        ax.set_ylabel("Valeurs")
        ax.legend()
        st.pyplot(fig)
        
        
        row1Col1,row1Col2 = st.columns(2)
        with row1Col1:
        #Plotting predictions vs test results
            fig_test, ax_test = plt.subplots()
            ax_test.scatter(y_test, y_test_pred, color='blue', label="Test Predictions")
            ax_test.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label="Perfect Prediction")
            ax_test.set_xlabel('True Values')
            ax_test.set_ylabel('Predictions')
            ax_test.set_title('True Values vs Predictions (Test Set)')
            ax_test.legend()
            st.pyplot(fig_test)
            
        with row1Col2:
            # Plotting predctions vs train results
            fig_train, ax_train = plt.subplots()
            ax_train.scatter(y_train, y_train_pred, color='green', label="Train Predictions")
            ax_train.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linewidth=2, label="Perfect Prediction")
            ax_train.set_xlabel('True Values')
            ax_train.set_ylabel('Predictions')
            ax_train.set_title('True Values vs Predictions (Train Set)')
            ax_train.legend()
            st.pyplot(fig_train)
            
        row2Col1,row2Col2 = st.columns(2)
        with row2Col1:
        # Plot predictions vs actual values
        # 'plasma', 'inferno', 'magma
            fig, ax = plt.subplots()
            scatter = ax.scatter(y_test,  y_test_pred, alpha=0.5, c=y_test, cmap='viridis', label="Prédictions vs Réelles")
            fig.colorbar(scatter, ax=ax, label='Valeurs réelles (y_test)')
            ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2, label="Référence")
            ax.set_xlabel("Valeurs réelles")
            ax.set_ylabel("Prédictions")
            ax.set_title(f"{model_name} - Prédictions vs Valeurs réelles")
            ax.legend()
            st.pyplot(fig)
            
        with row2Col2:
            # Plot distribution of residuals
            #  différence entre les valeurs réelle and prédict
            residuals = y_test -  y_test_pred
            fig, ax = plt.subplots()
            sns.histplot(residuals, kde=True, ax=ax, color='skyblue')
            ax.set_title(f"{model_name} - Distribution des résidus")
            ax.set_xlabel("Erreur (résidus)")
            st.pyplot(fig)
            
        row3Col3, row3Col4 = st.columns(2)
        with row3Col3:
            # Plot distribution of the prediction -coolwarm
            fig, ax = plt.subplots()
            sns.histplot( y_test_pred, kde=True, ax=ax, color='green') 
            ax.set_title(f"{model_name} - Distribution des prédictions")
            ax.set_xlabel("Valeurs prédites")
            ax.set_ylabel("Fréquence")
            st.pyplot(fig)
            
        with row3Col4:
            # Plot distribution of the reel value
            fig, ax = plt.subplots()
            sns.histplot(y_test, kde=True, ax=ax, color='orange') 
            ax.set_title(f"{model_name} - Distribution des valeurs réelles")
            ax.set_xlabel("Valeurs réelles")
            ax.set_ylabel("Fréquence")
            st.pyplot(fig)
        
    else:
        st.error("Les données traitées sont introuvables. Veuillez d'abord les traiter et analyser.")


def linear_dataframe():
    """Run linear regression"""
    analyse_dataframe()
    model = LinearRegression()
    train_model(model, "Régression Linéaire")

def decision_tree_dataframe():
    """Run decision tree regression"""
    analyse_dataframe()
    model = DecisionTreeRegressor(random_state=1000)
    train_model(model, "Arbre de Décision")
    
def lasso_dataframe():
    """Run Lasso regression"""
    analyse_dataframe()
    # Controle la force de la régularisation
    #  Previent du surapprentissage (overfit)
    alpha = st.slider("Sélectionnez le paramètre alpha pour le modèle Lasso", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    model = Lasso(alpha=alpha, random_state=42)
    train_model(model, "Lasso")
    
def interactions_features():
    data = load_data()
    st.write("Analyse des corrélations")
    correlation_matrix = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Matrice de corrélations")
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)
    
    # Interaction features
    data['age_bmi_interaction'] = data['age'] * data['bmi']
    data['bp_age_interaction'] = data['bp'] * data['age']
    data['sex_bmi_interaction'] = data['sex'] * data['bmi']
    data['sex_age_interaction'] = data['sex'] * data['age']

    #Separting X and Y
    X = data.drop('target', axis=1)
    y = data['target']

    #Divisin and entrain model :
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    st.write(f"Training data shape: {X_train.shape}")
    st.write(f"Test data shape: {X_test.shape}")
    st.write(f"Training labels shape: {y_train.shape}")
    st.write(f"Test labels shape: {y_test.shape}")

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Display model coeffcients and intercept :
    st.write("Model Coefficients:", model.coef_)
    st.write("Model Intercept:", model.intercept_)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluate model
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    #Show Metrics
    st.write(f"Training Mean Squared Error (MSE): {train_mse}")
    st.write(f"Training Coefficient of Determination (R²): {train_r2}")
    st.write(f"Test Mean Squared Error (MSE): {test_mse}")
    st.write(f"Test Coefficient of Determination (R²): {test_r2}")

    #comparaison real and test
    comparison_data = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})
    st.write("Comparing Predicted vs Actual values (Test Set)")
    st.write(comparison_data)

    # Plotting  Predictions vs test
    fig_test, ax_test = plt.subplots()
    ax_test.scatter(y_test, y_test_pred, color='blue', label="Test Predictions")
    ax_test.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label="Perfect Prediction")
    ax_test.set_xlabel('True Values')
    ax_test.set_ylabel('Predictions')
    ax_test.set_title('True Values vs Predictions (Test Set)')
    ax_test.legend()
    st.pyplot(fig_test)

    # Plotting predictions vs train 
    fig_train, ax_train = plt.subplots()
    ax_train.scatter(y_train, y_train_pred, color='green', label="Train Predictions")
    ax_train.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linewidth=2, label="Perfect Prediction")
    ax_train.set_xlabel('True Values')
    ax_train.set_ylabel('Predictions')
    ax_train.set_title('True Values vs Predictions (Train Set)')
    ax_train.legend()
    st.pyplot(fig_train)

def ridge_model():
    analyse_dataframe()
    # Controle la force de la régularisation
    #  Previent du surapprentissage (overfit)
    alpha = st.slider("Sélectionnez le paramètre alpha pour le modèle Lasso", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    model = Ridge(alpha=alpha, random_state=42)
    train_model(model, "Ridge")
    
def cross_validation_linear():
    data = load_data()
    #Separating data:
    X = data.drop(columns=['target'])
    y = data['target']   
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #  Cross-validation to get MSE scores
    model = LinearRegression()
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

    # Convert the negative MSE values to positive 
    cv_mse = -cv_scores

    # Calculate the mean and the std of the MSE across the folds
    mean_mse = np.mean(cv_mse)
    std_mse = np.std(cv_mse)

    # Calculate R² scores for cross-validation
    cv_r2_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    mean_r2 = np.mean(cv_r2_scores)
    std_r2 = np.std(cv_r2_scores)

    # Train the model 
    model.fit(X_train, y_train)

    #Predictions 
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluate on the test set
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Display results
    st.write("Cross-Validation Mean Squared Error (MSE) across 5 folds:", mean_mse)
    st.write("Cross-Validation Std of MSE:", std_mse)
    st.write("Cross-Validation R² across 5 folds:", mean_r2)
    st.write("Cross-Validation Std of R²:", std_r2)
    st.write(f"Training Mean Squared Error (MSE): {train_mse:.2f}")
    st.write(f"Training R²: {train_r2:.2f}")
    st.write(f"Test Mean Squared Error (MSE): {test_mse:.2f}")
    st.write(f"Test R²: {test_r2:.2f}")

    # Plotting MSE: Cross-Validation vs Test ans train results
    fig_mse, ax_mse = plt.subplots(figsize=(10, 6))
    ax_mse.bar(range(1, 6), cv_mse, color='blue', label='CV MSE')
    ax_mse.axhline(mean_mse, color='red', linestyle='--', label=f'Mean CV MSE = {mean_mse:.2f}')
    ax_mse.bar(6, test_mse, color='yellow', label=f'Test Set MSE = {test_mse:.2f}')
    ax_mse.bar(7, train_mse, color='green', label=f'Train Set MSE = {train_mse:.2f}')

    ax_mse.set_xticks(range(1, 8))
    ax_mse.set_xticklabels([f'Fold {i}' for i in range(1, 6)] + ['Test', 'Train'])
    ax_mse.set_xlabel('Fold')
    ax_mse.set_ylabel('Mean Squared Error')
    ax_mse.set_title('Comparison of MSE Scores: Cross-Validation, Test, and Train Sets')
    ax_mse.legend()
    st.pyplot(fig_mse)

    # Plotting R² : Cross-Validation vs Test and train results
    fig_r2, ax_r2 = plt.subplots(figsize=(10, 6))
    ax_r2.bar(range(1, 6), cv_r2_scores, color='purple', label='CV R² ')
    ax_r2.axhline(mean_r2, color='red', linestyle='--', label=f'Mean CV R² = {mean_r2:.2f}')
    ax_r2.bar(6, test_r2, color='yellow', label=f'Test Set R² = {test_r2:.2f}')
    ax_r2.bar(7, train_r2, color='green', label=f'Train Set R² = {train_r2:.2f}')

    ax_r2.set_xticks(range(1, 8))
    ax_r2.set_xticklabels([f'Fold {i}' for i in range(1, 6)] + ['Test', 'Train'])
    ax_r2.set_xlabel('Fold')
    ax_r2.set_ylabel('R²')
    ax_r2.set_title('Comparison of R² Scores: Cross-Validation, Test, and Train Sets')
    ax_r2.legend()
    st.pyplot(fig_r2)

def regression2_page():
    """Main page layout"""
    st.header("Bienvenue")
    st.caption("Bienvenue dans le Playground de Régression")

    if st.button("Afficher les informations sur le DataFrame", use_container_width=True):
        display_dataframe_info()

    options = st.selectbox(
        "Veuillez choisir un modèle",
        ["", "Régression Linéaire", "Arbre de Décision", "Lasso","Interaction features", "Ridge", "Cross Validation" ],
        format_func=lambda x: "Sélectionnez un modèle" if x == "" else x
    )

    if options == "Régression Linéaire":
        st.header("Régression Linéaire")
        linear_dataframe()
    elif options == "Arbre de Décision":
        st.header("Arbre de Décision")
        decision_tree_dataframe()
    elif options == "Lasso":
        st.header("Lasso")
        lasso_dataframe()
    elif options == "Interaction features":
        st.header("Interaction features")
        interactions_features()
    elif options == "Ridge":
        st.header("Ridge")
        ridge_model()
    elif options == "Cross Validation":
        st.header("Cross Validation")
        cross_validation_linear()