import streamlit as st
import pandas as pd
from joblib import load
from sklearn.inspection import partial_dependence
import datetime
import plotly.express as px
from Home import load_data
import joblib
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data():
    url = "https://raw.githubusercontent.com/remijul/dataset/master/Airline%20Passenger%20Satisfaction.csv"
    return pd.read_csv(url, delimiter=';')

if st.button("Entraîner le modèle de machine learning"):
    # Charger le modèle entraîné
    data = load_data()

    # Suppression des Na
    data.dropna(inplace=True)
    
    # Convertion des colonnes catégorielles en type "category"
    categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    data[categorical_columns] = data[categorical_columns].astype('category')

    # Map de la colonne 'Satisfaction' en valeurs numériques pour la prédiction
    data['Satisfaction'] = data['Satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})
    X = data.drop('Satisfaction', axis=1)
    y = data['Satisfaction']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numerical_columns = ['Age', 'Flight Distance', 'Seat comfort', 'Departure/Arrival time convenient', 'Food and drink', 'Gate location', 'Inflight wifi service', 'Inflight entertainment', 'Online support', 'Ease of Online booking', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Cleanliness', 'Online boarding', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']

    # Transformer pour les colonnes numériques et catégorielles
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()

    # Préprocessing qui applique les transformations appropriées aux colonnes correspondantes
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

    # Création pipeline qui inclut le préprocessing et un classifier
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Entraînement
    pipeline.fit(X_train, y_train)

    # Prédiction de la satisfaction des passagers sur les données de test
    y_pred = pipeline.predict(X_test)

    # Calcul de la précision et rapport de classification
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f'Accuracy: {accuracy:.2f}')
    st.text(classification_report(y_test, y_pred))

    # Affichage du diagramme dans Streamlit
    st.image('pipeline.png')
