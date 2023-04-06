import streamlit as st
import pandas as pd
import streamlit as st
from joblib import load
from sklearn.inspection import partial_dependence
import datetime
import plotly.express as px
from Home import load_data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def load_and_prepare_data():
    # Import des données
    url = "https://raw.githubusercontent.com/remijul/dataset/master/Airline%20Passenger%20Satisfaction.csv"
    data = pd.read_csv(url, delimiter=';')

    # Préparation des données
    X = data.drop(columns=['Satisfaction'])
    y = data['Satisfaction']

    # Encoder les variables catégorielles
    le = LabelEncoder()
    X = X.apply(le.fit_transform)
    y = le.fit_transform(y)

    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalisation des données
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_and_prepare_data()

if st.button("Entraîner le réseau de neurones"):
    # Création du modèle
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compilation du modèle
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Entraînement du modèle
    epochs = 10
    batch_size = 32
    for epoch in range(epochs):
        st.text(f"Epoch {epoch+1}/{epochs}")
        history = model.fit(X_train, y_train, epochs=1, batch_size=batch_size, validation_split=0.2, verbose=1)
        st.text(history.history)

    # Évaluation du modèle
    _, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    st.write(f"Test accuracy: {test_accuracy:.4f}")

    st.write("Attendre pour voir Training and Validation Accuracy & Training and Validation Loss")

    # Entraînement du modèle et récupération de l'historique
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Tracer la courbe de précision avec Plotly
    fig = px.line(history.history, y=['accuracy', 'val_accuracy'], labels={'value': 'Accuracy', 'variable': 'Dataset', 'index': 'Epochs'})
    fig.update_layout(title='Training and Validation Accuracy', xaxis_title='Epochs')
    st.plotly_chart(fig)

    # Tracer la courbe de perte avec Plotly
    fig = px.line(history.history, y=['loss', 'val_loss'], labels={'value': 'Loss', 'variable': 'Dataset', 'index': 'Epochs'})
    fig.update_layout(title='Training and Validation Loss', xaxis_title='Epochs')
    st.plotly_chart(fig)
