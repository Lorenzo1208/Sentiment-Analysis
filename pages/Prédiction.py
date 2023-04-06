import joblib
import streamlit as st
import pandas as pd
from joblib import load
from sklearn.inspection import partial_dependence
import datetime
import plotly.express as px
from Home import load_data
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

import streamlit as st
import joblib
import pandas as pd

# Charger le modèle enregistré
best_model = joblib.load('best_model.joblib')

# Créer les champs d'entrée pour l'application Streamlit
input_fields = {
    'Gender': st.selectbox('Gender', ['Female', 'Male']),
    'Customer Type': st.selectbox('Customer Type', ['Loyal Customer', 'disloyal Customer']),
    'Age': st.number_input('Age', value=30, step=1),
    'Type of Travel': st.selectbox('Type of Travel', ['Business travel', 'Personal Travel']),
    'Class': st.selectbox('Class', ['Business', 'Eco', 'Eco Plus']),
    'Flight Distance': st.number_input('Flight Distance', value=1000, step=100),
    'Seat comfort': st.number_input('Seat comfort', value=1, step=1, min_value=0, max_value=5),
    'Departure/Arrival time convenient': st.number_input('Departure/Arrival time convenient', value=1, step=1, min_value=0, max_value=5),
    'Food and drink': st.number_input('Food and drink', value=1, step=1, min_value=0, max_value=5),
    'Gate location': st.number_input('Gate location', value=1, step=1, min_value=0, max_value=5),
    'Inflight wifi service': st.number_input('Inflight wifi service', value=1, step=1, min_value=0, max_value=5),
    'Inflight entertainment': st.number_input('Inflight entertainment', value=1, step=1, min_value=0, max_value=5),
    'Online support': st.number_input('Online support', value=1, step=1, min_value=0, max_value=5),
    'Ease of Online booking': st.number_input('Ease of Online booking', value=1, step=1, min_value=0, max_value=5),
    'On-board service': st.number_input('On-board service', value=1, step=1, min_value=0, max_value=5),
    'Leg room service': st.number_input('Leg room service', value=1, step=1, min_value=0, max_value=5),
    'Baggage handling': st.number_input('Baggage handling', value=1, step=1, min_value=0, max_value=5),
    'Checkin service': st.number_input('Checkin service', value=1, step=1, min_value=0, max_value=5),
    'Cleanliness': st.number_input('Cleanliness', value=1, step=1, min_value=0, max_value=5),
    'Online boarding': st.number_input('Online boarding', value=1, step=1, min_value=0, max_value=5),
    'Departure Delay in Minutes': st.number_input('Departure Delay in Minutes', value=0, step=1),
    'Arrival Delay in Minutes': st.number_input('Arrival Delay in Minutes', value=0, step=1)
}


categorical_col = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

# Créer une fonction pour prédire les sorties à partir des entrées et du modèle
def predict_output(input_data):
    input_df = pd.DataFrame([input_data])
    input_df[categorical_col] = input_df[categorical_col].astype('category')
    prediction = best_model.predict(input_df)
    return prediction[0]

# Créer un bouton pour déclencher la prédiction
if st.button('Prédire si le client sera satisfait'):
    prediction = predict_output(input_fields)
    st.write('Prédiction:', prediction)
