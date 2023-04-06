import streamlit as st
import pandas as pd
import streamlit as st
import pandas as pd
from joblib import load
from sklearn.inspection import partial_dependence
import datetime
import plotly.express as px
from Home import load_data

data = load_data()

st.header('Data shape')
st.write(data.shape)

st.header('Data types')
st.write(data.dtypes)

st.header('Sommaires statistiques')
st.write(data.describe())

st.header('Valeurs manquantes')
st.write(data.isnull().sum())