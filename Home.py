import streamlit as st
import pandas as pd
import streamlit as st
import pandas as pd
from joblib import load
from sklearn.inspection import partial_dependence
import datetime
import plotly.express as px

st.title('Airline Satisfaction')

st.image("pic/united.jpg")
def load_data():
    data = pd.read_csv('Airline Passenger Satisfaction.csv', delimiter=';')
    return data

data = load_data()

# Afficher le DataFrame dans Streamlit
st.dataframe(data)