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

st.header("Comptage Satisfaction")
fig = px.histogram(data_frame=data, x='Satisfaction', color='Satisfaction', nbins=10)
st.plotly_chart(fig)

fig = px.box(data_frame=data, x="Satisfaction", y="Age", color="Satisfaction")

st.header("Distribution de l'âge des passagers en fonction du niveau de satisfaction")
fig.update_layout(
                xaxis_title="Niveau de satisfaction",
                yaxis_title="Age")

st.plotly_chart(fig)

st.header("Distribution de l'âge")
fig = px.histogram(data_frame=data, x="Age", nbins=20)
fig.update_layout(
                xaxis_title="Age",
                yaxis_title="Count")

st.plotly_chart(fig)

st.header("Distance de vol vs Retard à l'arrivée")
fig = px.scatter(data_frame=data, x="Flight Distance", y="Arrival Delay in Minutes")

fig.update_layout(
                xaxis_title="Distance de vol",
                yaxis_title="Retard à l'arrivée (minutes)")

st.plotly_chart(fig)