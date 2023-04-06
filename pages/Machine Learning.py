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

# Charger le modèle entraîné
best_model = joblib.load('best_model.joblib')

