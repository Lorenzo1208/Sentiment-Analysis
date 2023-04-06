import streamlit as st

st.header("Anova")
st.code("""import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

url = "https://raw.githubusercontent.com/remijul/dataset/master/Airline%20Passenger%20Satisfaction.csv"
df = pd.read_csv(url, delimiter=';')

# Convertion des variables catégorielles en variables numériques à l'aide d'un one-hot encoding
cat_vars = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
onehot = OneHotEncoder(sparse=False)
cat_data = onehot.fit_transform(df[cat_vars])
cat_df = pd.DataFrame(cat_data, columns=onehot.get_feature_names_out(cat_vars))

# Join des données numériques et catégorielles
num_vars = ['Age', 'Flight Distance', 'Seat comfort', 'Departure/Arrival time convenient',
            'Food and drink', 'Gate location', 'Inflight wifi service', 'Inflight entertainment',
            'Online support', 'Ease of Online booking', 'On-board service', 'Leg room service',
            'Baggage handling', 'Checkin service', 'Cleanliness', 'Online boarding',
            'Departure Delay in Minutes', 'Arrival Delay in Minutes']

df_processed = pd.concat([df[num_vars], cat_df, df['Satisfaction']], axis=1)

# Remplacement des espaces par des traits de soulignement dans les noms de colonnes
df_processed.columns = df_processed.columns.str.replace(' ', '_')

# Convertion de satisfaction en variable binaire
df_processed['Satisfaction'] = df_processed['Satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})

# Fit du modèle avec ANOVA
model = ols('Satisfaction ~ Age + Gender_Female + Gender_Male + Type_of_Travel_Business_travel + Type_of_Travel_Personal_Travel + Class_Business + Class_Eco + Class_Eco_Plus + Flight_Distance + Seat_comfort', data=df_processed)
entrainement = model.fit()
anova_results = anova_lm(entrainement)
anova_results""")

st.text("""	df	sum_sq	mean_sq	F	PR(>F)
Age	1.0	447.844322	447.844322	2297.071243	0.000000e+00
Gender_Female	1.0	1464.128798	1464.128798	7509.770680	0.000000e+00
Gender_Male	1.0	0.029611	0.029611	0.151878	6.967477e-01
Type_of_Travel_Business_travel	1.0	334.468588	334.468588	1715.547431	0.000000e+00
Type_of_Travel_Personal_Travel	1.0	0.087643	0.087643	0.449537	5.025559e-01
Class_Business	1.0	2749.254370	2749.254370	14101.402753	0.000000e+00
Class_Eco	1.0	9.311677	9.311677	47.761207	4.836307e-12
Class_Eco_Plus	1.0	0.043316	0.043316	0.222177	6.373870e-01
Flight_Distance	1.0	62.370654	62.370654	319.909906	1.847045e-71
Seat_comfort	1.0	1792.009374	1792.009374	9191.527052	0.000000e+00
Residual	129872.0	25320.258548	0.194963	NaN	NaN""")