import streamlit as st

st.header("Chi2")
st.code("""import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import OneHotEncoder

url = "https://raw.githubusercontent.com/remijul/dataset/master/Airline%20Passenger%20Satisfaction.csv"
df = pd.read_csv(url, delimiter=';')

cat_vars = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
onehot = OneHotEncoder(sparse=False)
cat_data = onehot.fit_transform(df[cat_vars])
cat_df = pd.DataFrame(cat_data, columns=onehot.get_feature_names_out(cat_vars))

# Convertion de la variable cible au format binaire
df['Satisfaction_binary'] = (df['Satisfaction'] == 'satisfied').astype(int)

# Pour obtenir les noms de variables encodés
encoded_vars = onehot.get_feature_names_out(cat_vars)

# Test du chi2 pour vérifier l'indépendance entre les variables catégorielles et la variable cible
for var in cat_vars:
    encoded_vars_for_var = [v for v in encoded_vars if v.startswith(var)]
    for encoded_var in encoded_vars_for_var:
        table = pd.crosstab(cat_df[encoded_var], df['Satisfaction_binary'])
        _, p, _, _ = stats.chi2_contingency(table)
        print(f"Chi-squared test for independence between {encoded_var} and Satisfaction:\n"
            f"\tP-value: {p:.4f}")""")

st.text("""Chi-squared test for independence between Gender_Female and Satisfaction:
	P-value: 0.0000
Chi-squared test for independence between Gender_Male and Satisfaction:
	P-value: 0.0000
Chi-squared test for independence between Customer Type_Loyal Customer and Satisfaction:
	P-value: 0.0000
Chi-squared test for independence between Customer Type_disloyal Customer and Satisfaction:
	P-value: 0.0000
Chi-squared test for independence between Type of Travel_Business travel and Satisfaction:
	P-value: 0.0000
Chi-squared test for independence between Type of Travel_Personal Travel and Satisfaction:
	P-value: 0.0000
Chi-squared test for independence between Class_Business and Satisfaction:
	P-value: 0.0000
Chi-squared test for independence between Class_Eco and Satisfaction:
	P-value: 0.0000
Chi-squared test for independence between Class_Eco Plus and Satisfaction:
	P-value: 0.0000""")