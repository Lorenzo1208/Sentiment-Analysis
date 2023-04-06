import streamlit as st

code = '''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

url = "https://raw.githubusercontent.com/remijul/dataset/master/Airline%20Passenger%20Satisfaction.csv"

# Chargement des données
data = pd.read_csv(url, delimiter=';')

# Suppression des Na
data.dropna(inplace=True)

# Imputation par la médiane
# data.fillna(data.median(), inplace=True)

# Imputation par la moyenne :
# data.fillna(data.mean(), inplace=True)

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
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

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

'''

st.header("Preprocessing")
st.code(code, language='python')

code2='''
# Entraînement
pipeline.fit(X_train, y_train)
'''

st.header("Entraînement")
st.code(code2, language='python')

code3='''
# Prédiction de la satisfaction des passagers sur les données de test
y_pred = pipeline.predict(X_test)
'''

st.header("Prédiction")
st.code(code3, language='python')

code4='''
# Calcul de la précision et rapport de classification
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))
'''

st.header("Performance du random forest")
st.code(code4, language='python')

st.text('Accuracy: 0.96')

code5='''
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Liste des modèles à comparer
models = [
    ('RandomForest', RandomForestClassifier(random_state=42)),
    ('LogisticRegression', LogisticRegression(random_state=42)),
    ('SVC', SVC(random_state=42))
]

# Paramètres de recherche pour GridSearchCV pour chaque modèle
model_params = {
    'RandomForest': {
        'classifier__n_estimators': list(range(10, 61, 10)),
        'classifier__max_depth': [None,10, 20, 30]
    },
    'LogisticRegression': {
        'classifier__C': [1]
    },
    'SVC': {
        'classifier__C': [1]
    }
}

# Métrique d'erreur
scoring = 'accuracy'

# Échantillon aléatoire de 10000 observations pour entrainer les modèles
X_train_sampled, _, y_train_sampled, _ = train_test_split(X_train, y_train, train_size=10000, stratify=y_train, random_state=42)

best_score = 0
best_model = None
best_params = None

for model_name, model in models:
    print(f"Training {model_name}...")

    # Mise à jour du pipeline avec le modèle en cours
    pipeline.steps[-1] = ('classifier', model)

    # Création de GridSearchCV avec les paramètres spécifiques au modèle en cours
    grid_search = GridSearchCV(pipeline, param_grid=model_params[model_name], cv=5, scoring=scoring)

    # Exécution de GridSearchCV sur l'échantillon de données d'apprentissage
    grid_search.fit(X_train_sampled, y_train_sampled)

    # Comparaison des scores pour sélectionner le meilleur modèle
    if grid_search.best_score_ > best_score:
        best_score = grid_search.best_score_
        best_model = model_name
        best_params = grid_search.best_params_

# Affichage du meilleur modèle et de ses paramètres
print(f"Best model: {best_model} with accuracy {best_score:.2f}")
print(f"Best parameters: {best_params}")

# Mise à jour du pipeline avec le meilleur modèle et ses paramètres
pipeline.steps[-1] = ('classifier', dict(models)[best_model])
pipeline.set_params(**best_params)

# Entraînement du meilleur modèle sur l'ensemble des données d'apprentissage
pipeline.fit(X_train, y_train)

# Prédiction de la satisfaction des passagers sur les données de test
y_pred = pipeline.predict(X_test)

# Calcul de la précision et rapport de classification
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))
'''

st.header("GridSearch et autres modèles")
st.code(code5, language='python')
st.text("""Training RandomForest...
Training LogisticRegression...
Training SVC...
Best model: RandomForest with accuracy 0.94
Best parameters: {'classifier__max_depth': None, 'classifier__n_estimators': 60}
Accuracy: 0.96
            precision    recall  f1-score   support

        0       0.95      0.96      0.96     11821
        1       0.97      0.96      0.96     14077

    accuracy                           0.96     25898
macro avg       0.96      0.96      0.96     25898
weighted avg       0.96      0.96      0.96     25898""")

code6='''
import joblib

# Enregistre le meilleur modèle et la pipeline avec joblib
joblib.dump(pipeline, 'best_model.joblib')
'''
st.header("Enregistrement du meilleur modèle et de la pipeline avec joblib")
st.code(code6, language='python')

st.header("Analyse du modèle, features importances etc sur github")
import webbrowser

if st.button('Accéder au code sur GitHub'):
    webbrowser.open_new_tab('https://github.com/Lorenzo1208/Sentiment-Analysis/blob/main/sentiment-analysis.ipynb')