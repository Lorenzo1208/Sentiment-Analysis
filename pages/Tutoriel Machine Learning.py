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

st.header("Features importances")
st.code("""import joblib

# Load le model
best_model = joblib.load('best_model.joblib')

# Récupération du modèle RandomForest dans la pipeline
rf_model = best_model.named_steps['classifier']

# Récupération des feature importances
feature_importances = rf_model.feature_importances_

# Récupération feature names depuis one-hot encoder sauf Satisfaction
feature_names = list(df_processed.drop('Satisfaction', axis=1).columns)

# Fusion des feature importances et du noms des colonnes, puis les trier par importance
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

# Affichage du résultat
print(feature_importance_df)""")

st.text("""                              Feature  Importance
7              Inflight_entertainment    0.190922
2                        Seat_comfort    0.133306
9              Ease_of_Online_booking    0.081098
8                      Online_support    0.063117
10                   On-board_service    0.040387
4                      Food_and_drink    0.039299
11                   Leg_room_service    0.033186
1                     Flight_Distance    0.031343
20       Customer_Type_Loyal_Customer    0.030109
15                    Online_boarding    0.030096
24                     Class_Business    0.029292
0                                 Age    0.028725
13                    Checkin_service    0.025955
12                   Baggage_handling    0.025133
21    Customer_Type_disloyal_Customer    0.023526
19                        Gender_Male    0.022258
18                      Gender_Female    0.021669
14                        Cleanliness    0.021363
3   Departure/Arrival_time_convenient    0.020152
5                       Gate_location    0.018869
23     Type_of_Travel_Personal_Travel    0.018609
22     Type_of_Travel_Business_travel    0.017752
17           Arrival_Delay_in_Minutes    0.015002
16         Departure_Delay_in_Minutes    0.014405
6               Inflight_wifi_service    0.013355
25                          Class_Eco    0.009110
26                     Class_Eco_Plus    0.001960""")

st.header("Matrice de confusion")

st.code("""from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.named_steps['classifier'].classes_)
disp.plot()""")

st.image("pic/output.png")

st.header("Courbe ROC")

st.code("""import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Prédiction des probabilités pour l'ensemble de test
y_prob = best_model.predict_proba(X_test)[:, 1]

# Calcul du taux de faux positifs (FPR), du taux de vrais positifs (TPR) et du score ROC AUC.
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

# Plot du ROC
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()""")

st.image("pic/output2.png")

st.header("Cross validation")

st.code("""from sklearn.model_selection import cross_val_score

scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
print("Scores cross-validation :", scores)
print("Moyenne score cross-validation :", scores.mean())
print("Écart-type des scores validés par croisement :", scores.std())""")

st.text("""Scores cross-validation : [0.95573897 0.95776619 0.95607684 0.95622164 0.95708838]
Moyenne score cross-validation : 0.956578406104658
Écart-type des scores validés par croisement : 0.0007422146085809153""")

st.header("Courbe d'apprentissage")

st.code("""from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

train_sizes, train_scores, test_scores = learning_curve(best_model, X_train, y_train, cv=5)

plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training score')
plt.plot(train_sizes, test_scores.mean(axis=1), 'o-', label='Cross-validation score')
plt.xlabel('Training set size')
plt.ylabel('Accuracy')
plt.legend()
plt.show()""")

st.image("pic/output3.png")

st.header("Prédiction sur de nouvelles données")

st.code("""import joblib
import pandas as pd

# Charger le modèle entraîné
best_model = joblib.load('best_model.joblib')

# Créer un DataFrame avec de nouvelles données
df = pd.DataFrame({
    'Gender': ['Female'],
    'Customer Type': ['Loyal Customer'],
    'Age': [30],
    'Type of Travel': ['Business travel'],
    'Class': ['Business'],
    'Flight Distance': [2000],
    'Seat comfort': [5],
    'Departure/Arrival time convenient': [5],
    'Food and drink': [4],
    'Gate location': [3],
    'Inflight wifi service': [4],
    'Inflight entertainment': [4],
    'Online support': [4],
    'Ease of Online booking': [5],
    'On-board service': [4],
    'Leg room service': [4],
    'Baggage handling': [4],
    'Checkin service': [4],
    'Cleanliness': [4],
    'Online boarding': [4],
    'Departure Delay in Minutes': [10],
    'Arrival Delay in Minutes': [20]
}, index=[0])

# Convertir les colonnes catégorielles en type 'category'
categorical_col = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
df[categorical_col] = df[categorical_col].astype('category')

# Faire des prédictions avec le modèle
predictions = best_model.predict(df)
print(predictions)""")
st.text("""[1] Ici le client sera satisfait d'après le modèle """)

st.code("""import joblib
import pandas as pd

# Charger le modèle entraîné
best_model = joblib.load('best_model.joblib')

# Créer un DataFrame avec de nouvelles données
df = pd.DataFrame({
    'Gender': ['Female'],
    'Customer Type': ['Loyal Customer'],
    'Age': [30],
    'Type of Travel': ['Business travel'],
    'Class': ['Business'],
    'Flight Distance': [2000],
    'Seat comfort': [1],
    'Departure/Arrival time convenient': [1],
    'Food and drink': [1],
    'Gate location': [1],
    'Inflight wifi service': [1],
    'Inflight entertainment': [0],
    'Online support': [1],
    'Ease of Online booking': [1],
    'On-board service': [1],
    'Leg room service': [1],
    'Baggage handling': [1],
    'Checkin service': [1],
    'Cleanliness': [1],
    'Online boarding': [1],
    'Departure Delay in Minutes': [10],
    'Arrival Delay in Minutes': [20]
}, index=[0])

# Convertir les colonnes catégorielles en type 'category'
categorical_col = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
df[categorical_col] = df[categorical_col].astype('category')

# Faire des prédictions avec le modèle
predictions = best_model.predict(df)
print(predictions)""")

st.text("""[0] Ici le client ne sera pas satisfait d'après le modèle """)

st.header("Notebook Github")
import webbrowser

if st.button('Accéder au code sur GitHub'):
    webbrowser.open_new_tab('https://github.com/Lorenzo1208/Sentiment-Analysis/blob/main/sentiment-analysis.ipynb')