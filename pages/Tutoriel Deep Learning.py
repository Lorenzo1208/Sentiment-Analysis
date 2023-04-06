import streamlit as st

code1='''
pip install tensorflow
'''

st.header("Deep Learning")
st.code(code1, language='python')

code2='''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

# Création du modèle
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilation du modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Évaluation du modèle
_, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")
'''

st.code(code2, language='python')
st.text("""Epoch 1/10
2598/2598 [==============================] - 5s 2ms/step - loss: 0.2437 - accuracy: 0.8979 - val_loss: 0.1798 - val_accuracy: 0.9261
Epoch 2/10
2598/2598 [==============================] - 4s 1ms/step - loss: 0.1690 - accuracy: 0.9297 - val_loss: 0.1613 - val_accuracy: 0.9332
Epoch 3/10
2598/2598 [==============================] - 3s 1ms/step - loss: 0.1516 - accuracy: 0.9365 - val_loss: 0.1464 - val_accuracy: 0.9376
Epoch 4/10
2598/2598 [==============================] - 3s 1ms/step - loss: 0.1409 - accuracy: 0.9403 - val_loss: 0.1389 - val_accuracy: 0.9404
Epoch 5/10
2598/2598 [==============================] - 3s 1ms/step - loss: 0.1345 - accuracy: 0.9424 - val_loss: 0.1344 - val_accuracy: 0.9414
Epoch 6/10
2598/2598 [==============================] - 3s 1ms/step - loss: 0.1295 - accuracy: 0.9447 - val_loss: 0.1342 - val_accuracy: 0.9415
Epoch 7/10
2598/2598 [==============================] - 3s 1ms/step - loss: 0.1262 - accuracy: 0.9459 - val_loss: 0.1295 - val_accuracy: 0.9442
Epoch 8/10
2598/2598 [==============================] - 3s 1ms/step - loss: 0.1235 - accuracy: 0.9472 - val_loss: 0.1264 - val_accuracy: 0.9456
Epoch 9/10
2598/2598 [==============================] - 3s 1ms/step - loss: 0.1212 - accuracy: 0.9477 - val_loss: 0.1238 - val_accuracy: 0.9479
Epoch 10/10
2598/2598 [==============================] - 3s 1ms/step - loss: 0.1191 - accuracy: 0.9490 - val_loss: 0.1247 - val_accuracy: 0.9462
Test accuracy: 0.9484""")

st.header("Architecture du réseau de neurones")
st.code("""model.summary()""")

st.text("""Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 32)                768       
                                                                 
 dense_1 (Dense)             (None, 16)                528       
                                                                 
 dense_2 (Dense)             (None, 1)                 17        
                                                                 
=================================================================
Total params: 1,313
Trainable params: 1,313
Non-trainable params: 0
_________________________________________________________________""")

st.image("output5.png")