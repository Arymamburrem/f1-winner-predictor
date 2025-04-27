
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import random

st.title('Predicción de Ganadores de F1')
st.write('Modelo de Machine Learning para predecir el resultado de carreras de Fórmula 1.')

# Cargar datos de ejemplo (puede ser reemplazado por datos reales)
@st.cache_data
def load_data():
    data = pd.DataFrame({
        'Piloto': ['Verstappen', 'Hamilton', 'Leclerc', 'Alonso', 'Norris'],
        'Equipo': ['Red Bull', 'Mercedes', 'Ferrari', 'Aston Martin', 'McLaren'],
        'Clasificación': [1, 2, 3, 4, 5],
        'Carreras Ganadas': [10, 8, 5, 4, 2]
    })
    return data

data = load_data()

st.subheader('Datos')
st.dataframe(data)

# Preprocesamiento
le_piloto = LabelEncoder()
le_equipo = LabelEncoder()

data['Piloto_encoded'] = le_piloto.fit_transform(data['Piloto'])
data['Equipo_encoded'] = le_equipo.fit_transform(data['Equipo'])

X = data[['Piloto_encoded', 'Equipo_encoded', 'Clasificación']]
y = data['Carreras Ganadas']

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluación
y_pred = model.predict(X_test)

st.subheader('Evaluación del modelo')
st.write('Precisión:', accuracy_score(y_test, y_pred))

# Predicción
st.subheader('Hacer una predicción')

piloto = st.selectbox('Selecciona un piloto', data['Piloto'].unique())
equipo = st.selectbox('Selecciona un equipo', data['Equipo'].unique())
clasificacion = st.slider('Clasificación en Qualy', 1, 20, 1)

if st.button('Predecir'):
    input_data = np.array([
        le_piloto.transform([piloto])[0],
        le_equipo.transform([equipo])[0],
        clasificacion
    ]).reshape(1, -1)
    prediccion = model.predict(input_data)
    st.write(f'Predicción de carreras ganadas: {prediccion[0]}')
