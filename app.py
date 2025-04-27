import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# --- Streamlit App ---
def inject_f1_style():
    st.markdown('''
        <style>
            body {
                background-image: url('https://images.unsplash.com/photo-1625658201247-3b65d7fae6bb?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80');
                background-size: cover;
                background-attachment: fixed;
                background-repeat: no-repeat;
                background-position: center;
                color: #FFFFFF;
            }
            .stApp {
                background-color: rgba(0, 0, 0, 0.85);
                padding: 2rem;
                border-radius: 12px;
            }
            h1, h2, h3, h4 {color: #E10600;}
            .css-1d391kg, .css-1kyxreq {background-color: #1C1C1C; color: white; border-radius: 10px;}
            .stButton>button {background-color: #E10600; color: white; border: none; padding: 0.5em 2em; border-radius: 8px; font-weight: bold;}
            .stButton>button:hover {background-color: #B30000;}
            .stSidebar {background-color: rgba(0, 0, 0, 0.85);}
        </style>
    ''', unsafe_allow_html=True)

inject_f1_style()

# Banner
st.markdown("""
    <div style='text-align: center;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/F1.svg/1200px-F1.svg.png' width='300'/>
        <h1 style='color: #E10600;'>Bienvenido al F1 Race Winner Predictor</h1>
    </div>
    <br>
""", unsafe_allow_html=True)

st.sidebar.header("Detalles de la Carrera")
year = st.sidebar.number_input("Año", min_value=1950, max_value=2025, value=2025)
round_number = st.sidebar.number_input("Ronda", min_value=1, max_value=25, value=3)

st.title('Predicción de Ganadores de F1')
st.write('Modelo de Machine Learning para predecir cuántas carreras ganará un piloto.')

# --- Cargar datos de ejemplo ---
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

# --- Preprocesamiento ---
le_piloto = LabelEncoder()
le_equipo = LabelEncoder()

data['Piloto_encoded'] = le_piloto.fit_transform(data['Piloto'])
data['Equipo_encoded'] = le_equipo.fit_transform(data['Equipo'])

X = data[['Piloto_encoded', 'Equipo_encoded', 'Clasificación']]
y = data['Carreras Ganadas']

# --- Entrenamiento del Modelo ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Evaluación ---
y_pred = model.predict(X_test)

st.subheader('Evaluación del modelo')
st.write('Error cuadrático medio (MSE):', mean_squared_error(y_test, y_pred))
st.write(f"Precisión aproximada (simulada): `{100 - (mean_squared_error(y_test, y_pred)):.2f}%`")

# --- Predicción ---
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
    st.success(f'Predicción de carreras ganadas: {prediccion[0]:.2f}')

# --- Sonido opcional ---
engine_sound_url = "https://www.soundjay.com/mechanical/sounds/race-car-engine-01.mp3"

if st.sidebar.button("¡Escuchar motor de F1!"):
    st.markdown(f"""
        <audio autoplay>
            <source src="{engine_sound_url}" type="audio/mpeg">
        </audio>
    """, unsafe_allow_html=True)
