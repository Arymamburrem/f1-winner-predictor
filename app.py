import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Estilo personalizado F1 ---
def inject_f1_style():
    st.markdown('''
        <style>
            body {
                background-image: url('https://images.unsplash.com/photo-1504384308090-c894fdcc538d');
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
            .stButton>button {background-color: #E10600; color: white; border: none; padding: 0.5em 2em; border-radius: 8px; font-weight: bold;}
            .stButton>button:hover {background-color: #B30000;}
            .stSidebar {background-color: rgba(0, 0, 0, 0.85);}
        </style>
    ''', unsafe_allow_html=True)

inject_f1_style()

# --- Banner principal ---
st.markdown("""
    <div style='text-align: center;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/F1.svg/1200px-F1.svg.png' width='300'/>
        <h1 style='color: #E10600;'>F1 Race Winner Predictor</h1>
    </div>
    <br>
""", unsafe_allow_html=True)

# --- Datos de ejemplo simulados ---
@st.cache_data
def load_data():
    # Simulación de datos de carreras de F1
    data = {
        'Driver': ['Verstappen', 'Hamilton', 'Leclerc', 'Alonso', 'Norris', 'Sainz', 'Russell', 'Perez', 'Ricciardo', 'Zhou'],
        'Constructor': ['Red Bull', 'Mercedes', 'Ferrari', 'Aston Martin', 'McLaren', 'Ferrari', 'Mercedes', 'Red Bull', 'Alfa Romeo', 'Alfa Romeo'],
        'Grid': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Position': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Points': [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
    }
    df = pd.DataFrame(data)
    df['Wins'] = (df['Position'] == 1).astype(int)  # Nueva columna: 1 si ganó
    return df

data = load_data()

# --- Gráfico de Barras ---
st.subheader('Victorias por Piloto')
wins_by_driver = data.groupby('Driver')['Wins'].sum().sort_values(ascending=False)
fig, ax = plt.subplots()
sns.barplot(x=wins_by_driver.values, y=wins_by_driver.index, palette='Reds_r', ax=ax)
ax.set_xlabel('Cantidad de Victorias')
ax.set_ylabel('Piloto')
st.pyplot(fig)

# --- Preprocesamiento ---
le_driver = LabelEncoder()
le_team = LabelEncoder()

data['Driver_encoded'] = le_driver.fit_transform(data['Driver'])
data['Constructor_encoded'] = le_team.fit_transform(data['Constructor'])

X = data[['Driver_encoded', 'Constructor_encoded', 'Grid']]
y = data['Wins']  # Variable objetivo: 1 si ganó

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_basic = RandomForestClassifier(n_estimators=100, random_state=42)
model_basic.fit(X_train, y_train)

# --- Sidebar - Entrada de datos Básica ---
st.sidebar.header("Predicción Básica")
piloto = st.sidebar.selectbox('Selecciona un Piloto', data['Driver'].unique())
equipo = st.sidebar.selectbox('Selecciona un Equipo', data['Constructor'].unique())
clasificacion = st.sidebar.slider('Clasificación en Qualy', 1, 20, 1)

if st.sidebar.button('Predecir Carreras Ganadas'):
    input_data = np.array([
        le_driver.transform([piloto])[0],
        le_team.transform([equipo])[0],
        clasificacion
    ]).reshape(1, -1)
    prediccion = model_basic.predict(input_data)

    st.success(f'Predicción de carreras ganadas: {prediccion[0]}')

    # Sonido Motor
    engine_sound_url = "https://www.soundjay.com/mechanical/sounds/race-car-engine-01.mp3"
    st.markdown(f"""
        <audio autoplay>
            <source src="{engine_sound_url}" type="audio/mpeg">
        </audio>
    """, unsafe_allow_html=True)

# --- Mostrar precisión de los modelos ---
st.markdown(f"### R² del Modelo Básico: `{r2_score(y_test, model_basic.predict(X_test)):.2f}`")















