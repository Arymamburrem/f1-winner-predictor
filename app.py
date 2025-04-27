import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Estilo personalizado ---
def inject_custom_style():
    st.markdown(f'''
        <style>
            body {{
                background-image: url('https://raw.githubusercontent.com/tu_usuario/tu_repositorio/main/f1%20image.jpeg');
                background-size: cover;
                background-attachment: fixed;
                background-position: center;
                background-repeat: no-repeat;
                color: white;
            }}
            .stApp {{
                background-color: rgba(0, 0, 0, 0.85);
                padding: 2rem;
                border-radius: 15px;
            }}
            h1, h2, h3, h4 {{
                color: #E10600;
            }}
            .stButton>button {{
                background-color: #E10600;
                color: white;
                font-weight: bold;
                border: none;
                padding: 0.5em 2em;
                border-radius: 8px;
            }}
            .stButton>button:hover {{
                background-color: #B30000;
            }}
            .stSidebar {{
                background-color: rgba(0, 0, 0, 0.9);
            }}
        </style>
    ''', unsafe_allow_html=True)

inject_custom_style()

# --- Banner superior ---
st.markdown("""
    <div style='text-align: center;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/F1.svg/1200px-F1.svg.png' width='300'/>
        <h1>🏎️ F1 Race Winner Predictor</h1>
    </div>
""", unsafe_allow_html=True)

st.sidebar.header("Input Race Details")

year = st.sidebar.number_input("Year", min_value=1950, max_value=2025, value=2025)
round_number = st.sidebar.number_input("Round", min_value=1, max_value=25, value=3)

st.title('Predicción de Ganadores de F1')
st.write('Modelo de Machine Learning para predecir el resultado de carreras de Fórmula 1.')

# --- Datos de ejemplo ---
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

st.subheader('Datos de Pilotos')
st.dataframe(data, use_container_width=True)

# --- Preprocesamiento ---
le_piloto = LabelEncoder()
le_equipo = LabelEncoder()

data['Piloto_encoded'] = le_piloto.fit_transform(data['Piloto'])
data['Equipo_encoded'] = le_equipo.fit_transform(data['Equipo'])

X = data[['Piloto_encoded', 'Equipo_encoded', 'Clasificación']]
y = data['Carreras Ganadas']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Modelo ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Evaluación del modelo ---
y_pred = model.predict(X_test)

st.subheader('Evaluación del modelo')
st.write('Precisión:', f"{accuracy_score(y_test, y_pred) * 100:.2f}%")

# --- Predicción de carreras ganadas ---
st.subheader('Realizar una Predicción')

piloto = st.selectbox('Selecciona un piloto', data['Piloto'].unique())
equipo = st.selectbox('Selecciona un equipo', data['Equipo'].unique())
clasificacion = st.slider('Clasificación en Qualy', 1, 20, 1)

if st.button('Predecir Ganancias'):
    input_data = np.array([
        le_piloto.transform([piloto])[0],
        le_equipo.transform([equipo])[0],
        clasificacion
    ]).reshape(1, -1)
    prediccion = model.predict(input_data)
    st.success(f'Predicción de carreras ganadas: {prediccion[0]}')

# --- Gráfico de barras carreras ganadas ---
st.subheader('Distribución de Carreras Ganadas')

fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(x='Carreras Ganadas', y='Piloto', data=data, palette='Reds_r')
ax.set_facecolor('#111111')
fig.patch.set_facecolor('#111111')
ax.tick_params(colors='white')
ax.set_xlabel('Carreras Ganadas', color='white')
ax.set_ylabel('Piloto', color='white')
ax.set_title('Comparativa de Pilotos', color='white')

st.pyplot(fig)

# --- Sonido de motor F1 ---
st.subheader('¡Arranquemos motores! 🔊')
engine_sound_url = "https://www.soundjay.com/mechanical/sounds/race-car-engine-01.mp3"

if st.button('🎵 Escuchar Sonido de Motor'):
    st.markdown(f"""
        <audio autoplay>
            <source src="{engine_sound_url}" type="audio/mpeg">
        </audio>
    """, unsafe_allow_html=True)










