# f1_predictor_app.py
import pandas as pd
import numpy as np
import streamlit as st
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- CONFIGURACIÓN GENERAL ---
st.set_page_config(page_title="F1 Predictor 2025", layout="wide")

# --- ESTILO PERSONALIZADO Y LOGO ---
st.markdown("""
    <style>
        body {background-color: #0d0d0d; color: white;}
        .main {background-color: #0d0d0d;}
        h1, h2, h3, h4 {color: #E10600;}
        .stButton>button {background-color: #E10600; color: white;}
        .stButton>button:hover {background-color: #990000;}
        .next-race-box {
            background: rgba(255, 255, 255, 0.05);
            border-left: 5px solid #E10600;
            padding: 1rem;
            margin-bottom: 2rem;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/F1.svg/800px-F1.svg.png", width=150)
st.title("🏎️ F1 Race Predictor 2025")

# --- CALENDARIO 2025 ---
calendario_2025 = [
    {"nombre": "GP de Bahréin", "circuito": "Sakhir", "fecha": "2025-03-14"},
    {"nombre": "GP de Arabia Saudita", "circuito": "Jeddah", "fecha": "2025-03-21"},
    {"nombre": "GP de Australia", "circuito": "Albert Park", "fecha": "2025-04-06"},
    {"nombre": "GP de Japón", "circuito": "Suzuka", "fecha": "2025-04-13"},
    {"nombre": "GP de China", "circuito": "Shanghai", "fecha": "2025-04-20"},
    {"nombre": "GP de Miami", "circuito": "Miami International Autodrome", "fecha": "2025-05-04"},
    {"nombre": "GP de Emilia-Romaña", "circuito": "Imola", "fecha": "2025-05-18"},
]

def obtener_proxima_carrera():
    hoy = datetime.now().date()
    for carrera in calendario_2025:
        fecha = datetime.strptime(carrera["fecha"], "%Y-%m-%d").date()
        if fecha >= hoy:
            return carrera
    return None

proxima = obtener_proxima_carrera()

if proxima:
    st.markdown('<div class="next-race-box">', unsafe_allow_html=True)
    st.markdown("### 🏁 Próxima Carrera")
    st.markdown(f"**{proxima['nombre']}**")
    st.markdown(f"📍 Circuito: *{proxima['circuito']}*")
    st.markdown(f"📆 Fecha: *{proxima['fecha']}*")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.warning("No hay más carreras registradas en el calendario 2025.")

# --- CARGAR DATOS DESDE ERGAST API ---
@st.cache_data
def cargar_datos():
    url = "https://ergast.com/api/f1/2023/results.json?limit=1000"
    r = requests.get(url)
    if r.status_code != 200:
        st.error("Error al acceder a la API.")
        return pd.DataFrame()
    json_data = r.json()
    races = json_data.get('MRData', {}).get('RaceTable', {}).get('Races', [])
    registros = []
    for carrera in races:
        for resultado in carrera.get('Results', []):
            position = resultado.get('position')
            if position and position.isdigit():
                registros.append({
                    'raceName': carrera.get('raceName'),
                    'date': carrera.get('date'),
                    'circuit': carrera.get('Circuit', {}).get('circuitName'),
                    'driver': resultado.get('Driver', {}).get('familyName'),
                    'constructor': resultado.get('Constructor', {}).get('name'),
                    'grid': int(resultado.get('grid', 0)),
                    'position': int(position),
                    'status': resultado.get('status')
                })
    df = pd.DataFrame(registros)
    if df.empty:
        st.error("No se encontraron datos válidos.")
        return df
    df['win'] = (df['position'] == 1).astype(int)
    return df

data = cargar_datos()

if data.empty:
    st.stop()

# --- DATOS VISUALES ---
st.subheader("📊 Datos Temporada 2023")
st.dataframe(data.head(10))

st.subheader("🏆 Victorias por Piloto")
wins = data[data['win'] == 1].groupby('driver').size().sort_values(ascending=False)
fig, ax = plt.subplots()
sns.barplot(x=wins.values, y=wins.index, ax=ax, palette="Reds_r")
ax.set_xlabel("Victorias")
ax.set_ylabel("Piloto")
st.pyplot(fig)

# --- ENTRENAMIENTO DEL MODELO ---
le_driver = LabelEncoder()
le_team = LabelEncoder()
data['driver_enc'] = le_driver.fit_transform(data['driver'])
data['team_enc'] = le_team.fit_transform(data['constructor'])

X = data[['driver_enc', 'team_enc', 'grid']]
y = data['win']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.markdown(f"### 🎯 Precisión del Modelo: `{accuracy_score(y_test, y_pred):.2f}`")

# --- FORMULARIO DE PREDICCIÓN ---
st.sidebar.header("🔮 Predicción Personalizada")
pilotos = list(le_driver.classes_)
equipos = list(le_team.classes_)
piloto_sel = st.sidebar.selectbox("Piloto", pilotos)
equipo_sel = st.sidebar.selectbox("Equipo", equipos)
grid_sel = st.sidebar.slider("Posición de largada", 1, 20, 5)

if st.sidebar.button("Predecir Ganador"):
    entrada = np.array([
        le_driver.transform([piloto_sel])[0],
        le_team.transform([equipo_sel])[0],
        grid_sel
    ]).reshape(1, -1)
    prediccion = model.predict(entrada)
    resultado = "GANARÁ la carrera" if prediccion[0] == 1 else "NO ganará"
    st.success(f"🧠 Según el modelo, {piloto_sel} {resultado}.")




























