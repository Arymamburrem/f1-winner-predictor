import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import plotly.express as px

# --- CONFIGURACIÓN GENERAL ---
st.set_page_config(page_title="F1 Predictor 2025 por AryMamburrem", layout="wide")

# --- ESTILO Y FONDO PERSONALIZADO ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto&display=swap');

        html, body, .stApp {
            font-family: 'Roboto', sans-serif;
            background: linear-gradient(to bottom, #000000cc, #000000cc),
                        url("https://images.pexels.com/photos/2076249/pexels-photo-2076249.jpeg") no-repeat center center fixed;
            background-size: cover;
            color: white;
        }
        h1, h2, h3, h4 {
            color: #E10600;
        }
        .stButton>button {
            background-color: #E10600;
            color: white;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #990000;
        }
        .next-race-box {
            background: rgba(0, 0, 0, 0.5);
            border-left: 5px solid #E10600;
            padding: 1rem;
            margin-bottom: 2rem;
            border-radius: 8px;
        }
        .logo-container {
            display: flex;
            justify-content: center;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- LOGO GRANDE Y CENTRADO ---
st.markdown('<div class="logo-container">', unsafe_allow_html=True)
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/F1.svg/1200px-F1.svg.png", width=400)
st.markdown('</div>', unsafe_allow_html=True)

st.title("🏎️ F1 Race Predictor 2025")

# --- CALENDARIO DE CARRERAS 2025 ---
calendario_2025 = [
    {"nombre": "GP de Bahréin", "circuito": "Sakhir", "fecha": "2025-03-14", "pais": "🇧🇭"},
    {"nombre": "GP de Arabia Saudita", "circuito": "Jeddah", "fecha": "2025-03-21", "pais": "🇸🇦"},
    {"nombre": "GP de Australia", "circuito": "Albert Park", "fecha": "2025-04-06", "pais": "🇦🇺"},
    {"nombre": "GP de Japón", "circuito": "Suzuka", "fecha": "2025-04-13", "pais": "🇯🇵"},
    {"nombre": "GP de China", "circuito": "Shanghai", "fecha": "2025-04-20", "pais": "🇨🇳"},
    {"nombre": "GP de Miami", "circuito": "Miami International Autodrome", "fecha": "2025-05-04", "pais": "🇺🇸"},
    {"nombre": "GP de Emilia-Romaña", "circuito": "Imola", "fecha": "2025-05-18", "pais": "🇮🇹"},
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
    st.markdown(f"### 🏁 Próxima Carrera de F1")
    st.markdown(f"**{proxima['nombre']}** {proxima['pais']}")
    st.markdown(f"📍 Circuito: *{proxima['circuito']}*")
    st.markdown(f"📆 Fecha: *{proxima['fecha']}*")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.warning("No hay más carreras registradas en el calendario 2025.")

@st.cache_data
def cargar_datos():
    url = "https://api.jolpi.ca/ergast/f1/2025/results.json?limit=1000"
    r = requests.get(url)
    if r.status_code != 200:
        st.error("Error al acceder a la API Jolpica.")
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
        st.error("No se encontraron datos válidos para la temporada 2025.")
        return df
    df['win'] = (df['position'] == 1).astype(int)
    return df

data = cargar_datos()
if data.empty:
    st.stop()

st.subheader("📊 Datos Reales Temporada 2025")
st.dataframe(data.head(10))

st.subheader("🏁 Victorias por Piloto (Interactivo)")
wins = data[data['win'] == 1].groupby('driver').size().sort_values(ascending=False)
fig = px.bar(wins.reset_index(), x=0, y='driver', orientation='h',
             color=0, color_continuous_scale='reds', labels={0: 'Victorias', 'driver': 'Piloto'})
fig.update_layout(template="plotly_dark")
st.plotly_chart(fig)

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

st.markdown(f"### 🎯 Precisión del modelo: `{accuracy_score(y_test, y_pred):.2f}`")

st.sidebar.header("🔮 Predicción Personalizada")
pilotos = list(le_driver.classes_)
equipos = list(le_team.classes_)
piloto_sel = st.sidebar.selectbox("Piloto", pilotos)
equipo_sel = st.sidebar.selectbox("Equipo", equipos)
grid_sel = st.sidebar.slider("Posición de largada (Grid)", 1, 20, 5)

datos_input = np.array([
    le_driver.transform([piloto_sel])[0],
    le_team.transform([equipo_sel])[0],
    grid_sel
]).reshape(1, -1)
prediccion = model.predict(datos_input)
proba = model.predict_proba(datos_input)[0][1]

if st.sidebar.button("Predecir Ganador"):
    if prediccion[0] == 1:
        st.success(f"🧠 Según el modelo, {piloto_sel} GANARÁ la carrera. (Probabilidad: {proba:.2%})")
    else:
        st.info(f"🧠 Según el modelo, {piloto_sel} NO ganará. (Probabilidad: {proba:.2%})")

    if st.sidebar.button("Guardar Predicción"):
        pred_df = pd.DataFrame([{
            'Piloto': piloto_sel,
            'Equipo': equipo_sel,
            'Grid': grid_sel,
            'Probabilidad de victoria': f"{proba:.2%}"
        }])
        st.download_button("Descargar Resultado", pred_df.to_csv(index=False), file_name="prediccion_f1.csv", mime="text/csv")































