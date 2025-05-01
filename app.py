import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURACIÓN GENERAL ---
st.set_page_config(page_title="F1 Predictor 2025 por AryMamburrem", layout="wide")

# --- ESTILO PERSONALIZADO Y FONDO SUTIL ---
st.markdown("""
    <style>
        .stApp {
            background-image: url("https://img.redbull.com/images/w_3000/q_auto,f_auto/redbullcom/2013/09/22/1331612504617_5/gran-premio-de-f%C3%B3rmula-1-de-singapur-2013.jpg");
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
            background: rgba(0, 0, 0, 0.6);
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

# --- LOGO CENTRADO ---
st.markdown('<div class="logo-container">', unsafe_allow_html=True)
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/F1.svg/1200px-F1.svg.png", width=400)
st.markdown('</div>', unsafe_allow_html=True)

st.title("🏎️ F1 Race Predictor 2025")

# --- CALENDARIO DE CARRERAS ---
calendario_2025 = [
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

# --- PRÓXIMA CARRERA ---
proxima = obtener_proxima_carrera()
if proxima:
    st.markdown('<div class="next-race-box">', unsafe_allow_html=True)
    st.markdown(f"### 🏁 Próxima Carrera")
    st.markdown(f"**{proxima['nombre']}** {proxima['pais']}")
    st.markdown(f"📍 Circuito: *{proxima['circuito']}*")
    st.markdown(f"📆 Fecha: *{proxima['fecha']}*")
    st.markdown("🌦️ **Clima estimado**: Soleado, 28°C")
    st.markdown("📈 **Promedio de victorias en este circuito**: Verstappen (3), Hamilton (2)")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.warning("No hay más carreras registradas en el calendario 2025.")

# --- CARGA DE DATOS DE CARRERAS ---
@st.cache_data
def cargar_datos():
    url = "https://api.jolpi.ca/ergast/f1/2025/results.json?limit=1000"
    r = requests.get(url)
    if r.status_code != 200:
        st.error("Error al acceder a la API Jolpica.")
        return pd.DataFrame()
    races = r.json().get('MRData', {}).get('RaceTable', {}).get('Races', [])
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
    df['top3'] = (df['position'] <= 3).astype(int)
    return df

# --- PROCESO DE DATOS Y VISUALIZACIÓN ---
data = cargar_datos()
if data.empty:
    st.stop()

st.subheader("📊 Resultados Temporada 2025")
st.dataframe(data.head(10))

st.subheader("🏆 Pilotos con más victorias")
wins = data[data['win'] == 1].groupby('driver').size().sort_values(ascending=False)
fig, ax = plt.subplots()
sns.barplot(x=wins.values, y=wins.index, ax=ax, palette="Reds_r")
ax.set_xlabel("Victorias")
ax.set_ylabel("Piloto")
st.pyplot(fig)

st.subheader("🏁 Comparativa entre Escuderías")
comparativa = data.groupby('constructor').agg(
    total_carreras=('raceName', 'count'),
    victorias=('win', 'sum'),
    promedio_grid=('grid', 'mean')
).sort_values(by='victorias', ascending=False)

st.dataframe(comparativa)

fig_eq, ax_eq = plt.subplots()
sns.barplot(x=comparativa['victorias'], y=comparativa.index, palette='Reds_r', ax=ax_eq)
ax_eq.set_title("Victorias por Escudería")
ax_eq.set_xlabel("Victorias")
st.pyplot(fig_eq)

# --- ENTRENAMIENTO DE MODELOS ---
le_driver = LabelEncoder()
le_team = LabelEncoder()
data['driver_enc'] = le_driver.fit_transform(data['driver'])
data['team_enc'] = le_team.fit_transform(data['constructor'])

X = data[['driver_enc', 'team_enc', 'grid']]
y = data['win']
y_top3 = data['top3']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y_top3, test_size=0.2, random_state=42)
model_top3 = RandomForestClassifier(n_estimators=100, random_state=42)
model_top3.fit(X_train3, y_train3)

st.markdown(f"### 🎯 Precisión del modelo: `{accuracy_score(y_test, y_pred):.2f}`")

# --- FORMULARIO DE PREDICCIÓN ---
st.sidebar.header("🔮 Predicción Personalizada")
pilotos = list(le_driver.classes_)
equipos = list(le_team.classes_)

piloto_sel = st.sidebar.selectbox("Piloto", pilotos)
equipo_sel = st.sidebar.selectbox("Equipo", equipos)
grid_sel = st.sidebar.slider("Posición de largada (Grid)", 1, 20, 5)

if st.sidebar.button("Predecir Ganador y Podio"):
    datos_input = np.array([
        le_driver.transform([piloto_sel])[0],
        le_team.transform([equipo_sel])[0],
        grid_sel
    ]).reshape(1, -1)

    pred1 = model.predict(datos_input)
    pred3 = model_top3.predict(datos_input)

    resultado1 = "GANARÁ" if pred1[0] == 1 else "NO ganará"
    resultado3 = "subirá al PODIO" if pred3[0] == 1 else "no subirá al podio"

    st.success(f"🧠 Según el modelo, {piloto_sel} {resultado1} y {resultado3}.")

































