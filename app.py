import pandas as pd
import numpy as np
import streamlit as st
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURACION GENERAL ---
st.set_page_config(page_title="F1 Race Predictor", layout="wide")
st.markdown("""
    <style>
        .main {background-color: #000000; color: white;}
        h1, h2, h3 {color: #E10600;}
        .stButton>button {background-color: #E10600; color: white;}
        .stButton>button:hover {background-color: #990000;}
    </style>
""", unsafe_allow_html=True)

st.title("🏎️ F1 Race Winner Predictor 2025")

# --- FUNCIONES DE DATOS REALES ---
@st.cache_data
def cargar_datos():
    url = "https://ergast.com/api/f1/2025/results.json?limit=1000"  # Actualizado a la temporada 2025
    r = requests.get(url)
    json_data = r.json()
    
    # Verificar si los datos se han cargado correctamente
    if "MRData" not in json_data or "RaceTable" not in json_data["MRData"]:
        st.error("Error al cargar los datos desde la API. Asegúrate de que la temporada 2025 esté disponible.")
        return pd.DataFrame()  # Retornar un DataFrame vacío si la API no responde correctamente
    
    races = json_data['MRData']['RaceTable']['Races']
    registros = []
    
    for carrera in races:
        for resultado in carrera['Results']:
            position = resultado.get('position', np.nan)
            # Si 'position' no es un valor válido, asignamos NaN
            registros.append({
                'raceName': carrera['raceName'],
                'date': carrera['date'],
                'circuit': carrera['Circuit']['circuitName'],
                'driver': resultado['Driver']['familyName'],
                'constructor': resultado['Constructor']['name'],
                'grid': int(resultado['grid']),
                'position': int(position) if str(position).isdigit() else np.nan,
                'status': resultado['status']
            })
    
    # Crear el DataFrame y eliminar cualquier fila donde 'position' sea NaN
    df = pd.DataFrame(registros)
    
    if df.empty:
        st.error("No se encontraron datos válidos para la temporada 2025.")
        return pd.DataFrame()  # Retornar un DataFrame vacío si no hay datos
    
    # Asegurarse de que la columna 'position' exista antes de intentar manipularla
    if 'position' in df.columns:
        df.dropna(subset=['position'], inplace=True)  # Eliminar filas donde no haya posición
        df['win'] = (df['position'] == 1).astype(int)  # 1 si la posición es la ganadora
    
    # Verificar si la columna 'win' se ha creado correctamente
    if 'win' not in df.columns:
        st.error("Error: la columna 'win' no se ha creado correctamente.")
        return pd.DataFrame()  # Retornar un DataFrame vacío en caso de error

    return df

# Llamada para cargar los datos
data = cargar_datos()

# Si data está vacío, no continuar
if data.empty:
    st.error("No se pudieron cargar los datos correctamente.")
else:
    # Mostrar los primeros datos cargados
    st.subheader("📊 Datos Reales Temporada 2025")
    st.dataframe(data.head(10))

    # --- VISUALIZACION ---
    st.subheader("🏁 Victorias por Piloto")
    wins = data[data['win'] == 1].groupby('driver').size().sort_values(ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x=wins.values, y=wins.index, ax=ax, palette="Reds_r")
    ax.set_xlabel("Victorias")
    ax.set_ylabel("Piloto")
    st.pyplot(fig)

    # --- MODELO PREDICTIVO ---
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

    # --- FORMULARIO DE PREDICCION ---
    st.sidebar.header("🔮 Predicción Personalizada")  # Aquí se cerró correctamente la cadena
    pilotos = list(le_driver.classes_)
    equipos = list(le_team.classes_)
    piloto_sel = st.sidebar.selectbox("Piloto", pilotos)
    equipo_sel = st.sidebar.selectbox("Equipo", equipos)
    grid_sel = st.sidebar.slider("Posición de largada (Grid)", 1, 20, 5)

    if st.sidebar.button("Predecir Ganador"):
        datos_input = np.array([
            le_driver.transform([piloto_sel])[0],
            le_team.transform([equipo_sel])[0],
            grid_sel
        ]).reshape(1, -1)
        prediccion = model.predict(datos_input)
        resultado = "GANARÁ la carrera" if prediccion[0] == 1 else "NO ganará"
        st.success(f"🧠 Según el modelo, {piloto_sel} {resultado}.")

























