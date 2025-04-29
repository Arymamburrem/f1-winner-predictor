import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
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

# --- Datos de ejemplo ---
@st.cache_data
def load_data():
    df = pd.read_csv("Formula1_2023season_raceResults.csv")
    df = df[['Driver', 'Constructor', 'Grid', 'Position', 'Points']]
    df = df.dropna()
    df = df[df['Position'].apply(lambda x: str(x).isdigit())]
    df['Position'] = df['Position'].astype(int)
    df['Wins'] = (df['Position'] == 1).astype(int)
    return df

data = load_data()

st.subheader('Resultados Temporada F1 2023')
st.dataframe(data.head(20))

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

X_basic = data[['Driver_encoded', 'Constructor_encoded', 'Grid']]
y_basic = data['Wins']

X_train_basic, X_test_basic, y_train_basic, y_test_basic = train_test_split(X_basic, y_basic, test_size=0.2, random_state=42)

model_basic = RandomForestClassifier(n_estimators=100, random_state=42)
model_basic.fit(X_train_basic, y_train_basic)

# --- Simulamos otro dataset para regresión ---
data_adv = pd.DataFrame({
    'Piloto': ['Verstappen', 'Hamilton', 'Leclerc', 'Alonso', 'Norris'],
    'Equipo': ['Red Bull', 'Mercedes', 'Ferrari', 'Aston Martin', 'McLaren'],
    'Clasificación': [1, 2, 3, 4, 5],
    'Carreras Ganadas': [10, 8, 5, 4, 2]
})

le_piloto = LabelEncoder()
le_equipo = LabelEncoder()

data_adv['Piloto_encoded'] = le_piloto.fit_transform(data_adv['Piloto'])
data_adv['Equipo_encoded'] = le_equipo.fit_transform(data_adv['Equipo'])

X = data_adv[['Piloto_encoded', 'Equipo_encoded', 'Clasificación']]
y = data_adv['Carreras Ganadas']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_reg = RandomForestRegressor(n_estimators=100, random_state=42)
model_reg.fit(X_train, y_train)

# --- Sidebar - Entrada de datos Básica ---
st.sidebar.header("Predicción Básica")
piloto = st.sidebar.selectbox('Selecciona un Piloto', data_adv['Piloto'].unique())
equipo = st.sidebar.selectbox('Selecciona un Equipo', data_adv['Equipo'].unique())
clasificacion = st.sidebar.slider('Clasificación en Qualy', 1, 20, 1)

if st.sidebar.button('Predecir Carreras Ganadas'):
    input_data = np.array([
        le_piloto.transform([piloto])[0],
        le_equipo.transform([equipo])[0],
        clasificacion
    ]).reshape(1, -1)
    prediccion = model_reg.predict(input_data)

    st.success(f'Predicción de carreras ganadas: {int(prediccion[0])}')

    engine_sound_url = "https://www.soundjay.com/mechanical/sounds/race-car-engine-01.mp3"
    st.markdown(f"""
        <audio autoplay>
            <source src="{engine_sound_url}" type="audio/mpeg">
        </audio>
    """, unsafe_allow_html=True)

# --- Datos para la Predicción Avanzada (simulados) ---
nationalities = ['Holandés', 'Británico', 'Monegasco', 'Español', 'Australiano']
teams = ['Red Bull', 'Mercedes', 'Ferrari', 'Aston Martin', 'McLaren']
countries = ['España', 'Italia', 'Mónaco', 'Reino Unido', 'Australia']
weather_conditions = ['Soleado', 'Lluvia', 'Nublado']
tire_types = ['Soft', 'Medium', 'Hard']

st.sidebar.header("Predicción Avanzada")
year = st.sidebar.number_input("Año", min_value=1950, max_value=2025, value=2025)
round_number = st.sidebar.number_input("Ronda", min_value=1, max_value=25, value=3)
nationality = st.sidebar.selectbox("Nacionalidad del Piloto", nationalities)
team = st.sidebar.selectbox("Equipo", teams)
country = st.sidebar.selectbox("País del Circuito", countries)
weather = st.sidebar.selectbox("Condición Climática", weather_conditions)
tire = st.sidebar.selectbox("Tipo de Neumático", tire_types)

# --- Modelo simulado para predicción avanzada ---
X_adv = pd.DataFrame({
    'nationality': np.random.randint(0, 5, 50),
    'team': np.random.randint(0, 5, 50),
    'country': np.random.randint(0, 5, 50),
    'weather': np.random.randint(0, 3, 50),
    'tire': np.random.randint(0, 3, 50),
    'year': np.random.randint(2000, 2025, 50),
    'round': np.random.randint(1, 23, 50)
})

y_adv = np.random.choice(['Verstappen', 'Hamilton', 'Leclerc', 'Alonso', 'Norris'], 50)

X_adv_train, X_adv_test, y_adv_train, y_adv_test = train_test_split(X_adv, y_adv, test_size=0.2, random_state=42)

model_adv = RandomForestRegressor(n_estimators=100, random_state=42)
model_adv.fit(X_adv_train, np.arange(len(y_adv_train)))

if st.sidebar.button("Predecir Ganador"):
    input_data_adv = pd.DataFrame({
        'nationality': [nationalities.index(nationality)],
        'team': [teams.index(team)],
        'country': [countries.index(country)],
        'weather': [weather_conditions.index(weather)],
        'tire': [tire_types.index(tire)],
        'year': [year],
        'round': [round_number]
    })

    pred_adv = model_adv.predict(input_data_adv)
    predicted_index = int(pred_adv[0]) % len(y_adv)
    predicted_winner = y_adv[predicted_index]

    st.success(f'Ganador Predicho: {predicted_winner}')

    engine_sound_url = "https://www.soundjay.com/mechanical/sounds/race-car-engine-01.mp3"
    st.markdown(f"""
        <audio autoplay>
            <source src="{engine_sound_url}" type="audio/mpeg">
        </audio>
    """, unsafe_allow_html=True)

st.markdown(f"### R² del Modelo de Clasificación: `{r2_score(y_test_basic, model_basic.predict(X_test_basic)):.2f}`")













