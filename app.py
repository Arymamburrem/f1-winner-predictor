import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Estilos F1 ---
def inject_f1_style():
    st.markdown('''
        <style>
            body {
                background-color: #000000;
                color: #FFFFFF;
            }
            .stApp {
                background-color: #111111;
                padding: 2rem;
                border-radius: 15px;
            }
            h1, h2, h3, h4 {
                color: #E10600;
                font-family: 'Formula1-Regular', sans-serif;
            }
            .stButton>button {
                background-color: #E10600;
                color: white;
                font-weight: bold;
                border-radius: 8px;
                border: none;
                padding: 0.5em 2em;
            }
            .stButton>button:hover {
                background-color: #B30000;
                color: #FFFFFF;
            }
            .css-1d391kg, .css-1kyxreq {
                background-color: #1C1C1C;
                border-radius: 10px;
            }
        </style>
    ''', unsafe_allow_html=True)

inject_f1_style()

# --- Banner Superior ---
st.markdown("""
<div style='text-align: center;'>
    <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/F1.svg/1200px-F1.svg.png' width='300'/>
    <h1>🏎️ F1 Race Winner Predictor 🏆</h1>
</div>
""", unsafe_allow_html=True)

# --- Datos Profesionales ---
@st.cache_data
def load_pilots_data():
    return pd.DataFrame({
        'Piloto': ['Verstappen', 'Hamilton', 'Leclerc', 'Alonso', 'Norris'],
        'Nacionalidad': ['Holandés', 'Británico', 'Monegasco', 'Español', 'Británico'],
        'Equipo': ['Red Bull', 'Mercedes', 'Ferrari', 'Aston Martin', 'McLaren'],
        'Años en F1': [8, 18, 6, 21, 6],
        'Poles': [35, 104, 23, 22, 1],
        'Podios': [95, 197, 35, 106, 16],
        'Campeonatos': [3, 7, 0, 2, 0],
        'Carreras Ganadas': [60, 103, 5, 32, 1]
    })

pilots = load_pilots_data()

# --- Mostrar tarjetas de pilotos ---
st.subheader('Pilotos Destacados')

for idx, row in pilots.iterrows():
    st.markdown(f"""
    <div style='background-color: #1C1C1C; padding: 1rem; border-radius: 12px; margin-bottom: 1rem;'>
        <h3>🏎️ {row['Piloto']}</h3>
        <ul>
            <li><strong>🌍 Nacionalidad:</strong> {row['Nacionalidad']}</li>
            <li><strong>🚥 Equipo:</strong> {row['Equipo']}</li>
            <li><strong>📅 Años en F1:</strong> {row['Años en F1']}</li>
            <li><strong>🎯 Poles:</strong> {row['Poles']}</li>
            <li><strong>🏆 Podios:</strong> {row['Podios']}</li>
            <li><strong>👑 Campeonatos:</strong> {row['Campeonatos']}</li>
            <li><strong>🥇 Carreras Ganadas:</strong> {row['Carreras Ganadas']}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# --- Gráfico de Barras ---
st.subheader('Comparación de Carreras Ganadas')

fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(x='Carreras Ganadas', y='Piloto', data=pilots,
            palette='Reds_r', edgecolor="black")

ax.set_facecolor("#111111")
fig.patch.set_facecolor('#111111')
ax.set_xlabel('Carreras Ganadas', color='white')
ax.set_ylabel('Piloto', color='white')
ax.set_title('🏁 Pilotos con Más Victorias', color='white', fontsize=18)
ax.tick_params(colors='white')

st.pyplot(fig)

# --- Sonido de Motor ---
st.subheader('¡Arranquemos motores! 🔊')
engine_sound_url = "https://www.soundjay.com/mechanical/sounds/race-car-engine-01.mp3"

if st.button('🎵 Reproducir Sonido de Motor'):
    st.markdown(f"""
        <audio autoplay>
            <source src="{engine_sound_url}" type="audio/mpeg">
        </audio>
    """, unsafe_allow_html=True)






