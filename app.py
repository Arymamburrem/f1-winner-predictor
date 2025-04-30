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

# --- ESTILO Y FONDO PERSONALIZADO ---
st.markdown("""
    <style>
        body, .main {
            background-image: url("https://www.transparenttextures.com/patterns/black-paper.png");
            background-size: cover;
            color: white;
        }
        h1, h2, h3, h4 {
            color: #E10600;
        }
        .stButton>button {
            background-color: #E10600;
            color: white;
        }
        .stButton>button:hover {
            background-color: #990000;
        }
        .next-race-box {
            background: rgba(255, 255, 255, 0.05);
            border-left: 5px solid #E10600;
            padding: 1rem;
            margin-bottom: 2rem;
            border-radius: 8px;
        }
        .centered {
            display: flex;
            justify-content: center;
            align-items: center;
        }
    </style>
""", unsafe_allow_html=True)

# --- LOGO GRANDE Y CENTRADO ---
st.markdown('<div class="centered">', unsafe_allow_html=True)
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/F1.svg/1200px-F1.svg.png", width=300)
st.markdown('</div>', unsafe_allow_html=True)

st.title("🏎️ F1 Race Predictor 2025")





























