# -*- coding: utf-8 -*-
"""
FinRisk Pro - Análisis Profesional de Portafolios
Ingeniería Financiera
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="FinRisk Pro - Análisis de Portafolios",
    page_icon="💼",
    layout="wide"
)

# ============================================================================
# ESTILOS PERSONALIZADOS
# ============================================================================
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1E90FF;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# ENCABEZADO PRINCIPAL
# ============================================================================
st.markdown("""
    <h1 class='main-header'>
        💼 FinRisk Pro - Grupo 10
    </h1>
""", unsafe_allow_html=True)

st.markdown("### 📊 Análisis Profesional de Rentabilidad y Riesgo")
st.write("""
Este proyecto analiza el comportamiento financiero de empresas líderes del mercado.
A través de indicadores como la rentabilidad esperada, la volatilidad y el Ratio de Sharpe,
evaluamos la relación entre riesgo y retorno para identificar las mejores oportunidades de inversión. 💼
""")


    
