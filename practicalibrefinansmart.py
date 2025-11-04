# FinanSmart - Análisis de Portafolio Simple
# Proyecto para principiantes en Python

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========== CONFIGURACIÓN INICIAL ==========
st.set_page_config(page_title="FinanSmart", page_icon="📊")

# ========== TÍTULO ==========
st.title("📊 FinanSmart - Análisis de Inversiones")
st.write("Aplicación simple para analizar acciones")
st.markdown("---")

# ========== BARRA LATERAL (INPUTS) ==========
st.sidebar.header("Configuración")

# Pedir los tickers al usuario
ticker1 = st.sidebar.text_input("Primera acción (Ticker)", "AAPL")
ticker2 = st.sidebar.text_input("Segunda acción (Ticker)", "MSFT")

# Convertir a mayúsculas
ticker1 = ticker1.upper()
ticker2 = ticker2.upper()

# Pedir las fechas
