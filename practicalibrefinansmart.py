import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Análisis de Rentabilidad y Riesgo - Portafolio Financiero")

# Entrada de usuario
tickers = st.text_input("Ingrese los símbolos de las acciones (separados por comas):", "AAPL, MSFT, TSLA")

if st.button("Analizar"):
    acciones = [t.strip().upper() for t in tickers.split(",")]
    data = yf.download(acciones, start="2020-01-01")['Adj Close']
    
    retornos = data.pct_change().dropna()
    rent_anual = retornos.mean() * 252
    riesgo_anual = retornos.std() * np.sqrt(252)
    
    resultados = pd.DataFrame({
        'Rentabilidad Anual': rent_anual,
        'Riesgo Anual': riesgo_anual
    })
    
    st.subheader("Resultados Individuales")
    st.dataframe(resultados)
    
    st.subheader("Matriz de Correlación")
    fig, ax = plt.subplots()
    sns.heatmap(retornos.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    
    st.subheader("Precios Históricos")
    st.line_chart(data)
