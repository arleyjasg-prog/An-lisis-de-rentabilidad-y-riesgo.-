import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuración inicial de la app ---
st.set_page_config(page_title="Análisis de Rentabilidad y Riesgo", layout="wide")
st.title("📈 Análisis de Rentabilidad y Riesgo de Acciones")

st.markdown("""
Esta herramienta permite analizar la **rentabilidad y el riesgo histórico** de una o dos acciones globales.
- Se descargan precios ajustados desde Yahoo Finance.
- Se calculan métricas anuales de retorno y volatilidad.
- Se grafican precios, retornos y comparación riesgo-rentabilidad.
""")

# --- Entrada de usuario ---
tickers = st.text_input("🔹 Ingrese 1 o 2 símbolos de acciones separados por coma (ejemplo: AAPL, TSLA):", "AAPL, TSLA")
acciones = [t.strip().upper() for t in tickers.split(",") if t.strip()]

# --- Fecha de análisis ---
start_date = st.date_input("📅 Fecha de inicio", pd.to_datetime("2020-01-01"))
end_date = st.date_input("📅 Fecha final", pd.Timestamp.today())

if st.button("Analizar"):
    if len(acciones) == 0:
        st.warning("Por favor, ingrese al menos una acción.")
    elif len(acciones) > 2:
        st.warning("El análisis está limitado a máximo dos acciones.")
    else:
        # --- Descarga de datos ---
        data = yf.download(acciones, start=start_date, end=end_date)['Adj Close']
        st.subheader("📊 Precios históricos")
        st.line_chart(data)

        # --- Cálculo de rentabilidades ---
        retornos = data.pct_change().dropna()

        rent_diaria = retornos.mean()
        rent_anual = rent_diaria * 252
        riesgo_anual = retornos.std() * np.sqrt(252)

        resultados = pd.DataFrame({
            'Rentabilidad Diaria Promedio': rent_diaria,
            'Rentabilidad Anual (%)': rent_anual * 100,
            'Riesgo Anual (%)': riesgo_anual * 100
        }).round(2)

        st.subheader("📈 Resultados de Rentabilidad y Riesgo")
        st.dataframe(resultados)

        # --- Correlación (si hay 2 acciones) ---
        if len(acciones) == 2:
            corr = retornos.corr().iloc[0, 1]
            st.markdown(f"**Correlación entre {acciones[0]} y {acciones[1]}:** `{corr:.2f}`")

            fig_corr, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(retornos.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig_corr)

        # --- Gráfico de dispersión riesgo vs rentabilidad ---
        st.subheader("📉 Relación Rentabilidad vs Riesgo")
        fig, ax = plt.subplots()
        ax.scatter(riesgo_anual * 100, rent_anual * 100, color='teal', s=100)
        for i, txt in enumerate(acciones):
            ax.annotate(txt, (riesgo_anual[i] * 100, rent_anual[i] * 100), xytext=(5, 5), textcoords="offset points")
        ax.set_xlabel("Riesgo (Volatilidad Anual %)")
        ax.set_ylabel("Rentabilidad Anual (%)")
        ax.set_title("Comparación de Rentabilidad y Riesgo")
        ax.grid(True)
        st.pyplot(fig)

        # --- Gráfico de distribución de retornos ---
        st.subheader("📊 Distribución de Retornos Diarios")
        fig2, ax2 = plt.subplots()
        retornos.plot(kind='hist', bins=50, alpha=0.6, ax=ax2)
        ax2.set_title("Distribución de Retornos Diarios")
        ax2.set_xlabel("Retorno Diario")
        st.pyplot(fig2)
