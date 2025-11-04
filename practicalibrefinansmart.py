# -- coding: utf-8 --
"""
FinSight - Analizador de Rentabilidad y Riesgo Empresarial
Aplicación Streamlit para análisis financiero de empresas
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ⚙️ Configuración de página
st.set_page_config(page_title="FinSight", page_icon="💼", layout="wide")

# 🎯 Encabezado principal
st.title("💼 FinSight – Analizador de Rentabilidad y Riesgo Empresarial")
st.markdown("Explora el desempeño financiero de distintas empresas a través de indicadores de *rentabilidad* y *riesgo*.")
st.divider()

# 🔍 Entrada del usuario
ticker = st.text_input("📊 Ingresa el ticker de la empresa (por ejemplo: AAPL, MSFT, NVDA):", "AAPL")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("📅 Fecha inicial:", pd.to_datetime("2020-01-01"))
with col2:
    end_date = st.date_input("📅 Fecha final:", pd.to_datetime("2024-12-31"))

# 🚀 Botón de análisis
if st.button("Analizar"):
    with st.spinner("Descargando datos financieros..."):
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    # Verificación de datos
    if data.empty:
        st.error("❌ No se encontraron datos para el ticker especificado. Verifica que sea válido.")
        st.stop()

    st.success(f"✅ Datos descargados exitosamente para *{ticker}*")

    # 🧮 Asegurar que las columnas sean planas (a veces vienen en MultiIndex)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Verificar qué columna usar (Adj Close o Close)
    if "Adj Close" in data.columns:
        price_col = "Adj Close"
    elif "Close" in data.columns:
        price_col = "Close"
    else:
        st.error("❌ No se encontró ninguna columna de precios ('Adj Close' o 'Close').")
        st.stop()

    # 📈 Cálculos de rentabilidad y riesgo
    data["Daily Return"] = data[price_col].pct_change()
    avg_return = data["Daily Return"].mean()
    std_dev = data["Daily Return"].std()
    sharpe_ratio = avg_return / std_dev if std_dev != 0 else 0

    # 📊 Mostrar métricas
    st.subheader("📈 Indicadores de Rentabilidad y Riesgo")
    metrics_df = pd.DataFrame({
        'Indicador': ['Rentabilidad promedio (%)', 'Riesgo (Desviación estándar %)', 'Sharpe Ratio'],
        'Valor': [avg_return * 100, std_dev * 100, sharpe_ratio]
    })
    st.table(metrics_df.style.format({'Valor': '{:.2f}'}))

    # 🕰 Evolución del precio
    st.subheader("📉 Evolución del Precio Ajustado")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data[price_col], color='royalblue', linewidth=2)
    ax.set_title(f"Precio histórico de {ticker}", fontsize=14)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio ($)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # 🔢 Histograma de rendimientos
    st.subheader("📊 Distribución de los Rendimientos Diarios")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.histplot(data["Daily Return"].dropna(), bins=30, kde=True, ax=ax2, color='teal')
    ax2.set_title("Distribución de Retornos Diarios")
    st.pyplot(fig2)

    # 🧾 Datos adicionales
    st.subheader("🧾 Vista previa de los datos")
    st.dataframe(data.tail(), use_container_width=True)

# 🪪 Footer
st.markdown("---")
st.markdown("Desarrollado con ❤️ por *Angie* | Fuente de datos: Yahoo Finance")
