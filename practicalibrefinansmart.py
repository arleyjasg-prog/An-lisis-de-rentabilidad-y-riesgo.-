# -*- coding: utf-8 -*-
"""
FinanSmart Lite - Análisis de Rentabilidad y Riesgo
Versión optimizada para presentación académica
"""

# Importamos las librerías
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Análisis de Rentabilidad y Riesgo - Grupo 10",
    page_icon="💼",
    layout="wide"
)

# --- TÍTULO PRINCIPAL ---
st.markdown(
    """
    <h1 style='text-align: center; color: #1E90FF;'>
        💼 Análisis de Rentabilidad y Riesgo - Grupo 10
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

st.write("""
Este proyecto permite analizar **la rentabilidad esperada, el riesgo (volatilidad)** y el **índice de Sharpe** de acciones globales.
Podrás comparar el comportamiento de una o varias empresas, visualizar su relación **riesgo-retorno** y obtener conclusiones financieras. 📊
""")

# --- SIDEBAR PARA CONFIGURACIÓN ---
st.sidebar.header("⚙️ Configuración del Análisis")

# Campo de texto para ingresar tickers personalizados
default_tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]
tickers_input = st.sidebar.text_input(
    "📈 Ingrese los tickers (separados por comas):",
    value=",".join(default_tickers)
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# Periodo de análisis
periodo = st.sidebar.selectbox(
    "⏱️ Periodo de análisis:",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
    index=3
)

# --- BOTÓN PARA CALCULAR ---
if st.sidebar.button("🚀 Calcular Rentabilidad y Riesgo"):
    if not tickers:
        st.warning("Por favor, ingrese al menos un ticker válido.")
    else:
        try:
            # --- DESCARGA DE DATOS ---
            st.info("Descargando datos desde Yahoo Finance...")
            data = yf.download(tickers=tickers, period=periodo, progress=False)["Adj Close"]

            if data.empty:
                st.error("No se encontraron datos para los tickers ingresados.")
                st.stop()

            st.success("✅ Datos descargados correctamente.")
            st.markdown("### 1️⃣ Evolución de Precios")
            st.line_chart(data)

            # --- CÁLCULO DE RENTABILIDAD Y RIESGO ---
            rent_diaria = data.pct_change().dropna()
            rent_promedio = rent_diaria.mean() * 252      # rentabilidad anual esperada
            riesgo = rent_diaria.std() * (252 ** 0.5)     # riesgo (volatilidad anual)
            sharpe = rent_promedio / riesgo               # índice de Sharpe (sin tasa libre de riesgo)

            resumen = pd.DataFrame({
                "Rentabilidad esperada (%)": rent_promedio * 100,
                "Riesgo (Volatilidad %)": riesgo * 100,
                "Ratio Sharpe": sharpe
            }).round(3)

            # --- TABLA DE RESULTADOS ---
            st.markdown("### 2️⃣ Resultados del Análisis")
            st.dataframe(resumen.style.format("{:.2f}"))

            # --- GRÁFICO DE RENTABILIDAD VS RIESGO ---
            st.markdown("### 3️⃣ Gráfico de Rentabilidad vs Riesgo")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(riesgo * 100, rent_promedio * 100, color="#1E90FF", s=120)

            for i, txt in enumerate(resumen.index):
                ax.annotate(txt, (riesgo[i] * 100, rent_promedio[i] * 100), xytext=(5, 5), textcoords="offset points")

            ax.set_xlabel("Riesgo (Volatilidad %)")
            ax.set_ylabel("Rentabilidad Esperada (%)")
            ax.set_title("Riesgo vs Rentabilidad")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # --- INTERPRETACIÓN BÁSICA ---
            st.markdown("### 4️⃣ Interpretación")
            mejor_accion = resumen["Rentabilidad esperada (%)"].idxmax()
            menor_riesgo = resumen["Riesgo (Volatilidad %)"].idxmin()
            st.success(f"📈 La acción con mayor rentabilidad esperada es **{mejor_accion}**.")
            st.info(f"🛡️ La acción con menor riesgo es **{menor_riesgo}**.")
            st.caption("El ratio de Sharpe indica qué activo ofrece mejor rentabilidad ajustada al riesgo.")

        except Exception as e:
            st.error(f"⚠️ Error al procesar los datos: {e}")

else:
    st.info("👈 Configura los parámetros en el panel lateral y presiona **Calcular Rentabilidad y Riesgo** para comenzar.")
    st.image(
        "https://cdn.pixabay.com/photo/2017/06/16/07/37/stock-exchange-2408858_1280.jpg",
        use_container_width=True
    )
    st.markdown(
        """
        ---
        **Guía de uso:**
        1. Ingresa uno o más *tickers* (símbolos bursátiles) — Ejemplo: `AAPL, TSLA, NVDA`
        2. Selecciona el periodo de análisis (por ejemplo, 1 año o 5 años)
        3. Haz clic en **Calcular Rentabilidad y Riesgo**
        4. Observa los resultados, gráficos e interpretación final.
        """
    )

# --- FOOTER ---
st.markdown("---")
st.caption("Desarrollado por Grupo 10 | Proyecto Académico de Análisis de Rentabilidad y Riesgo | Datos: Yahoo Finance 📊")
