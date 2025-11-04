# -- coding: utf-8 --
"""
FinSight - Analizador de Rentabilidad y Riesgo Empresarial
Aplicación Streamlit para análisis financiero de empresas
Autor: Angie, Dayana y Jhony
Versión: 2.0 (con barra lateral y comparación múltiple)
"""

# ==========================
# 📦 Importaciones necesarias
# ==========================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# 🎨 Configuración visual y de página
# ==========================
st.set_page_config(page_title="FinSight", page_icon="💼", layout="wide")

st.markdown("""
    <style>
        .main { background-color: #f9f9fb; }
        h1, h2, h3 { color: #1f4e79; }
        .stButton>button {
            background-color: #1f4e79;
            color: white;
            font-weight: bold;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #16385a;
            color: #fff;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================
# 🎯 SELECCIÓN DE EMPRESAS (SIDEBAR)
# ==========================
st.sidebar.header("📊 Configuración de análisis")

default_tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]
tickers_input = st.sidebar.text_input(
    "Empresas a analizar (separadas por comas):",
    value=",".join(default_tickers)
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip() != ""]
st.sidebar.write("*Empresas seleccionadas:*", ", ".join(tickers))

start_date = st.sidebar.date_input("📅 Fecha inicial:", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("📅 Fecha final:", pd.to_datetime("2024-12-31"))

# ==========================
# 💼 Encabezado principal
# ==========================
st.title("💼 FinSight – Analizador de Rentabilidad y Riesgo Empresarial")
st.write("Explora y compara el desempeño financiero de distintas empresas mediante métricas de rentabilidad, riesgo y eficiencia.")
st.markdown("---")

# ==========================
# 📈 SECCIÓN 1: ANÁLISIS INDIVIDUAL
# ==========================
st.header("📈 Análisis Individual de Empresa")

ticker = st.selectbox("Selecciona la empresa a analizar:", tickers)

if st.button("Analizar Empresa"):
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if data.empty:
        st.error("❌ No se encontraron datos para el ticker especificado.")
    else:
        st.success(f"✅ Datos descargados exitosamente para *{ticker}*")

        # --- Cálculos ---
        data["Daily Return"] = data["Adj Close"].pct_change()
        avg_return = data["Daily Return"].mean()
        std_dev = data["Daily Return"].std()
        sharpe_ratio = avg_return / std_dev if std_dev != 0 else 0

        # --- Tabla de métricas ---
        metrics_df = pd.DataFrame({
            'Indicador': ['Rentabilidad promedio (%)', 'Riesgo (Desviación estándar %)', 'Sharpe Ratio'],
            'Valor': [avg_return * 100, std_dev * 100, sharpe_ratio]
        })
        st.table(metrics_df)

        # --- Gráfico de precios ---
        st.subheader("Evolución del precio ajustado")
        fig, ax = plt.subplots()
        ax.plot(data["Adj Close"], color='#1f77b4', linewidth=2)
        ax.set_title(f"Precio ajustado de {ticker}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Precio ($)")
        st.pyplot(fig)

        # --- Histograma de rendimientos ---
        st.subheader("Distribución de los rendimientos diarios")
        fig2, ax2 = plt.subplots()
        sns.histplot(data["Daily Return"].dropna(), bins=30, kde=True, ax=ax2, color='#ff7f0e')
        ax2.set_title("Histograma de retornos diarios")
        st.pyplot(fig2)

# ==========================
# ⚖ SECCIÓN 2: COMPARATIVO DE EMPRESAS
# ==========================
st.markdown("---")
st.header("📊 Comparativo de Empresas")

col1, col2 = st.columns(2)
with col1:
    ticker1 = st.selectbox("Ticker empresa 1:", tickers, index=0)
with col2:
    ticker2 = st.selectbox("Ticker empresa 2:", tickers, index=1)

if st.button("Comparar Empresas"):
    data = yf.download([ticker1, ticker2], start=start_date, end=end_date, progress=False)['Adj Close']

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(1)

    if data.empty:
        st.error("❌ No se encontraron datos para los tickers.")
    else:
        st.success("✅ Datos cargados correctamente.")

        # --- Gráfico comparativo ---
        st.subheader("Evolución comparativa de precios ajustados")
        fig, ax = plt.subplots()
        data.plot(ax=ax, linewidth=2)
        ax.set_title("Comparación de precios ajustados")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Precio ($)")
        ax.legend(title="Empresas")
        st.pyplot(fig)

        # --- Matriz de correlación ---
        returns = data.pct_change().dropna()
        st.subheader("Matriz de correlación de retornos diarios")
        fig2, ax2 = plt.subplots()
        sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', center=0, ax=ax2)
        st.pyplot(fig2)

# ==========================
# 💹 SECCIÓN 3: SIMULADOR DE PORTAFOLIO
# ==========================
st.markdown("---")
st.header("💹 Simulador de Portafolio de Inversión")

colp1, colp2 = st.columns(2)
with colp1:
    p_ticker1 = st.selectbox("📊 Empresa 1 (Ticker):", tickers, index=0)
    w1 = st.slider("Peso (%) Empresa 1", 0, 100, 50)
with colp2:
    p_ticker2 = st.selectbox("📈 Empresa 2 (Ticker):", tickers, index=1)
    w2 = 100 - w1
    st.write(f"Peso Empresa 2: *{w2}%*")

if st.button("Calcular Portafolio"):
    p_data = yf.download([p_ticker1, p_ticker2], start=start_date, end=end_date, progress=False)['Adj Close']

    if isinstance(p_data.columns, pd.MultiIndex):
        p_data.columns = p_data.columns.get_level_values(1)

    p_data = p_data.dropna()

    if p_data.empty:
        st.error("❌ No se encontraron datos para los tickers seleccionados.")
    else:
        st.success(f"✅ Datos obtenidos para *{p_ticker1}* y *{p_ticker2}*")

        returns = p_data.pct_change().dropna()
        weights = np.array([w1/100, w2/100])

        # --- Cálculos del portafolio ---
        exp_return = np.sum(returns.mean() * weights) * 252
        cov_matrix = returns.cov() * 252
        port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        port_std_dev = np.sqrt(port_variance)
        sharpe_ratio = exp_return / port_std_dev

        st.subheader("📈 Resultados del Portafolio")
        st.write(f"*Rentabilidad esperada (anualizada):* {exp_return*100:.2f}%")
        st.write(f"*Riesgo (Desviación estándar anual):* {port_std_dev*100:.2f}%")
        st.write(f"*Sharpe Ratio:* {sharpe_ratio:.2f}")

        # --- Pie chart ---
        st.subheader("📊 Composición del portafolio")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(weights, labels=[p_ticker1, p_ticker2], autopct='%1.1f%%', startangle=90, colors=['#1f77b4', '#ff7f0e'])
        ax.axis('equal')
        st.pyplot(fig)

        # --- Frontera eficiente ---
        st.subheader("📈 Frontera eficiente simulada")
        port_returns, port_risks = [], []
        for w in np.linspace(0, 1, 100):
            wts = np.array([w, 1-w])
            r = np.sum(returns.mean() * wts) * 252
            s = np.sqrt(np.dot(wts.T, np.dot(cov_matrix, wts)))
            port_returns.append(r)
            port_risks.append(s)

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.plot(port_risks, port_returns, 'b-', linewidth=2)
        ax2.scatter(port_std_dev, exp_return, color='red', s=80, label='Tu portafolio')
        ax2.set_xlabel('Riesgo (Desviación estándar)')
        ax2.set_ylabel('Rentabilidad esperada')
        ax2.set_title('Frontera eficiente (2 activos)')
        ax2.legend()
        st.pyplot(fig2)

# ==========================
# 🧾 Pie de página
# ==========================
st.markdown("---")
st.markdown("""
#### 💼 FinSight – “Analiza, Decide, Invierte”
Desarrollado por *Angie, Dayana y Jhony*, estudiantes de Análisis de costos y presupuestos.  
Hecho con Python usando *Streamlit* | Datos: Yahoo Finance API  
""")
