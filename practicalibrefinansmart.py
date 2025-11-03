# -*- coding: utf-8 -*-
"""
FinRisk Pro - Análisis Profesional de Portafolios
Grupo 10 - Ingeniería Financiera
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

st.markdown("---")

# ============================================================================
# BARRA LATERAL - CONFIGURACIÓN
# ============================================================================
st.sidebar.header("⚙️ Configuración del Análisis")
st.sidebar.markdown("---")

# Lista ampliada de tickers
lista_tickers = {
    "🇺🇸 Tecnología": ["AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN", "TSLA"],
    "🇺🇸 Financiero": ["JPM", "BAC", "WFC", "GS"],
    "🇺🇸 Consumo": ["KO", "PEP", "NKE", "MCD"],
    "🇨🇴 Colombianas": ["ECOPETROL.SN", "GRUPOSURA.SN", "NUTRESA.SN"]
}

# Selección por categoría
categoria = st.sidebar.selectbox(
    "Selecciona una categoría",
    options=list(lista_tickers.keys())
)

# Multiselect de tickers
ticker = st.sidebar.multiselect(
    "Elige las empresas a analizar",
    options=lista_tickers[categoria],
    default=lista_tickers[categoria][:3] if len(lista_tickers[categoria]) >= 3 else lista_tickers[categoria]
)

st.sidebar.markdown("---")

# Selección de fechas
st.sidebar.subheader("📅 Rango de Fechas")
col1, col2 = st.sidebar.columns(2)
with col1:
    fecha_inicio = st.date_input(
        "Desde",
        value=datetime.now() - timedelta(days=365)
    )
with col2:
    fecha_fin = st.date_input(
        "Hasta",
        value=datetime.now()
    )

st.sidebar.markdown("---")

# Tasa libre de riesgo
tasa_libre_riesgo = st.sidebar.slider(
    "Tasa Libre de Riesgo Anual (%)",
    min_value=0.0,
    max_value=10.0,
    value=4.5,
    step=0.1
) / 100

# Número de simulaciones para Monte Carlo
num_simulaciones = st.sidebar.slider(
    "Simulaciones Monte Carlo",
    min_value=1000,
    max_value=20000,
    value=5000,
    step=1000
)

st.sidebar.markdown("---")

# ============================================================================
# IMAGEN DECORATIVA
# ============================================================================
st.image("https://cdn.pixabay.com/photo/2017/06/16/07/37/stock-exchange-2408858_1280.jpg", 
         use_container_width=True)

st.markdown("---")

# ============================================================================
# BOTÓN PRINCIPAL DE ANÁLISIS
# ============================================================================
if st.button("🚀 Calcular Rentabilidad y Riesgo", type="primary", use_container_width=True):
    if not ticker:
        st.warning("⚠️ Selecciona al menos una empresa para continuar.")
    else:
        with st.spinner("🔄 Descargando datos y realizando análisis..."):
            try:
                # ================================================================
                # DESCARGA DE DATOS
                # ================================================================
                data = yf.download(
                    tickers=ticker,
                    start=fecha_inicio,
                    end=fecha_fin,
                    progress=False
                )["Close"]
                
                if data.empty:
                    st.error("❌ No se pudieron descargar datos. Verifica los tickers y fechas.")
                    st.stop()
                
                # Si solo hay un ticker, convertir a DataFrame
                if len(ticker) == 1:
                    data = pd.DataFrame(data, columns=ticker)
                
                st.success("✅ Datos descargados exitosamente")
                
                # ================================================================
                # SECCIÓN 1: DATOS HISTÓRICOS
                # ================================================================
                st.header("1️⃣ Datos Históricos de Precios")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.dataframe(data.tail(10), use_container_width=True)
                with col2:
                    st.info(f"""
                    **📈 Información del dataset:**
                    - Empresas: {len(ticker)}
                    - Periodo: {len(data)} días
                    - Desde: {data.index[0].strftime('%Y-%m-%d')}
                    - Hasta: {data.index[-1].strftime('%Y-%m-%d')}
                    """)
                
                # Gráfico de evolución
                st.subheader("📈 Evolución de Precios")
                fig1, ax1 = plt.subplots(figsize=(12, 6))
                for col in data.columns:
                    ax1.plot(data.index, data[col], label=col, linewidth=2)
                ax1.set_title('Evolución de Precios de Cierre', fontsize=16, fontweight='bold')
                ax1.set_xlabel('Fecha', fontsize=12)
                ax1.set_ylabel('Precio ($)', fontsize=12)
                ax1.legend(loc='best')
                ax1.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig1)
                
                st.markdown("---")
                
                # ================================================================
                # CÁLCULO DE RENTABILIDADES
                # ================================================================
                rent_diaria = data.pct_change().dropna()
                
                # ================================================================
                # SECCIÓN 2: ANÁLISIS DE RETORNOS
                # ================================================================
                st.header("2️⃣ Análisis de Retornos Diarios")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Estadísticas Descriptivas")
                    st.dataframe(
                        rent_diaria.describe().style.format("{:.4f}"),
                        use_container_width=True
                    )
                
                with col2:
                    st.subheader("📉 Retornos Diarios")
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    for col in rent_diaria.columns:
                        ax2.plot(rent_diaria.index, rent_diaria[col], 
                                label=col, alpha=0.7, linewidth=1)
                    ax2.set_title('Retornos Diarios', fontsize=14, fontweight='bold')
                    ax2.set_xlabel('Fecha')
                    ax2.set_ylabel('Retorno')
                    ax2.legend(loc='best')
                    ax2.grid(True, alpha=0.3)
                    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    st.pyplot(fig2)
                
                st.markdown("---")
                
                # ================================================================
            
