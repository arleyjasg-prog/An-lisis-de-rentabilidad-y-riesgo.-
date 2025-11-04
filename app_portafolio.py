# -*- coding: utf-8 -*-
"""
Portafolio Proactivo - Optimizador de Inversiones
Aplicación para análisis de riesgo, rentabilidad y optimización de portafolios.
"""

# 1. IMPORTACIÓN DE LIBRERÍAS
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

# 2. CONFIGURACIÓN INICIAL DE LA PÁGINA
st.set_page_config(
    page_title="Portafolio Proactivo",
    page_icon="📈",
    layout="wide"
)

# --- TÍTULO Y DESCRIPCIÓN ---
st.title("📈 Portafolio Proactivo")
st.markdown("### Una herramienta para el análisis de rentabilidad, riesgo y optimización de portafolios de inversión.")
st.write("""
Esta aplicación te permite construir y analizar un portafolio de acciones. Ingresa los tickers de las empresas que te interesan,
selecciona un rango de fechas y la herramienta calculará las métricas clave y encontrará la combinación óptima de activos
para maximizar tu rentabilidad ajustada al riesgo (Ratio de Sharpe).
""")
st.markdown("---")

# 3. BARRA LATERAL (SIDEBAR) PARA ENTRADAS DEL USUARIO
st.sidebar.header("⚙️ Parámetros de Análisis")

tickers_input = st.sidebar.text_input(
    "Introduce los tickers (separados por comas)",
    value="AAPL, MSFT, NVDA, GOOGL"
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Fecha de Inicio", date(2021, 1, 1))
with col2:
    end_date = st.date_input("Fecha de Fin", date.today())

num_simulaciones = st.sidebar.slider(
    "Número de Simulaciones Monte Carlo",
    min_value=1000,
    max_value=20000,
    value=5000,
    step=1000
)

# 4. BOTÓN PARA EJECUTAR EL ANÁLISIS
if st.sidebar.button("🚀 Analizar Portafolio", type="primary"):

    if not tickers:
        st.error("Por favor, introduce al menos un ticker para analizar.")
    else:
        # --- BLOQUE CORREGIDO ---
        with st.spinner(f"Descargando datos para: {', '.join(tickers)}..."):
            try:
                # Paso 1: Descargar todos los datos sin seleccionar ninguna columna todavía.
                full_data = yf.download(tickers, start=start_date, end=end_date, progress=False)

                # Paso 2: Verificar si el DataFrame está vacío.
                if full_data.empty:
                    st.error("No se pudieron descargar datos. Revisa los tickers o el rango de fechas.")
                    st.stop()

                # Paso 3: Procesar los datos según si se pidió 1 o más tickers.
                if len(tickers) == 1:
                    data = full_data[['Adj Close']]
                    data.columns = tickers
                else:
                    data = full_data['Adj Close']

                # Paso 4: Limpiar datos nulos.
                data.dropna(inplace=True)
                
                if data.empty:
                    st.error("No hay datos disponibles para el período de tiempo seleccionado después de la limpieza.")
                    st.stop()

            except Exception as e:
                st.error(f"Ocurrió un error inesperado durante la descarga: {e}")
                st.stop()
        # --- FIN DEL BLOQUE CORREGIDO ---

        st.success("✅ Datos descargados exitosamente.")

        # --- SECCIÓN 1: ANÁLISIS DE ACTIVOS INDIVIDUALES ---
        st.header("1. Análisis de Activos Individuales")
        retornos_diarios = data.pct_change().dropna()
        rentabilidad_anual = retornos_diarios.mean() * 252
        volatilidad_anual = retornos_diarios.std() * np.sqrt(252)
        ratio_sharpe = rentabilidad_anual / volatilidad_anual

        resumen_activos = pd.DataFrame({
            'Rentabilidad Anual (%)': rentabilidad_anual * 100,
            'Volatilidad Anual (%)': volatilidad_anual * 100,
            'Ratio de Sharpe': ratio_sharpe
        })

        st.subheader("Métricas de Riesgo y Rentabilidad")
        st.dataframe(resumen_activos.style.format("{:.2f}"))

        st.subheader("Rendimiento Histórico Normalizado")
        precios_normalizados = (data / data.iloc[0] * 100)
        st.line_chart(precios_normalizados)

        if len(tickers) > 1:
            st.subheader("Matriz de Correlación")
            fig_corr, ax_corr = plt.subplots()
            sns.heatmap(retornos_diarios.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
            st.pyplot(fig_corr)
            st.info("La correlación mide cómo se mueven los activos entre sí. Valores cercanos a -1 indican diversificación, mientras que valores cercanos a 1 indican que se mueven juntos.")

        # --- SECCIÓN 2: OPTIMIZACIÓN DE PORTAFOLIO (MONTE CARLO) ---
        if len(tickers) > 1:
            st.header("2. Optimización del Portafolio con Simulación Monte Carlo")
            num_activos = len(tickers)
            resultados_simulacion = np.zeros((3, num_simulaciones))
            pesos_portafolios = []
            cov_matrix_anual = retornos_diarios.cov() * 252

            with st.spinner(f"Realizando {num_simulaciones} simulaciones..."):
                for i in range(num_simulaciones):
                    pesos = np.random.random(num_activos)
                    pesos /= np.sum(pesos)
                    pesos_portafolios.append(pesos)
                    retorno_portafolio = np.sum(rentabilidad_anual * pesos)
                    riesgo_portafolio = np.sqrt(np.dot(pesos.T, np.dot(cov_matrix_anual, pesos)))
                    resultados_simulacion[0, i] = riesgo_portafolio
                    resultados_simulacion[1, i] = retorno_portafolio
                    resultados_simulacion[2, i] = retorno_portafolio / riesgo_portafolio

            max_sharpe_idx = np.argmax(resultados_simulacion[2])
            riesgo_optimo = resultados_simulacion[0, max_sharpe_idx]
            retorno_optimo = resultados_simulacion[1, max_sharpe_idx]
            max_sharpe_ratio = resultados_simulacion[2, max_sharpe_idx]
            pesos_optimos = pesos_portafolios[max_sharpe_idx]

            # --- SECCIÓN 3: RESULTADOS DEL PORTAFOLIO ÓPTIMO ---
            st.header("3. Resultados del Portafolio Óptimo (Máximo Ratio de Sharpe)")
            col1, col2, col3 = st.columns(3)
            col1.metric("Rentabilidad Anual", f"{retorno_optimo*100:.2f}%")
            col2.metric("Volatilidad Anual", f"{riesgo_optimo*100:.2f}%")
            col3.metric("Ratio de Sharpe", f"{max_sharpe_ratio:.2f}")

            col_graf, col_pesos = st.columns([2, 1])
            with col_graf:
                st.subheader("Frontera Eficiente")
                fig_frontera, ax_frontera = plt.subplots(figsize=(10, 6))
                scatter = ax_frontera.scatter(
                    resultados_simulacion[0, :],
                    resultados_simulacion[1, :],
                    c=resultados_simulacion[2, :],
                    cmap='viridis', marker='o', s=10, alpha=0.5
                )
                ax_frontera.scatter(
                    riesgo_optimo, retorno_optimo,
                    marker='*', color='r', s=200, label='Portafolio Óptimo'
                )
                ax_frontera.set_title('Frontera Eficiente y Portafolio Óptimo')
                ax_frontera.set_xlabel('Volatilidad (Riesgo)')
                ax_frontera.set_ylabel('Rentabilidad Esperada')
                ax_frontera.legend()
                fig_frontera.colorbar(scatter, label='Ratio de Sharpe')
                st.pyplot(fig_frontera)

            with col_pesos:
                st.subheader("Distribución Óptima de Activos")
                df_pesos = pd.DataFrame({'Activo': tickers, 'Peso': pesos_optimos})
                fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
                ax_pie.pie(
                    df_pesos['Peso'], labels=df_pesos['Activo'],
                    autopct='%1.1f%%', startangle=90, pctdistance=0.85
                )
                centre_circle = plt.Circle((0,0),0.70,fc='white')
                fig_pie.gca().add_artist(centre_circle)
                ax_pie.axis('equal')
                plt.tight_layout()
                st.pyplot(fig_pie)
                st.dataframe(df_pesos.style.format({'Peso': '{:.2%}'}), use_container_width=True)

else:
    st.info("👈 Configura los parámetros en el panel lateral y haz clic en 'Analizar Portafolio' para empezar.")
    with st.expander("ℹ️ ¿Cómo funciona esta aplicación?"):
        st.markdown("""
        1.  **Introduce los Tickers:** Escribe los símbolos de las acciones que quieres analizar (ej. `MELI, BCOLOMBIA.CN, TSLA`).
        2.  **Define el Periodo:** Selecciona el rango de fechas para el análisis histórico.
        3.  **Simulación Monte Carlo:** La aplicación genera miles de portafolios con combinaciones de pesos aleatorias para los activos que elegiste.
        4.  **Frontera Eficiente:** Se grafica cada portafolio simulado en un mapa de riesgo vs. rentabilidad. La curva que se forma es la frontera eficiente.
        5.  **Portafolio Óptimo:** Identificamos el punto en esa frontera con el **Ratio de Sharpe** más alto. Este ratio mide la rentabilidad que obtienes por cada unidad de riesgo que asumes. Un Sharpe más alto es mejor.
        6.  **Resultados:** Te mostramos las métricas de este portafolio óptimo y, lo más importante, qué porcentaje de tu dinero deberías invertir en cada activo para lograrlo.
        """)

st.markdown("---")
st.markdown("Desarrollado por el **Grupo 10** | Ingeniería Financiera")
