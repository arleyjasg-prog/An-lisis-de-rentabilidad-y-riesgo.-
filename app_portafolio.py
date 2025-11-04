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

# Usamos un text_input para que el usuario pueda ingresar cualquier ticker
# Esto resuelve la limitación de la lista fija
tickers_input = st.sidebar.text_input(
    "Introduce los tickers (separados por comas)",
    value="AAPL, MSFT, NVDA, GOOGL"
)
# Procesamos el input para tener una lista de tickers en mayúsculas
tickers = [t.strip().upper() for t in tickers_input.split(",")]

# Selección de fechas con valores por defecto razonables
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Fecha de Inicio", date(2021, 1, 1))
with col2:
    end_date = st.date_input("Fecha de Fin", date.today())

# Slider para definir el número de simulaciones
num_simulaciones = st.sidebar.slider(
    "Número de Simulaciones Monte Carlo",
    min_value=1000,
    max_value=20000,
    value=5000, # Un valor intermedio para que sea rápido
    step=1000
)

# 4. BOTÓN PARA EJECUTAR EL ANÁLISIS
if st.sidebar.button("🚀 Analizar Portafolio", type="primary"):

    # Validamos que el usuario haya ingresado al menos un ticker
    if not tickers_input:
        st.error("Por favor, introduce al menos un ticker para analizar.")
    else:
        # --- INICIO DEL ANÁLISIS ---
        with st.spinner(f"Descargando datos para: {', '.join(tickers)}..."):
            try:
                # Descargamos los precios de cierre ajustados
                data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

                # Si solo se descarga un ticker, yfinance devuelve una Serie, la convertimos a DataFrame
                if isinstance(data, pd.Series):
                    data = data.to_frame(name=tickers[0])
                
                # Verificamos si la descarga fue exitosa
                if data.empty:
                    st.error("No se pudieron descargar los datos. Revisa los tickers o el rango de fechas.")
                    st.stop() # Detiene la ejecución si no hay datos
                
                # Eliminamos filas con valores nulos que puedan aparecer
                data.dropna(inplace=True)

            except Exception as e:
                st.error(f"Ocurrió un error al descargar los datos: {e}")
                st.stop()

        st.success("✅ Datos descargados exitosamente.")

        # --- SECCIÓN 1: ANÁLISIS DE ACTIVOS INDIVIDUALES ---
        st.header("1. Análisis de Activos Individuales")

        # Calculamos los retornos diarios
        retornos_diarios = data.pct_change().dropna()

        # Calculamos métricas anualizadas para cada activo
        # 252 es el número aproximado de días de trading en un año
        rentabilidad_anual = retornos_diarios.mean() * 252
        volatilidad_anual = retornos_diarios.std() * np.sqrt(252)
        ratio_sharpe = rentabilidad_anual / volatilidad_anual

        # Creamos un DataFrame para mostrar el resumen
        resumen_activos = pd.DataFrame({
            'Rentabilidad Anual (%)': rentabilidad_anual * 100,
            'Volatilidad Anual (%)': volatilidad_anual * 100,
            'Ratio de Sharpe': ratio_sharpe
        })

        st.subheader("Métricas de Riesgo y Rentabilidad")
        st.dataframe(resumen_activos.style.format("{:.2f}"))

        # Gráfico de precios normalizados para comparar el rendimiento
        st.subheader("Rendimiento Histórico Normalizado")
        precios_normalizados = (data / data.iloc[0] * 100) # Todos empiezan en 100
        st.line_chart(precios_normalizados)

        # Matriz de correlación (solo si hay más de un activo)
        if len(tickers) > 1:
            st.subheader("Matriz de Correlación")
            fig_corr, ax_corr = plt.subplots()
            sns.heatmap(retornos_diarios.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
            st.pyplot(fig_corr)
            st.info("La correlación mide cómo se mueven los activos entre sí. Valores cercanos a -1 indican diversificación, mientras que valores cercanos a 1 indican que se mueven juntos.")

        # --- SECCIÓN 2: OPTIMIZACIÓN DE PORTAFOLIO (MONTE CARLO) ---
        # Esta sección solo tiene sentido si tenemos más de un activo para combinar
        if len(tickers) > 1:
            st.header("2. Optimización del Portafolio con Simulación Monte Carlo")

            # Preparamos las variables para la simulación
            num_activos = len(tickers)
            resultados_simulacion = np.zeros((3, num_simulaciones)) # [Riesgo, Retorno, Sharpe]
            pesos_portafolios = []
            
            # Matriz de covarianza anualizada
            cov_matrix_anual = retornos_diarios.cov() * 252

            with st.spinner(f"Realizando {num_simulaciones} simulaciones..."):
                for i in range(num_simulaciones):
                    # 1. Generar pesos aleatorios
                    pesos = np.random.random(num_activos)
                    # 2. Normalizar los pesos para que sumen 1
                    pesos /= np.sum(pesos)
                    pesos_portafolios.append(pesos)

                    # 3. Calcular retorno y riesgo del portafolio con estos pesos
                    retorno_portafolio = np.sum(rentabilidad_anual * pesos)
                    riesgo_portafolio = np.sqrt(np.dot(pesos.T, np.dot(cov_matrix_anual, pesos)))

                    # 4. Guardar resultados
                    resultados_simulacion[0, i] = riesgo_portafolio
                    resultados_simulacion[1, i] = retorno_portafolio
                    # 5. Calcular y guardar el Ratio de Sharpe
                    resultados_simulacion[2, i] = retorno_portafolio / riesgo_portafolio

            # Localizar el portafolio con el MÁXIMO Ratio de Sharpe
            max_sharpe_idx = np.argmax(resultados_simulacion[2])
            
            # Obtener los datos de ese portafolio óptimo
            riesgo_optimo = resultados_simulacion[0, max_sharpe_idx]
            retorno_optimo = resultados_simulacion[1, max_sharpe_idx]
            max_sharpe_ratio = resultados_simulacion[2, max_sharpe_idx]
            pesos_optimos = pesos_portafolios[max_sharpe_idx]

            # --- SECCIÓN 3: RESULTADOS DEL PORTAFOLIO ÓPTIMO ---
            st.header("3. Resultados del Portafolio Óptimo (Máximo Ratio de Sharpe)")

            # Mostramos las métricas clave del mejor portafolio encontrado
            col1, col2, col3 = st.columns(3)
            col1.metric("Rentabilidad Anual", f"{retorno_optimo*100:.2f}%")
            col2.metric("Volatilidad Anual", f"{riesgo_optimo*100:.2f}%")
            col3.metric("Ratio de Sharpe", f"{max_sharpe_ratio:.2f}")

            # Dividimos la sección de resultados en dos columnas
            col_graf, col_pesos = st.columns([2, 1]) # La primera columna es el doble de ancha

            with col_graf:
                # Gráfico de la Frontera Eficiente
                st.subheader("Frontera Eficiente")
                fig_frontera, ax_frontera = plt.subplots(figsize=(10, 6))
                scatter = ax_frontera.scatter(
                    resultados_simulacion[0, :], # Eje X: Riesgo
                    resultados_simulacion[1, :], # Eje Y: Retorno
                    c=resultados_simulacion[2, :], # Color: Sharpe Ratio
                    cmap='viridis',
                    marker='o',
                    s=10,
                    alpha=0.5
                )
                # Marcar el portafolio óptimo con una estrella roja
                ax_frontera.scatter(
                    riesgo_optimo,
                    retorno_optimo,
                    marker='*',
                    color='r',
                    s=200,
                    label='Portafolio Óptimo'
                )
                ax_frontera.set_title('Frontera Eficiente y Portafolio Óptimo')
                ax_frontera.set_xlabel('Volatilidad (Riesgo)')
                ax_frontera.set_ylabel('Rentabilidad Esperada')
                ax_frontera.legend()
                fig_frontera.colorbar(scatter, label='Ratio de Sharpe')
                st.pyplot(fig_frontera)

            with col_pesos:
                # El "Factor Distintivo": un gráfico de pastel claro con la asignación
                st.subheader("Distribución Óptima de Activos")
                df_pesos = pd.DataFrame({'Activo': tickers, 'Peso': pesos_optimos})
                
                fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
                ax_pie.pie(
                    df_pesos['Peso'],
                    labels=df_pesos['Activo'],
                    autopct='%1.1f%%',
                    startangle=90,
                    pctdistance=0.85
                )
                # Dibuja un círculo en el centro para hacerlo un "donut chart"
                centre_circle = plt.Circle((0,0),0.70,fc='white')
                fig_pie.gca().add_artist(centre_circle)
                
                ax_pie.axis('equal')  # Asegura que el pastel sea un círculo
                plt.tight_layout()
                st.pyplot(fig_pie)
                
                # También mostramos los pesos en una tabla
                st.dataframe(df_pesos.style.format({'Peso': '{:.2%}'}), use_container_width=True)

# 5. MENSAJE INICIAL SI NO SE HA PRESIONADO EL BOTÓN
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

# --- PIE DE PÁGINA ---
st.markdown("---")
st.markdown("Desarrollado por el **Grupo 10** | Ingeniería Financiera")
