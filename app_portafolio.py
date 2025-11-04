# Importamos las librerías necesarias
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Análisis de Portafolio", layout="wide")

# --- TÍTULO Y DESCRIPCIÓN ---
st.title('📊 Analizador de Rentabilidad y Riesgo de Acciones')
st.markdown("""
Esta aplicación realiza un análisis básico de la rentabilidad y el riesgo para una o dos acciones globales.
- **Instrucciones:** Ingresa los tickers de las acciones separados por coma (ej. `AAPL,MSFT`).
- **Fuente de Datos:** Yahoo Finance.
""")

# --- BARRA LATERAL (SIDEBAR) PARA ENTRADAS DEL USUARIO ---
st.sidebar.header('Parámetros de Entrada')

# Input para los tickers
tickers_input = st.sidebar.text_input('Ingresa los Tickers (separados por coma)', 'GOOGL,TSLA')

# Selección de fechas
start_date = st.sidebar.date_input('Fecha de Inicio', pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input('Fecha de Fin', pd.to_datetime('today'))

# Botón para ejecutar el análisis
analyze_button = st.sidebar.button('Ejecutar Análisis')


# --- LÓGICA PRINCIPAL DEL ANÁLISIS ---
if analyze_button:
    # 1. Procesar los tickers de entrada
    # .strip() elimina espacios en blanco y .upper() los convierte a mayúsculas
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]
    
    # Validamos que no se ingresen más de 2 tickers
    if len(tickers) == 0 or tickers == ['']:
        st.warning('Por favor, ingresa al menos un ticker.')
    elif len(tickers) > 2:
        st.error('Análisis limitado a un máximo de 2 tickers.')
    else:
        st.header(f"Análisis para: {', '.join(tickers)}")

        try:
            # 2. Descargar los datos de Yahoo Finance
            # Usamos 'Adj Close' porque ajusta el precio a dividendos y splits
            @st.cache_data # Usamos caché para no descargar los mismos datos repetidamente
            def descargar_datos(tickers, start, end):
                return yf.download(tickers, start=start, end=end)['Adj Close']

            data = descargar_datos(tickers, start_date, end_date)

            # Si solo se pide un ticker, yfinance no devuelve un DataFrame sino una Serie.
            # Lo convertimos a DataFrame para que el resto del código funcione igual.
            if len(tickers) == 1:
                data = data.to_frame(name=tickers[0])
            
            if data.empty:
                st.error("No se pudieron descargar los datos. Verifica los tickers y el rango de fechas.")
            else:
                # --- CÁLCULOS FINANCIEROS ---

                # 3. Calcular los retornos diarios
                # El retorno es el cambio porcentual del precio de un día para otro
                retornos_diarios = data.pct_change().dropna()

                # 4. Calcular los retornos acumulados
                # Esto nos muestra cómo habría crecido una inversión inicial de 1 dólar
                retornos_acumulados = (1 + retornos_diarios).cumprod() - 1

                # 5. Métricas de Rentabilidad y Riesgo Anualizadas
                # (Asumimos 252 días de trading en un año)
                rentabilidad_anualizada = retornos_diarios.mean() * 252
                volatilidad_anualizada = retornos_diarios.std() * np.sqrt(252)

                # --- VISUALIZACIÓN DE RESULTADOS ---

                st.subheader('Evolución de Precios (Normalizados)')
                # Normalizamos los precios para poder compararlos en la misma escala
                data_normalizada = data / data.iloc[0]
                fig_precios = px.line(data_normalizada, title='Rendimiento de Precios (Inicio = 1)')
                st.plotly_chart(fig_precios, use_container_width=True)
                
                st.subheader('Retornos Acumulados')
                fig_acumulados = px.line(retornos_acumulados, title='Crecimiento de la Inversión')
                st.plotly_chart(fig_acumulados, use_container_width=True)

                st.subheader('Métricas Clave Anualizadas')
                
                # Creamos un DataFrame para mostrar las métricas de forma ordenada
                metricas = pd.DataFrame({
                    'Rentabilidad Anualizada': rentabilidad_anualizada,
                    'Volatilidad Anualizada (Riesgo)': volatilidad_anualizada
                })
                # Formateamos los números como porcentajes
                st.dataframe(metricas.style.format("{:.2%}"))

                # Solo calculamos y mostramos la correlación si hay 2 tickers
                if len(tickers) == 2:
                    st.subheader('Análisis de Correlación')
                    correlacion = retornos_diarios.corr()
                    st.write("La correlación mide cómo se mueven dos activos entre sí. Un valor cercano a 1 significa que se mueven juntos; cercano a -1, que se mueven en direcciones opuestas.")
                    
                    fig_corr = px.imshow(correlacion, text_auto=True, title=f'Correlación entre {tickers[0]} y {tickers[1]}')
                    st.plotly_chart(fig_corr, use_container_width=True)

                # Mostramos los datos crudos en un expansor para no ocupar mucho espacio
                with st.expander("Ver datos descargados"):
                    st.dataframe(data)

        except Exception as e:
            st.error(f"Ocurrió un error: {e}")
