# =============================================================================
# LIBRERÍAS
# =============================================================================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Analizador de Portafolios Pro",
    page_icon="📈",
    layout="wide"
)

# =============================================================================
# FUNCIONES AUXILIARES (PARA UN CÓDIGO MÁS LIMPIO)
# =============================================================================

@st.cache_data # Clave para el rendimiento: no vuelve a descargar datos si los inputs no cambian
def cargar_datos(tickers, start_date, end_date):
    """
    Descarga los datos de precios de cierre ajustados de Yahoo Finance para una lista de tickers.
    También descarga el S&P 500 (^GSPC) como benchmark del mercado.
    """
    try:
        # Añadimos el S&P 500 a la lista para el cálculo de Beta
        tickers_con_benchmark = tickers + ['^GSPC']
        data = yf.download(tickers_con_benchmark, start=start_date, end=end_date)['Adj Close']
        
        # Eliminar filas donde todos los valores son nulos (por ejemplo, fines de semana)
        data.dropna(how='all', inplace=True)
        # Rellenar valores nulos puntuales (ej. por festivos de un mercado específico)
        data.fillna(method='ffill', inplace=True)
        
        if data.empty:
            return None, None
        
        # Separamos los datos del benchmark y de los activos
        benchmark_data = data[['^GSPC']]
        asset_data = data[tickers]
        
        return asset_data, benchmark_data
    except Exception as e:
        st.error(f"Error al descargar los datos: {e}")
        return None, None

def calcular_metricas(retornos, retornos_benchmark):
    """Calcula las métricas clave de rentabilidad y riesgo."""
    
    # Métricas anualizadas (252 días de trading)
    dias_trading = 252
    
    # Rentabilidad anualizada
    rentabilidad_anualizada = retornos.mean() * dias_trading
    
    # Volatilidad anualizada (riesgo)
    volatilidad_anualizada = retornos.std() * np.sqrt(dias_trading)
    
    # Ratio de Sharpe (asumiendo tasa libre de riesgo del 1%)
    tasa_libre_riesgo = 0.01
    ratio_sharpe = (rentabilidad_anualizada - tasa_libre_riesgo) / volatilidad_anualizada
    
    # Beta (respecto al S&P 500)
    # Covarianza(activo, mercado) / Varianza(mercado)
    covarianza = retornos.cov(retornos_benchmark.iloc[:,0])
    varianza_mercado = retornos_benchmark.iloc[:,0].var()
    beta = covarianza / varianza_mercado
    
    # Máximo Drawdown (peor caída)
    retornos_acumulados = (1 + retornos).cumprod()
    pico_anterior = retornos_acumulados.cummax()
    drawdown = (retornos_acumulados - pico_anterior) / pico_anterior
    max_drawdown = drawdown.min()

    metricas = pd.DataFrame({
        'Rentabilidad Anualizada': rentabilidad_anualizada,
        'Volatilidad Anualizada': volatilidad_anualizada,
        'Ratio de Sharpe': ratio_sharpe,
        'Beta vs. S&P 500': beta,
        'Máximo Drawdown': max_drawdown
    })
    
    return metricas.T # .T transpone la tabla para mejor visualización

# =============================================================================
# INTERFAZ DE USUARIO (UI)
# =============================================================================

st.title('📈 Analizador de Portafolios Pro')
st.markdown("Una herramienta avanzada para evaluar la rentabilidad y el riesgo de activos globales.")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ Parámetros de Análisis")
    
    tickers_input = st.text_area(
        "Ingrese los Tickers (separados por comas)",
        "AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA",
        help="Ej: para Apple, Microsoft y Google, ingrese `AAPL,MSFT,GOOGL`"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Fecha de Inicio", pd.to_datetime('2021-01-01'))
    with col2:
        end_date = st.date_input("Fecha de Fin", pd.to_datetime('today'))
        
    analyze_button = st.button("🚀 Analizar Activos", type="primary")

# =============================================================================
# LÓGICA PRINCIPAL Y VISUALIZACIÓN
# =============================================================================
if analyze_button:
    # Limpiar y validar tickers
    tickers = sorted([ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()])

    if not tickers:
        st.warning("Por favor, ingrese al menos un ticker para analizar.")
    else:
        # Mostrar un spinner mientras se cargan los datos
        with st.spinner(f"Cargando datos para: {', '.join(tickers)}..."):
            asset_data, benchmark_data = cargar_datos(tickers, start_date, end_date)

        if asset_data is None or asset_data.empty:
            st.error("No se pudieron obtener los datos. Verifique los tickers y el rango de fechas.")
        else:
            st.success("¡Datos cargados correctamente!")
            
            # --- CÁLCULOS ---
            retornos_diarios = asset_data.pct_change().dropna()
            retornos_benchmark = benchmark_data.pct_change().dropna()
            retornos_acumulados = (1 + retornos_diarios).cumprod() - 1

            # Calcular métricas
            df_metricas = calcular_metricas(retornos_diarios, retornos_benchmark)

            # --- PESTAÑAS PARA ORGANIZAR LA INFO ---
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Visión General", 
                "🔬 Análisis de Riesgo", 
                "⚖️ Comparativa y Correlación", 
                "💾 Datos Crudos"
            ])

            with tab1:
                st.header("Resumen de Métricas Clave")
                st.markdown("Aquí se muestran los indicadores más importantes de rendimiento y riesgo anualizados.")
                st.dataframe(df_metricas.style.format({
                    'Rentabilidad Anualizada': '{:.2%}',
                    'Volatilidad Anualizada': '{:.2%}',
                    'Ratio de Sharpe': '{:.2f}',
                    'Beta vs. S&P 500': '{:.2f}',
                    'Máximo Drawdown': '{:.2%}'
                }))

                st.header("Evolución de Precios (Normalizados)")
                st.markdown("Compara el rendimiento de los activos si todos hubieran empezado con un valor de 100.")
                precios_normalizados = (asset_data / asset_data.iloc[0]) * 100
                fig_precios = px.line(precios_normalizados, title="Rendimiento de Precios Normalizados")
                st.plotly_chart(fig_precios, use_container_width=True)

            with tab2:
                st.header("Análisis Detallado de Riesgo")
                
                st.subheader("Volatilidad Móvil (30 días)")
                st.markdown("Muestra cómo ha cambiado el riesgo (volatilidad) de los activos a lo largo del tiempo.")
                volatilidad_movil = retornos_diarios.rolling(window=30).std() * np.sqrt(252)
                fig_vol_movil = px.line(volatilidad_movil, title="Volatilidad Anualizada Móvil (30 días)")
                st.plotly_chart(fig_vol_movil, use_container_width=True)
                
                st.subheader("Distribución de Retornos Diarios")
                st.markdown("Este histograma muestra la frecuencia de las ganancias y pérdidas diarias.")
                
                # Selector para el histograma
                ticker_seleccionado_hist = st.selectbox("Seleccione un activo para ver su distribución:", tickers)
                fig_hist = px.histogram(
                    retornos_diarios[ticker_seleccionado_hist], 
                    nbins=100, 
                    title=f'Distribución de Retornos Diarios para {ticker_seleccionado_hist}'
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with tab3:
                st.header("Análisis Comparativo")

                if len(tickers) > 1:
                    st.subheader("Mapa de Calor de Correlaciones")
                    st.markdown("Mide cómo se mueven los activos entre sí. Un valor de 1 indica un movimiento perfecto en la misma dirección; -1 en dirección opuesta.")
                    correlaciones = retornos_diarios.corr()
                    fig_corr = px.imshow(
                        correlaciones, 
                        text_auto=True, 
                        aspect="auto", 
                        title="Matriz de Correlación de Retornos Diarios"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)

                    st.subheader("Diagrama de Dispersión de Retornos")
                    st.markdown("Visualiza la relación directa entre los retornos diarios de dos activos.")
                    col_sel1, col_sel2 = st.columns(2)
                    with col_sel1:
                        ticker1 = st.selectbox("Eje X:", tickers, index=0)
                    with col_sel2:
                        ticker2 = st.selectbox("Eje Y:", tickers, index=min(1, len(tickers)-1))
                    
                    fig_scatter = px.scatter(
                        retornos_diarios, 
                        x=ticker1, 
                        y=ticker2,
                        title=f'Dispersión de Retornos Diarios: {ticker1} vs. {ticker2}',
                        trendline='ols', # Añade una línea de tendencia
                        trendline_color_override='red'
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info("Se necesitan al menos dos activos para realizar un análisis comparativo.")

            with tab4:
                st.header("Datos de Precios de Cierre Ajustados")
                st.dataframe(asset_data)
