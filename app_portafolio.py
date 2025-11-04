import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date

# --- Configuración de la Página y Título ---
st.set_page_config(page_title="Análisis de Portafolio", layout="wide")
st.title("📈 Analizador de Rentabilidad y Riesgo de Activos")
st.header("Compara el rendimiento de hasta dos activos financieros")
st.write("Introduce los tickers de los activos (ej. AAPL, MSFT, GOOGL, BTC-USD) y un rango de fechas para analizar.")

# --- Barra Lateral (Sidebar) para Entradas de Usuario ---
st.sidebar.header("Parámetros de Análisis")
start_date = st.sidebar.date_input('Fecha de Inicio', date(2020, 1, 1))
end_date = st.sidebar.date_input('Fecha de Fin', date.today())
ticker1 = st.sidebar.text_input('Ticker 1', 'AAPL').upper()
ticker2 = st.sidebar.text_input('Ticker 2 (Opcional)', 'MSFT').upper()
run_button = st.sidebar.button("Analizar Activos")

# --- Lógica Principal de la App ---
if run_button:
    # 1. Crear lista de tickers y descargar datos
    tickers_list = [ticker1]
    if ticker2:
        tickers_list.append(ticker2)

    st.header(f"Análisis Comparativo para: {', '.join(tickers_list)}")

    try:
        data = yf.download(tickers_list, start=start_date, end=end_date)['Adj Close']
        
        # Manejar el caso de un solo ticker para que la estructura de datos sea consistente
        if len(tickers_list) == 1:
            data = data.to_frame(name=tickers_list[0])

        if data.empty:
            st.error("No se encontraron datos para los tickers o el rango de fechas seleccionado. Por favor, verifica.")
        else:
            st.success("Datos descargados correctamente.")

            # 2. Cálculos de Rentabilidad y Riesgo
            daily_returns = data.pct_change().dropna()
            cumulative_returns = (1 + daily_returns).cumprod() - 1

            # 3. Mostrar Métricas Clave
            st.subheader("Métricas Clave (Anualizadas)")
            cols = st.columns(len(tickers_list))
            for i, ticker in enumerate(tickers_list):
                with cols[i]:
                    st.markdown(f"### {ticker}")
                    
                    annual_return = daily_returns[ticker].mean() * 252
                    st.metric(label="Rentabilidad Anualizada", value=f"{annual_return:.2%}")
                    
                    annual_volatility = daily_returns[ticker].std() * (252**0.5)
                    st.metric(label="Volatilidad Anualizada (Riesgo)", value=f"{annual_volatility:.2%}")
                    
                    sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
                    st.metric(label="Ratio de Sharpe", value=f"{sharpe_ratio:.2f}")

            # 4. Visualizaciones
            st.subheader("Evolución de la Rentabilidad Acumulada")
            st.line_chart(cumulative_returns)

            st.subheader("Datos Históricos (Precio de Cierre Ajustado)")
            st.dataframe(data.style.format("{:.2f}"))
            
            st.subheader("Rentabilidades Diarias")
            st.dataframe(daily_returns.style.format("{:.2%}"))

    except Exception as e:
        st.error(f"Ocurrió un error: {e}. Asegúrate de que los tickers son válidos.")
