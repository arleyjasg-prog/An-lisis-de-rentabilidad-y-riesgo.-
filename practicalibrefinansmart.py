import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(page_title="FinanSmart", page_icon="📊", layout="wide")

# Título
st.title("📊 FinanSmart - Análisis de Portafolio de Inversión")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Configuración")

# Input de tickers
ticker1 = st.sidebar.text_input("Ticker 1", "AAPL").upper()
ticker2 = st.sidebar.text_input("Ticker 2 (opcional)", "MSFT").upper()

# Crear lista de tickers
tickers = [ticker1]
if ticker2:
    tickers.append(ticker2)

# Fechas
start_date = st.sidebar.date_input("Fecha inicio", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("Fecha fin", pd.to_datetime("2023-12-31"))

# Número de simulaciones
num_portfolios = st.sidebar.slider("Simulaciones", 1000, 20000, 5000, 1000)

# Botón de análisis
if st.sidebar.button("🚀 Ejecutar Análisis", type="primary"):
    
    try:
        # Descargar datos
        with st.spinner("Descargando datos..."):
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
            
            if data.empty:
                st.error("No se pudieron descargar datos.")
                st.stop()
        
        st.success("✅ Datos descargados")
        
        # 1. PRECIOS HISTÓRICOS
        st.header("1️⃣ Precios Históricos")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        data.plot(ax=ax1)
        ax1.set_title('Evolución de Precios')
        ax1.set_xlabel('Fecha')
        ax1.set_ylabel('Precio ($)')
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        plt.close()
        
        # Calcular retornos
        returns = data.pct_change().dropna()
        
        # 2. ESTADÍSTICAS
        st.header("2️⃣ Estadísticas de Retornos")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Estadísticas Descriptivas")
            st.dataframe(returns.describe())
        
        with col2:
            st.subheader("Retornos Diarios")
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            returns.plot(ax=ax2, alpha=0.7)
            ax2.set_title('Retornos Diarios')
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            plt.close()
        
        # 3. CORRELACIÓN (solo si hay 2 tickers)
        if len(tickers) > 1:
            st.header("3️⃣ Correlación")
            fig3, ax3 = plt.subplots(figsize=(6, 5))
            sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', ax=ax3, center=0)
            ax3.set_title('Matriz de Correlación')
            st.pyplot(fig3)
            plt.close()
        
        # Métricas anualizadas
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        risk = returns.std() * np.sqrt(252)
        
        # 4. MÉTRICAS INDIVIDUALES
        st.header("4️⃣ Métricas Anualizadas")
        metrics_df = pd.DataFrame({
            'Rendimiento Anual': mean_returns,
            'Volatilidad': risk
        })
        st.dataframe(metrics_df.style.format("{:.2%}"))
        
        # 5. SIMULACIÓN MONTE CARLO
        st.header("5️⃣ Simulación de Portafolios")
        
        with st.spinner(f"Simulando {num_portfolios:,} portafolios..."):
            results = np.zeros((3, num_portfolios))
            
            for i in range(num_portfolios):
                # Generar pesos aleatorios
                weights = np.random.random(len(tickers))
                weights /= weights.sum()
                
                # Calcular retorno del portafolio
                portfolio_return = np.dot(weights, mean_returns)
                
                # Calcular riesgo del portafolio
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                # Calcular Sharpe (asumiendo tasa libre de riesgo = 0)
                sharpe = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
                
                results[0, i] = portfolio_risk
                results[1, i] = portfolio_return
                results[2, i] = sharpe
        
        # Gráfico de frontera eficiente
        st.subheader("Frontera Eficiente")
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        scatter = ax4.scatter(results[0, :], results[1, :], 
                            c=results[2, :], cmap='viridis', 
                            alpha=0.5, s=10)
        ax4.set_xlabel('Riesgo (Volatilidad)')
        ax4.set_ylabel('Retorno Esperado')
        ax4.set_title('Frontera Eficiente')
        plt.colorbar(scatter, label='Índice Sharpe', ax=ax4)
        
        # Marcar portafolio óptimo
        max_sharpe_idx = np.argmax(results[2])
        ax4.scatter(results[0, max_sharpe_idx], results[1, max_sharpe_idx],
                   c='red', s=300, marker='*', edgecolors='black',
                   label='Portafolio Óptimo')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        st.pyplot(fig4)
        plt.close()
        
        # 6. PORTAFOLIO ÓPTIMO
        st.header("6️⃣ 🏆 Portafolio Óptimo")
        mejor_riesgo = results[0, max_sharpe_idx]
        mejor_retorno = results[1, max_sharpe_idx]
        mejor_sharpe = results[2, max_sharpe_idx]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sharpe", f"{mejor_sharpe:.2f}")
        with col2:
            st.metric("Retorno", f"{mejor_retorno:.2%}")
        with col3:
            st.metric("Riesgo", f"{mejor_riesgo:.2%}")
        
        # 7. EXPORTAR
        st.header("7️⃣ Exportar Resultados")
        df_results = pd.DataFrame(results.T, columns=['Riesgo', 'Retorno', 'Sharpe'])
        csv = df_results.to_csv(index=False)
        st.download_button("📥 Descargar CSV", csv, "portafolio.csv", "text/csv")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Verifica los tickers y las fechas")

else:
    st.info("👈 Configura los parámetros y presiona 'Ejecutar Análisis'")
    
    with st.expander("ℹ️ Guía de uso"):
        st.markdown("""
        ### ¿Cómo funciona?
        
        1. **Ingresa 1 o 2 tickers** (ej: AAPL, MSFT, GOOGL)
        2. **Selecciona fechas** de análisis
        3. **Ajusta simulaciones** (más = mayor precisión)
        4. **Ejecuta el análisis**
        
        ### ¿Qué hace?
        - Descarga precios históricos
        - Calcula retornos y riesgo
        - Simula miles de portafolios
        - Encuentra el óptimo (mejor Sharpe)
        
        ### Índice de Sharpe
        Mide rentabilidad ajustada por riesgo.  
        **Más alto = mejor relación riesgo/retorno**
        """)

st.markdown("---")
st.markdown("Desarrollado para ITM - Análisis de Costos y Presupuestos")
