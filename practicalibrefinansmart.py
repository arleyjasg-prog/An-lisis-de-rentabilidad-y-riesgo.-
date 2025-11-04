# -*- coding: utf-8 -*-
"""
FinanSmart - Análisis de Portafolio de Inversión
Aplicación Streamlit para análisis financiero
"""

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="FinanSmart - Análisis de Portafolio",
    page_icon="📊",
    layout="wide"
)

# Título principal
st.title("📊 FinanSmart - Análisis de Portafolio de Inversión")
st.markdown("---")

# Sidebar para configuración
st.sidebar.header("⚙️ Configuración")

# Selección de tickers
default_tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]
tickers_input = st.sidebar.text_input(
    "Tickers (separados por comas)",
    value=",".join(default_tickers)
)
tickers = [t.strip().upper() for t in tickers_input.split(",")]

# Selección de fechas
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Fecha inicio", value=pd.to_datetime("2020-01-01"))
with col2:
    end_date = st.date_input("Fecha fin", value=pd.to_datetime("2023-12-31"))

# Número de simulaciones
num_portfolios = st.sidebar.slider(
    "Número de simulaciones",
    min_value=1000,
    max_value=50000,
    value=10000,
    step=1000
)

# Botón para ejecutar análisis
if st.sidebar.button("🚀 Ejecutar Análisis", type="primary"):
    
    with st.spinner("Descargando datos..."):
        try:
            # Descarga de datos
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
            
            if data.empty:
                st.error("No se pudieron descargar datos. Verifica los tickers y las fechas.")
                st.stop()
            
            st.success("✅ Datos descargados exitosamente")
            
            # Sección 1: Datos descargados
            st.header("1️⃣ Datos Históricos de Precios")
            st.dataframe(data.head(10), use_container_width=True)
            
            # Gráfico de evolución de precios
            st.subheader("Evolución de Precios Ajustados")
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            data.plot(ax=ax1)
            ax1.set_title('Evolución de precios ajustados')
            ax1.set_xlabel('Fecha')
            ax1.set_ylabel('Precio ($)')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
            
            # Cálculo de retornos
            returns = data.pct_change().dropna()
            
            # Sección 2: Retornos
            st.header("2️⃣ Análisis de Retornos Diarios")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Estadísticas Descriptivas")
                st.dataframe(returns.describe(), use_container_width=True)
            
            with col2:
                st.subheader("Retornos Diarios")
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                returns.plot(ax=ax2, alpha=0.7)
                ax2.set_title('Retornos diarios')
                ax2.set_xlabel('Fecha')
                ax2.set_ylabel('Retorno')
                ax2.legend(loc='best')
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)
            
            # Sección 3: Correlación
            st.header("3️⃣ Matriz de Correlación")
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', ax=ax3, center=0)
            ax3.set_title('Matriz de correlación del portafolio')
            st.pyplot(fig3)
            
            # Cálculo de métricas anualizadas
            mean_returns = returns.mean() * 252
            risk = returns.std() * np.sqrt(252)
            
            # Sección 4: Métricas
            st.header("4️⃣ Métricas de Riesgo y Retorno Anualizadas")
            metrics_df = pd.DataFrame({
                'Rendimiento Anual': mean_returns,
                'Riesgo (Volatilidad)': risk
            })
            st.dataframe(metrics_df.style.format("{:.2%}"), use_container_width=True)
            
            # Portafolio con pesos iguales
            st.subheader("Portafolio con Pesos Iguales")
            weights_equal = np.array([1/len(tickers)] * len(tickers))
            portfolio_return = np.dot(weights_equal, mean_returns)
            portfolio_risk = np.sqrt(np.dot(weights_equal.T, np.dot(returns.cov() * 252, weights_equal)))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rendimiento Esperado", f"{portfolio_return:.2%}")
            with col2:
                st.metric("Riesgo (Volatilidad)", f"{portfolio_risk:.2%}")
            
            # Sección 5: Simulación Monte Carlo
            st.header("5️⃣ Simulación de Portafolios (Monte Carlo)")
            
            with st.spinner(f"Simulando {num_portfolios:,} portafolios..."):
                results = np.zeros((3, num_portfolios))
                
                for i in range(num_portfolios):
                    weights = np.random.random(len(tickers))
                    weights /= np.sum(weights)
                    
                    ret = np.dot(weights, mean_returns)
                    risk_calc = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
                    sharpe = ret / risk_calc if risk_calc != 0 else 0
                    
                    results[0, i] = risk_calc
                    results[1, i] = ret
                    results[2, i] = sharpe
            
            # Gráfico de frontera eficiente
            st.subheader("Frontera Eficiente")
            fig4, ax4 = plt.subplots(figsize=(12, 8))
            scatter = ax4.scatter(
                results[0, :],
                results[1, :],
                c=results[2, :],
                cmap='viridis',
                alpha=0.5,
                s=10
            )
            ax4.set_xlabel('Riesgo (Volatilidad)')
            ax4.set_ylabel('Retorno Esperado')
            ax4.set_title('Frontera Eficiente Simulada')
            plt.colorbar(scatter, label='Índice de Sharpe', ax=ax4)
            ax4.grid(True, alpha=0.3)
            
            # Marcar el portafolio óptimo
            max_sharpe_idx = np.argmax(results[2])
            ax4.scatter(
                results[0, max_sharpe_idx],
                results[1, max_sharpe_idx],
                c='red',
                s=200,
                marker='*',
                edgecolors='black',
                label='Portafolio Óptimo'
            )
            ax4.legend()
            st.pyplot(fig4)
            
            # Portafolio óptimo
            st.subheader("🏆 Portafolio Óptimo (Máximo Sharpe)")
            mejor_riesgo, mejor_retorno, mejor_sharpe = results[:, max_sharpe_idx]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Índice de Sharpe", f"{mejor_sharpe:.2f}")
            with col2:
                st.metric("Retorno Esperado", f"{mejor_retorno:.2%}")
            with col3:
                st.metric("Riesgo Asociado", f"{mejor_riesgo:.2%}")
            
            # Exportar resultados
            st.header("6️⃣ Exportar Resultados")
            df_resultados = pd.DataFrame(
                results.T,
                columns=['Riesgo', 'Retorno', 'Sharpe']
            )
            
            csv = df_resultados.to_csv(index=False)
            st.download_button(
                label="📥 Descargar resultados CSV",
                data=csv,
                file_name="resultados_portafolio.csv",
                mime="text/csv"
            )
            
            st.dataframe(df_resultados.head(10), use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error al procesar los datos: {str(e)}")
            st.info("Verifica que los tickers sean válidos y que haya datos disponibles para el rango de fechas seleccionado.")

else:
    st.info("👈 Configura los parámetros en el panel lateral y presiona 'Ejecutar Análisis'")
    
    # Mostrar información de ayuda
    with st.expander("ℹ️ Información sobre la aplicación"):
        st.markdown("""
        ### ¿Qué hace esta aplicación?
        
        Esta aplicación realiza un análisis completo de portafolio de inversión utilizando:
        
        1. **Descarga de datos históricos** de precios de acciones usando Yahoo Finance
        2. **Cálculo de retornos diarios** y métricas estadísticas
        3. **Análisis de correlación** entre activos
        4. **Simulación Monte Carlo** de miles de portafolios posibles
        5. **Identificación del portafolio óptimo** según el índice de Sharpe
        
        ### ¿Cómo usar?
        
        1. Ingresa los tickers de las acciones (ej: AAPL, MSFT, GOOGL)
        2. Selecciona el rango de fechas
        3. Ajusta el número de simulaciones
        4. Presiona "Ejecutar Análisis"
        5. Descarga los resultados en CSV
        
        ### Índice de Sharpe
        
        Mide el retorno ajustado por riesgo. Un valor más alto indica mejor relación riesgo-retorno.
        """)

# Footer
st.markdown("---")
st.markdown("Desarrollado con ❤️ usando Streamlit | Datos: Yahoo Finance")
