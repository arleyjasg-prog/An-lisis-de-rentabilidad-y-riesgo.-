# -*- coding: utf-8 -*-
"""
FinanSmart - Análisis de Portafolio de Inversión
Aplicación Streamlit para análisis financiero con datos reales de Yahoo Finance
Desarrollado para: ITM - Análisis de Costos y Presupuestos
"""

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ==================== CONFIGURACIÓN DE LA PÁGINA ====================
st.set_page_config(
    page_title="FinanSmart - Análisis de Portafolio",
    page_icon="📊",
    layout="wide"
)

# ==================== TÍTULO PRINCIPAL ====================
st.title("📊 FinanSmart - Análisis de Portafolio de Inversión")
st.markdown("### Análisis de Rentabilidad y Riesgo con Datos Reales")
st.markdown("---")

# ==================== SIDEBAR - CONFIGURACIÓN ====================
st.sidebar.header("⚙️ Configuración del Análisis")

# Input de tickers
st.sidebar.subheader("Selección de Activos")
ticker1 = st.sidebar.text_input("Ticker 1", "AAPL", help="Ejemplo: AAPL, MSFT, GOOGL").upper().strip()
ticker2 = st.sidebar.text_input("Ticker 2 (opcional)", "MSFT", help="Dejar vacío para analizar solo 1 activo").upper().strip()

# Crear lista de tickers
tickers = [ticker1]
if ticker2:
    tickers.append(ticker2)

st.sidebar.info(f"📌 Analizando: {', '.join(tickers)}")

# Selección de fechas
st.sidebar.subheader("Período de Análisis")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Fecha inicio", pd.to_datetime("2020-01-01"))
with col2:
    end_date = st.date_input("Fecha fin", pd.to_datetime("2023-12-31"))

# Número de simulaciones para Monte Carlo
st.sidebar.subheader("Parámetros de Simulación")
num_portfolios = st.sidebar.slider(
    "Número de simulaciones",
    min_value=1000,
    max_value=20000,
    value=5000,
    step=1000,
    help="Mayor número = más precisión pero más tiempo de cálculo"
)

# Tasa libre de riesgo (opcional)
risk_free_rate = st.sidebar.number_input(
    "Tasa libre de riesgo (%)",
    min_value=0.0,
    max_value=10.0,
    value=0.0,
    step=0.1,
    help="Para cálculo del ratio de Sharpe"
) / 100

# ==================== BOTÓN DE ANÁLISIS ====================
if st.sidebar.button("🚀 Ejecutar Análisis", type="primary"):
    
    try:
        # ==================== DESCARGA DE DATOS ====================
        with st.spinner("📡 Descargando datos de Yahoo Finance..."):
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
            
            # Verificar si se descargaron datos
            if data.empty:
                st.error("❌ No se pudieron descargar datos. Verifica los tickers y las fechas.")
                st.stop()
            
            # Si solo hay un ticker, convertir a DataFrame
            if len(tickers) == 1:
                data = pd.DataFrame(data, columns=[ticker1])
        
        st.success(f"✅ Datos descargados exitosamente: {len(data)} días de cotización")
        
        # ==================== 1. PRECIOS HISTÓRICOS ====================
        st.header("1️⃣ Evolución de Precios Históricos")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            for ticker in tickers:
                if ticker in data.columns:
                    ax1.plot(data.index, data[ticker], label=ticker, linewidth=2)
            ax1.set_title('Evolución de Precios de Cierre Ajustados', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Fecha', fontsize=12)
            ax1.set_ylabel('Precio (USD)', fontsize=12)
            ax1.legend(loc='best', fontsize=10)
            ax1.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close()
        
        with col2:
            st.subheader("📊 Últimos Precios")
            ultimo_precio = data.iloc[-1]
            for ticker in tickers:
                if ticker in data.columns:
                    st.metric(
                        label=ticker,
                        value=f"${ultimo_precio[ticker]:.2f}",
                        delta=f"{((data[ticker].iloc[-1] / data[ticker].iloc[0] - 1) * 100):.2f}%"
                    )
        
        # ==================== CÁLCULO DE RETORNOS ====================
        returns = data.pct_change().dropna()
        
        # ==================== 2. ANÁLISIS DE RETORNOS ====================
        st.header("2️⃣ Análisis de Retornos Diarios")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Estadísticas Descriptivas")
            stats = returns.describe()
            st.dataframe(stats.style.format("{:.4f}"), use_container_width=True)
        
        with col2:
            st.subheader("📉 Distribución de Retornos")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            for ticker in tickers:
                if ticker in returns.columns:
                    ax2.plot(returns.index, returns[ticker], alpha=0.7, label=ticker, linewidth=1)
            ax2.set_title('Retornos Diarios', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Fecha')
            ax2.set_ylabel('Retorno')
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()
        
        # ==================== 3. MATRIZ DE CORRELACIÓN ====================
        if len(tickers) > 1:
            st.header("3️⃣ Análisis de Correlación")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                fig3, ax3 = plt.subplots(figsize=(6, 5))
                correlation = returns.corr()
                sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax3, 
                           center=0, vmin=-1, vmax=1, square=True,
                           fmt='.3f', linewidths=1)
                ax3.set_title('Matriz de Correlación', fontsize=12, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig3)
                plt.close()
            
            with col2:
                st.subheader("📊 Interpretación")
                corr_value = correlation.iloc[0, 1] if len(correlation) > 1 else 0
                st.metric("Correlación", f"{corr_value:.3f}")
                
                if corr_value > 0.7:
                    st.info("🔵 **Alta correlación positiva**: Los activos tienden a moverse juntos")
                elif corr_value > 0.3:
                    st.info("🟢 **Correlación moderada**: Cierta relación entre movimientos")
                elif corr_value > -0.3:
                    st.info("🟡 **Baja correlación**: Movimientos independientes (buena diversificación)")
                else:
                    st.info("🟠 **Correlación negativa**: Los activos tienden a moverse en direcciones opuestas")
        
        # ==================== MÉTRICAS ANUALIZADAS ====================
        mean_returns = returns.mean() * 252  # Anualizar retornos
        cov_matrix = returns.cov() * 252     # Anualizar covarianza
        risk = returns.std() * np.sqrt(252)  # Anualizar volatilidad
        
        # ==================== 4. MÉTRICAS DE RIESGO Y RENTABILIDAD ====================
        st.header("4️⃣ Métricas Anualizadas de Riesgo y Rentabilidad")
        
        # Crear DataFrame con métricas
        sharpe_individual = (mean_returns - risk_free_rate) / risk
        metrics_df = pd.DataFrame({
            'Rendimiento Anual (%)': mean_returns * 100,
            'Volatilidad (%)': risk * 100,
            'Ratio Sharpe': sharpe_individual
        })
        
        st.dataframe(
            metrics_df.style.format({
                'Rendimiento Anual (%)': '{:.2f}%',
                'Volatilidad (%)': '{:.2f}%',
                'Ratio Sharpe': '{:.2f}'
            }),
            use_container_width=True
        )
        
        # Mostrar métricas individuales
        cols = st.columns(len(tickers))
        for idx, ticker in enumerate(tickers):
            with cols[idx]:
                st.subheader(f"📊 {ticker}")
                st.metric("Retorno Anual", f"{mean_returns[ticker]:.2%}")
                st.metric("Volatilidad", f"{risk[ticker]:.2%}")
                st.metric("Ratio Sharpe", f"{sharpe_individual[ticker]:.2f}")
        
        # ==================== 5. SIMULACIÓN MONTE CARLO ====================
        st.header("5️⃣ Simulación de Portafolios (Monte Carlo)")
        
        st.info(f"🎲 Simulando {num_portfolios:,} portafolios aleatorios para encontrar la frontera eficiente...")
        
        with st.spinner("⏳ Ejecutando simulación..."):
            # Arrays para almacenar resultados
            results = np.zeros((3, num_portfolios))
            weights_record = []
            
            for i in range(num_portfolios):
                # Generar pesos aleatorios que sumen 1
                weights = np.random.random(len(tickers))
                weights /= weights.sum()
                weights_record.append(weights)
                
                # Calcular retorno del portafolio
                portfolio_return = np.dot(weights, mean_returns)
                
                # Calcular riesgo del portafolio
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                # Calcular Sharpe ratio
                sharpe = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
                
                # Guardar resultados
                results[0, i] = portfolio_risk    # Riesgo
                results[1, i] = portfolio_return  # Retorno
                results[2, i] = sharpe            # Sharpe
        
        st.success(f"✅ Simulación completada: {num_portfolios:,} portafolios analizados")
        
        # ==================== GRÁFICO DE FRONTERA EFICIENTE ====================
        st.subheader("📈 Frontera Eficiente")
        
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        
        # Scatter plot de todos los portafolios
        scatter = ax4.scatter(
            results[0, :] * 100,  # Convertir a porcentaje
            results[1, :] * 100,  # Convertir a porcentaje
            c=results[2, :],
            cmap='viridis',
            alpha=0.6,
            s=15,
            edgecolors='none'
        )
        
        # Encontrar portafolio óptimo (máximo Sharpe)
        max_sharpe_idx = np.argmax(results[2])
        
        # Encontrar portafolio de mínima volatilidad
        min_vol_idx = np.argmin(results[0])
        
        # Marcar portafolio óptimo
        ax4.scatter(
            results[0, max_sharpe_idx] * 100,
            results[1, max_sharpe_idx] * 100,
            c='red',
            s=500,
            marker='*',
            edgecolors='black',
            linewidths=2,
            label='Portafolio Óptimo (Max Sharpe)',
            zorder=5
        )
        
        # Marcar portafolio de mínima volatilidad
        ax4.scatter(
            results[0, min_vol_idx] * 100,
            results[1, min_vol_idx] * 100,
            c='blue',
            s=300,
            marker='D',
            edgecolors='black',
            linewidths=2,
            label='Min Volatilidad',
            zorder=5
        )
        
        # Marcar activos individuales
        for i, ticker in enumerate(tickers):
            ax4.scatter(
                risk[ticker] * 100,
                mean_returns[ticker] * 100,
                c='orange',
                s=200,
                marker='o',
                edgecolors='black',
                linewidths=2,
                zorder=5
            )
            ax4.annotate(
                ticker,
                (risk[ticker] * 100, mean_returns[ticker] * 100),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold'
            )
        
        ax4.set_xlabel('Riesgo / Volatilidad (%)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Retorno Esperado (%)', fontsize=12, fontweight='bold')
        ax4.set_title('Frontera Eficiente - Análisis de Portafolios', fontsize=14, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax4, label='Ratio de Sharpe')
        cbar.set_label('Ratio de Sharpe', fontsize=11)
        
        ax4.legend(loc='best', fontsize=10)
        ax4.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()
        
        # ==================== 6. PORTAFOLIO ÓPTIMO ====================
        st.header("6️⃣ 🏆 Portafolio Óptimo (Máximo Ratio de Sharpe)")
        
        mejor_riesgo = results[0, max_sharpe_idx]
        mejor_retorno = results[1, max_sharpe_idx]
        mejor_sharpe = results[2, max_sharpe_idx]
        mejores_pesos = weights_record[max_sharpe_idx]
        
        # Mostrar métricas del portafolio óptimo
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="📊 Ratio de Sharpe",
                value=f"{mejor_sharpe:.3f}",
                help="Mide el retorno ajustado por riesgo. Mayor es mejor."
            )
        
        with col2:
            st.metric(
                label="💰 Retorno Esperado",
                value=f"{mejor_retorno:.2%}",
                help="Retorno anualizado esperado del portafolio"
            )
        
        with col3:
            st.metric(
                label="⚠️ Riesgo (Volatilidad)",
                value=f"{mejor_riesgo:.2%}",
                help="Volatilidad anualizada del portafolio"
            )
        
        # Mostrar pesos del portafolio óptimo
        st.subheader("⚖️ Distribución de Pesos Óptima")
        
        pesos_df = pd.DataFrame({
            'Activo': tickers,
            'Peso (%)': mejores_pesos * 100
        })
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.dataframe(
                pesos_df.style.format({'Peso (%)': '{:.2f}%'}),
                use_container_width=True,
                hide_index=True
            )
        
        with col2:
            # Gráfico de torta
            fig5, ax5 = plt.subplots(figsize=(6, 6))
            colors = plt.cm.Set3(range(len(tickers)))
            ax5.pie(
                mejores_pesos,
                labels=tickers,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                textprops={'fontsize': 12, 'fontweight': 'bold'}
            )
            ax5.set_title('Distribución del Portafolio Óptimo', fontsize=12, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig5)
            plt.close()
        
        # ==================== EJEMPLO DE INVERSIÓN ====================
        st.subheader("💵 Ejemplo de Inversión")
        
        inversion = st.number_input(
            "¿Cuánto dinero deseas invertir? (USD)",
            min_value=100,
            max_value=1000000,
            value=10000,
            step=100
        )
        
        st.write("**Distribución recomendada:**")
        for i, ticker in enumerate(tickers):
            monto = inversion * mejores_pesos[i]
            st.write(f"- **{ticker}**: ${monto:,.2f} ({mejores_pesos[i]*100:.2f}%)")
        
        # ==================== 7. COMPARACIÓN DE PORTAFOLIOS ====================
        st.header("7️⃣ Comparación de Estrategias")
        
        # Portafolio de pesos iguales
        weights_equal = np.array([1/len(tickers)] * len(tickers))
        equal_return = np.dot(weights_equal, mean_returns)
        equal_risk = np.sqrt(np.dot(weights_equal.T, np.dot(cov_matrix, weights_equal)))
        equal_sharpe = (equal_return - risk_free_rate) / equal_risk
        
        # Portafolio de mínima volatilidad
        min_vol_return = results[1, min_vol_idx]
        min_vol_risk = results[0, min_vol_idx]
        min_vol_sharpe = results[2, min_vol_idx]
        
        comparison_df = pd.DataFrame({
            'Estrategia': ['Pesos Iguales', 'Mínima Volatilidad', 'Óptimo (Max Sharpe)'],
            'Retorno Anual': [equal_return, min_vol_return, mejor_retorno],
            'Volatilidad': [equal_risk, min_vol_risk, mejor_riesgo],
            'Ratio Sharpe': [equal_sharpe, min_vol_sharpe, mejor_sharpe]
        })
        
        st.dataframe(
            comparison_df.style.format({
                'Retorno Anual': '{:.2%}',
                'Volatilidad': '{:.2%}',
                'Ratio Sharpe': '{:.3f}'
            }).background_gradient(subset=['Ratio Sharpe'], cmap='RdYlGn'),
            use_container_width=True,
            hide_index=True
        )
        
        # ==================== 8. EXPORTAR RESULTADOS ====================
        st.header("8️⃣ Exportar Resultados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV de simulaciones
            df_simulaciones = pd.DataFrame({
                'Riesgo': results[0, :],
                'Retorno': results[1, :],
                'Sharpe': results[2, :]
            })
            csv_sim = df_simulaciones.to_csv(index=False)
            st.download_button(
                label="📥 Descargar Simulaciones (CSV)",
                data=csv_sim,
                file_name=f"simulaciones_{'-'.join(tickers)}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # CSV de portafolio óptimo
            df_optimo = pd.DataFrame({
                'Activo': tickers,
                'Peso': mejores_pesos,
                'Retorno_Anual': mean_returns.values,
                'Volatilidad': risk.values
            })
            csv_opt = df_optimo.to_csv(index=False)
            st.download_button(
                label="📥 Descargar Portafolio Óptimo (CSV)",
                data=csv_opt,
                file_name=f"portafolio_optimo_{'-'.join(tickers)}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        # ==================== RESUMEN FINAL ====================
        st.success("✅ Análisis completado exitosamente")
        
    except Exception as e:
        st.error(f"❌ Error al procesar los datos: {str(e)}")
        st.info("**Sugerencias:**")
        st.write("- Verifica que los tickers sean válidos (ej: AAPL, MSFT, GOOGL)")
        st.write("- Asegúrate de tener conexión a internet")
        st.write("- Confirma que las fechas tengan datos disponibles")
        st.write("- Para acciones colombianas usa .CO al final (ej: ECOPETROL.CO)")

else:
    # ==================== PANTALLA DE INICIO ====================
    st.info("👈 **Configura los parámetros en el panel lateral y presiona 'Ejecutar Análisis'**")
    
    # Guía de uso
    with st.expander("📚 Guía de Uso"):
        st.markdown("""
        ### ¿Cómo usar esta aplicación?
        
        1. **Selecciona los activos** ingresando 1 o 2 tickers en el panel lateral
        2. **Define el período** de análisis con las fechas de inicio y fin
        3. **Ajusta el número de simulaciones** (recomendado: 5,000 - 10,000)
        4. **Presiona "Ejecutar Análisis"** y espera los resultados
        5. **Descarga los resultados** en formato CSV
        
        ### Ejemplos de Tickers:
        
        **🇺🇸 Acciones Estadounidenses:**
        - **AAPL** - Apple Inc.
        - **MSFT** - Microsoft
        - **GOOGL** - Google (Alphabet)
        - **AMZN** - Amazon
        - **TSLA** - Tesla
        - **NVDA** - NVIDIA
        - **META** - Meta (Facebook)
        
        **🇨🇴 Acciones Colombianas:**
        - **ECOPETROL.CO** - Ecopetrol
        - **BANCOLOMBIA.CO** - Bancolombia
        - **GRUPOSURA.CO** - Grupo Sura
        - **ISA.CO** - ISA
        
        **📊 ETFs y Fondos:**
        - **SPY** - S&P 500 ETF
        - **QQQ** - NASDAQ 100 ETF
        - **VOO** - Vanguard S&P 500
        """)
    
    with st.expander("📖 ¿Qué hace esta aplicación?"):
        st.markdown("""
        ### Funcionalidades:
        
        1. **📡 Descarga de Datos Reales**
           - Obtiene precios históricos de Yahoo Finance
           - Datos actualizados y confiables
        
        2. **📊 Análisis de Rentabilidad**
           - Retornos diarios, mensuales y anualizados
           - Retorno acumulado del período
           - Comparación entre activos
        
        3. **⚠️ Análisis de Riesgo**
           - Volatilidad (desviación estándar)
           - Matriz de correlación
           - Diversificación del portafolio
        
        4. **🎲 Simulación Monte Carlo**
           - Genera miles de portafolios aleatorios
           - Identifica la frontera eficiente
           - Encuentra el portafolio óptimo
        
        5. **🏆 Portafolio Óptimo**
           - Maximiza el Ratio de Sharpe
           - Balance ideal entre riesgo y rentabilidad
           - Distribución óptima de pesos
        
        6. **📥 Exportación de Resultados**
           - Descarga datos en formato CSV
           - Listo para análisis adicional
        """)
    
    with st.expander("🎓 Conceptos Financieros Clave"):
        st.markdown("""
        ### Ratio de Sharpe
        Mide el retorno ajustado por riesgo. Se calcula como:
        ```
        Sharpe = (Retorno - Tasa Libre de Riesgo) / Volatilidad
        ```
        - **> 1**: Bueno
        - **> 2**: Muy bueno
        - **> 3**: Excelente
        
        ### Frontera Eficiente
        Conjunto de portafolios que ofrecen el máximo retorno para un nivel de riesgo dado.
        
        ### Diversificación
        Estrategia de combinar diferentes activos para reducir el riesgo total del portafolio.
        
        ### Volatilidad
        Medida de la variabilidad de los retornos. Mayor volatilidad = mayor riesgo.
        
        ### Correlación
        - **+1**: Los activos se mueven juntos
        - **0**: Movimientos independientes
        - **-1**: Los activos se mueven en direcciones opuestas
        """)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>FinanSmart - Análisis de Portafolio de Inversión</strong></p>
    <p>Desarrollado para ITM - Análisis de Costos y Presupuestos</p>
    <p>Datos proporcionados por Yahoo Finance | © 2024</p>
</div>
""", unsafe_allow_html=True)
