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
                # SECCIÓN 3: MATRIZ DE CORRELACIÓN
                # ================================================================
                if len(ticker) > 1:
                    st.header("3️⃣ Matriz de Correlación")
                    st.write("Analiza cómo se mueven los activos entre sí. Valores cercanos a 1 indican movimientos similares.")
                    
                    fig3, ax3 = plt.subplots(figsize=(10, 8))
                    sns.heatmap(
                        rent_diaria.corr(),
                        annot=True,
                        cmap='RdYlGn',
                        center=0,
                        fmt='.2f',
                        square=True,
                        linewidths=1,
                        cbar_kws={"shrink": 0.8},
                        ax=ax3
                    )
                    ax3.set_title('Matriz de Correlación del Portafolio', 
                                 fontsize=16, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig3)
                    
                    st.markdown("---")
                
                # ================================================================
                # CÁLCULO DE MÉTRICAS ANUALIZADAS
                # ================================================================
                rent_promedio = rent_diaria.mean() * 252  # Anualizada
                riesgo = rent_diaria.std() * (252 ** 0.5)  # Anualizada
                sharpe = (rent_promedio - tasa_libre_riesgo) / riesgo
                
                # ================================================================
                # SECCIÓN 4: MÉTRICAS PRINCIPALES
                # ================================================================
                st.header("4️⃣ Métricas de Riesgo y Retorno (Anualizadas)")
                
                resumen = pd.DataFrame({
                    "Rentabilidad Esperada (%)": rent_promedio * 100,
                    "Riesgo (Volatilidad %)": riesgo * 100,
                    "Ratio Sharpe": sharpe
                })
                
                # Mostrar tabla con formato
                st.dataframe(
                    resumen.style.format({
                        "Rentabilidad Esperada (%)": "{:.2f}%",
                        "Riesgo (Volatilidad %)": "{:.2f}%",
                        "Ratio Sharpe": "{:.2f}"
                    }).background_gradient(cmap='RdYlGn', subset=['Ratio Sharpe']),
                    use_container_width=True
                )
                
                # Métricas destacadas
                st.subheader("🏆 Mejores Activos")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    mejor_rentabilidad = resumen['Rentabilidad Esperada (%)'].idxmax()
                    st.metric(
                        "Mayor Rentabilidad",
                        mejor_rentabilidad,
                        f"{resumen.loc[mejor_rentabilidad, 'Rentabilidad Esperada (%)']:.2f}%"
                    )
                
                with col2:
                    menor_riesgo = resumen['Riesgo (Volatilidad %)'].idxmin()
                    st.metric(
                        "Menor Riesgo",
                        menor_riesgo,
                        f"{resumen.loc[menor_riesgo, 'Riesgo (Volatilidad %)']:.2f}%"
                    )
                
                with col3:
                    mejor_sharpe = resumen['Ratio Sharpe'].idxmax()
                    st.metric(
                        "Mejor Sharpe",
                        mejor_sharpe,
                        f"{resumen.loc[mejor_sharpe, 'Ratio Sharpe']:.2f}"
                    )
                
                st.markdown("---")
                
                # ================================================================
                # SECCIÓN 5: GRÁFICO RIESGO VS RENTABILIDAD
                # ================================================================
                st.header("5️⃣ Análisis Riesgo vs Rentabilidad")
                
                fig4, ax4 = plt.subplots(figsize=(12, 8))
                
                # Scatter plot
                scatter = ax4.scatter(
                    resumen['Riesgo (Volatilidad %)'],
                    resumen['Rentabilidad Esperada (%)'],
                    c=resumen['Ratio Sharpe'],
                    cmap='viridis',
                    s=500,
                    alpha=0.6,
                    edgecolors='black',
                    linewidth=2
                )
                
                # Anotaciones
                for idx, row in resumen.iterrows():
                    ax4.annotate(
                        idx,
                        (row['Riesgo (Volatilidad %)'], row['Rentabilidad Esperada (%)']),
                        fontsize=10,
                        fontweight='bold',
                        ha='center'
                    )
                
                ax4.set_xlabel("Riesgo (Volatilidad %)", fontsize=12, fontweight='bold')
                ax4.set_ylabel("Rentabilidad Esperada (%)", fontsize=12, fontweight='bold')
                ax4.set_title("Rendimiento vs Riesgo", fontsize=16, fontweight='bold')
                ax4.grid(True, alpha=0.3)
                
                # Colorbar
                cbar = plt.colorbar(scatter, ax=ax4)
                cbar.set_label('Ratio Sharpe', fontsize=12)
                
                plt.tight_layout()
                st.pyplot(fig4)
                
                st.markdown("---")
                
                # ================================================================
                # SECCIÓN 6: PORTAFOLIO EQUIPONDERADO
                # ================================================================
                st.header("6️⃣ Portafolio con Pesos Iguales")
                st.write(f"Análisis de un portafolio donde cada activo tiene {100/len(ticker):.2f}% de participación.")
                
                weights_equal = np.array([1/len(ticker)] * len(ticker))
                portfolio_return = np.dot(weights_equal, rent_promedio)
                portfolio_risk = np.sqrt(
                    np.dot(weights_equal.T, np.dot(rent_diaria.cov() * 252, weights_equal))
                )
                portfolio_sharpe = (portfolio_return - tasa_libre_riesgo) / portfolio_risk
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rendimiento Esperado", f"{portfolio_return*100:.2f}%")
                with col2:
                    st.metric("Riesgo (Volatilidad)", f"{portfolio_risk*100:.2f}%")
                with col3:
                    st.metric("Ratio Sharpe", f"{portfolio_sharpe:.2f}")
                
                # Mostrar composición
                st.subheader("Composición del Portafolio")
                weights_df = pd.DataFrame({
                    'Activo': ticker,
                    'Peso (%)': weights_equal * 100
                })
                
                fig5, ax5 = plt.subplots(figsize=(10, 6))
                ax5.pie(
                    weights_df['Peso (%)'],
                    labels=weights_df['Activo'],
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=plt.cm.Set3.colors
                )
                ax5.set_title('Distribución del Portafolio', fontsize=14, fontweight='bold')
                st.pyplot(fig5)
                
                st.markdown("---")
                
                # ================================================================
                # SECCIÓN 7: SIMULACIÓN MONTE CARLO
                # ================================================================
                if len(ticker) > 1:
                    st.header("7️⃣ Simulación Monte Carlo - Frontera Eficiente")
                    st.write(f"Simulando {num_simulaciones:,} portafolios aleatorios para encontrar la frontera eficiente...")
                    
                    with st.spinner("Ejecutando simulación..."):
                        results = np.zeros((3, num_simulaciones))
                        weights_record = []
                        
                        for i in range(num_simulaciones):
                            weights = np.random.random(len(ticker))
                            weights /= np.sum(weights)
                            weights_record.append(weights)
                            
                            ret = np.dot(weights, rent_promedio)
                            risk_calc = np.sqrt(
                                np.dot(weights.T, np.dot(rent_diaria.cov() * 252, weights))
                            )
                            sharpe_calc = (ret - tasa_libre_riesgo) / risk_calc if risk_calc != 0 else 0
                            
                            results[0, i] = risk_calc
                            results[1, i] = ret
                            results[2, i] = sharpe_calc
                    
                    # Gráfico de frontera eficiente
                    fig6, ax6 = plt.subplots(figsize=(14, 9))
                    scatter = ax6.scatter(
                        results[0, :] * 100,
                        results[1, :] * 100,
                        c=results[2, :],
                        cmap='plasma',
                        alpha=0.5,
                        s=10
                    )
                    ax6.set_xlabel('Riesgo (Volatilidad %)', fontsize=12, fontweight='bold')
                    ax6.set_ylabel('Retorno Esperado (%)', fontsize=12, fontweight='bold')
                    ax6.set_title('Frontera Eficiente - Simulación Monte Carlo', 
                                 fontsize=16, fontweight='bold')
                    
                    cbar = plt.colorbar(scatter, label='Índice de Sharpe', ax=ax6)
                    ax6.grid(True, alpha=0.3)
                    
                    # Marcar portafolio óptimo
                    max_sharpe_idx = np.argmax(results[2])
                    ax6.scatter(
                        results[0, max_sharpe_idx] * 100,
                        results[1, max_sharpe_idx] * 100,
                        c='red',
                        s=500,
                        marker='*',
                        edgecolors='black',
                        linewidth=2,
                        label='Portafolio Óptimo (Max Sharpe)',
                        zorder=5
                    )
                    
                    # Marcar portafolio de mínimo riesgo
                    min_risk_idx = np.argmin(results[0])
                    ax6.scatter(
                        results[0, min_risk_idx] * 100,
                        results[1, min_risk_idx] * 100,
                        c='blue',
                        s=500,
                        marker='D',
                        edgecolors='black',
                        linewidth=2,
                        label='Mínimo Riesgo',
                        zorder=5
                    )
                    
                    ax6.legend(fontsize=12)
                    plt.tight_layout()
                    st.pyplot(fig6)
                    
                    # Información del portafolio óptimo
                    st.subheader("🏆 Portafolio Óptimo (Máximo Sharpe)")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Índice de Sharpe",
                            f"{results[2, max_sharpe_idx]:.2f}"
                        )
                    with col2:
                        st.metric(
                            "Retorno Esperado",
                            f"{results[1, max_sharpe_idx]*100:.2f}%"
                        )
                    with col3:
                        st.metric(
                            "Riesgo Asociado",
                            f"{results[0, max_sharpe_idx]*100:.2f}%"
                        )
                    
                    # Pesos del portafolio óptimo
                    st.subheader("📊 Composición del Portafolio Óptimo")
                    optimal_weights = weights_record[max_sharpe_idx]
                    optimal_df = pd.DataFrame({
                        'Activo': ticker,
                        'Peso (%)': optimal_weights * 100
                    }).sort_values('Peso (%)', ascending=False)
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.dataframe(
                            optimal_df.style.format({'Peso (%)': '{:.2f}%'}),
                            use_container_width=True
                        )
                    
                    with col2:
                        fig7, ax7 = plt.subplots(figsize=(8, 6))
                        colors = plt.cm.Set3.colors
                        bars = ax7.barh(optimal_df['Activo'], optimal_df['Peso (%)'], color=colors)
                        ax7.set_xlabel('Peso (%)', fontsize=12)
                        ax7.set_title('Distribución del Portafolio Óptimo', 
                                     fontsize=14, fontweight='bold')
                        ax7.grid(True, alpha=0.3, axis='x')
                        
                        # Agregar valores en las barras
                        for bar in bars:
                            width = bar.get_width()
                            ax7.text(width, bar.get_y() + bar.get_height()/2,
                                   f'{width:.1f}%',
                                   ha='left', va='center', fontsize=10)
                        
                        plt.tight_layout()
                        st.pyplot(fig7)
                    
                    st.markdown("---")
                    
                    # ================================================================
                    # SECCIÓN 8: EXPORTAR RESULTADOS
                    # ================================================================
                    st.header("8️⃣ Exportar Resultados")
                    
                    # Preparar datos para exportar
                    df_simulaciones = pd.DataFrame({
                        'Riesgo (%)': results[0, :] * 100,
                        'Retorno (%)': results[1, :] * 100,
                        'Sharpe': results[2, :]
                    })
                    
                    # Agregar información del portafolio óptimo
                    df_optimal = optimal_df.copy()
                    df_optimal['Portafolio'] = 'Óptimo'
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv_simulaciones = df_simulaciones.to_csv(index=False)
                        st.download_button(
                            label="📥 Descargar Simulaciones (CSV)",
                            data=csv_simulaciones,
                            file_name=f"simulaciones_portafolio_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        csv_optimal = df_optimal.to_csv(index=False)
                        st.download_button(
                            label="📥 Descargar Portafolio Óptimo (CSV)",
                            data=csv_optimal,
                            file_name=f"portafolio_optimo_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                
                # ================================================================
                # RECOMENDACIONES FINALES
                # ================================================================
                st.markdown("---")
                st.header("💡 Recomendaciones del Análisis")
                
                with st.expander("Ver Recomendaciones Detalladas"):
                    st.markdown(f"""
                    ### Resumen Ejecutivo:
                    
                    **Mejor activo individual por rentabilidad:** {mejor_rentabilidad} 
                    ({resumen.loc[mejor_rentabilidad, 'Rentabilidad Esperada (%)']:.2f}%)
                    
                    **Activo más conservador:** {menor_riesgo} 
                    ({resumen.loc[menor_riesgo, 'Riesgo (Volatilidad %)']:.2f}%)
                    
                    **Mejor relación riesgo-retorno:** {mejor_sharpe} 
                    (Sharpe: {resumen.loc[mejor_sharpe, 'Ratio Sharpe']:.2f})
                    
                    ---
                    
                    ### Interpretación del Ratio de Sharpe:
                    - **Sharpe < 1:** Retorno bajo para el riesgo asumido
                    - **Sharpe 1-2:** Relación aceptable
                    - **Sharpe 2-3:** Muy buena relación riesgo-retorno
                    - **Sharpe > 3:** Excelente relación riesgo-retorno
                    
                    ---
                    
                    ### Estrategias Sugeridas:
                    
                    1. **Inversión Conservadora:** Enfocarse en {menor_riesgo} para minimizar riesgo
                    2. **Inversión Agresiva:** Considerar {mejor_rentabilidad} para maximizar retorno
                    3. **Inversión Equilibrada:** Utilizar el portafolio óptimo identificado
                    """)
                
            except Exception as e:
                st.error(f"❌ Error al procesar los datos: {str(e)}")
                st.info("Verifica que los tickers sean válidos y que haya datos disponibles para el rango de fechas seleccionado.")

else:
    # ============================================================================
    # PANTALLA DE INICIO
    # ============================================================================
    st.info("👈 **Configura los parámetros en el panel lateral y presiona 'Calcular Rentabilidad y Riesgo'**")
    
    # Información de ayuda
    with st.expander("ℹ️ ¿Cómo usar FinRisk Pro?"):
        st.markdown("""
        ### 📚 Guía de Uso
        
        **FinRisk Pro** te permite analizar múltiples activos financieros y construir portafolios eficientes.
        
        #### Pasos para usar la aplicación:
        
        1. **Selecciona una categoría** de empresas (Tecnología, Financiero, Consumo, etc.)
        2. **Elige las empresas** que deseas analizar (puedes seleccionar múltiples)
        3. **Define el periodo** de análisis (fecha inicio y fin)
        4. **Ajusta parámetros** como la tasa libre de riesgo y número de simulaciones
        5. **Presiona el botón** "Calcular Rentabilidad y Riesgo"
        6. **Explora los resultados** en las diferentes secciones
        7. **Descarga los datos** en formato CSV para análisis adicional
        
        ---
        
        ### 📊 Métricas Clave
        
        **Rentabilidad Esperada:** Retorno promedio anualizado del activo
        
        **Riesgo (Volatilidad):** Desviación estándar anualizada de los retornos
        
        **Ratio de Sharpe:** Mide el retorno ajustado por riesgo
        - Fórmula: (Retorno - Tasa Libre de Riesgo) / Volatilidad
        - Valores más altos indican mejor relación riesgo-retorno
        
        **Frontera Eficiente:** Conjunto de portafolios que maximizan el retorno para un nivel de riesgo dado
        
        ---
        
        ### 🎯 Ventajas de FinRisk Pro
        
        ✅ Análisis profesional de múltiples activos  
        ✅ Visualizaciones interactivas y claras  
        ✅ Simulación Monte Carlo para optimización  
        ✅ Identificación automática del portafolio óptimo  
        ✅ Exportación de resultados en CSV  
        ✅ Análisis de correlación entre activos  
        ✅ Comparación visual riesgo vs retorno  
        
        ---
        
        ### ⚠️ Advertencias
        
        - Los resultados se basan en datos históricos y no garantizan rendimientos futuros
        - La inversión en mercados financieros conlleva riesgos
        - Esta herramienta es para fines educativos y de análisis
        - Consulta con un asesor financiero antes de tomar decisiones de inversión
        """)
    
    with st.expander("📖 Conceptos Financieros Importantes"):
        st.markdown("""
        ### Diversificación
        Es la estrategia de distribuir las inversiones entre diferentes activos para reducir el riesgo.
        No pongas todos los huevos en la misma canasta.
        
        ### Correlación
        Mide cómo se mueven dos activos entre sí:
