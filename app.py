from shiny import App, ui, render, reactive
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from modelo_forecasting import (
    cargar_datos_aforos,
    predecir_valor,
    totales_por_vehiculo_anual,
    VEHICULOS,
)

# ---------------------------
# Datos y opciones
df = cargar_datos_aforos()
TIPOS = VEHICULOS
ANIOS = sorted(df["FECHA"].dt.year.unique())
MESES_NUM = list(range(1, 13))
MESES_NOMBRE = [
    "ENERO",
    "FEBRERO",
    "MARZO",
    "ABRIL",
    "MAYO",
    "JUNIO",
    "JULIO",
    "AGOSTO",
    "SEPTIEMBRE",
    "OCTUBRE",
    "NOVIEMBRE",
    "DICIEMBRE",
]

# --------------------------------
# Utilidades auxiliares

def meses_adelante_hasta(anio_obj: int, mes_obj: int) -> int:
    """Calcula cuántos meses faltan desde el último dato hasta (anio_obj, mes_obj)."""
    ultima = df["FECHA"].max()
    return max((anio_obj - ultima.year) * 12 + (mes_obj - ultima.month), 1)


def obtener_tipos_seleccionados(input):
    return list(input.tipos_seleccionados() or ["AUTOS"])

# -----------------------------
# Estilos

CUSTOM_CSS = """
    .sidebar { 
        background-color: #f8f9fa; 
        padding: 20px; 
        height: 100vh; 
        overflow-y: auto; 
        position: sticky; 
        top: 0; 
        border-right: 1px solid #ddd; 
    }
    .valueBox { 
        border-radius: 10px !important; 
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important; 
        margin-bottom: 20px !important;
    }
    .valueBox .value { 
        font-size: 28px !important; 
        font-weight: bold !important; 
    }
    .valueBox .subtitle { 
        font-size: 16px !important; 
        font-weight: bold !important; 
    }
    .card { 
        border-radius: 10px !important; 
        box-shadow: 0 2px 5px rgba(0,0,0,0.05) !important; 
        margin-bottom: 30px !important;
    }
    h2 {
        color: #005A9C; 
        font-size: 28px; 
        margin-bottom: 20px; 
    }
    h3 { 
        color: #ffffff; 
        margin-top: 40px; 
        margin-bottom: 25px;
        font-size: 22px;
    }
    .control-label {
        font-weight: 600;
        margin-top: 10px;
    }
    .tab-content {
        padding-top: 20px;
    }
    .nav-tabs {
        margin-bottom: 15px;
    }
    .quick-answer {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .container-fluid {
    background-color: #9bc1bc;
    }
"""

# --------------
# UI

controls_sidebar = ui.sidebar(
    ui.h4("Parámetros", class_="mb-4"),
    ui.input_slider("anio_range", "Rango de años para filtrar", 
                   min=min(ANIOS), max=max(ANIOS), 
                   value=(min(ANIOS), max(ANIOS)),
                   step=1),
    ui.hr(),
    ui.input_select("tipo", "Tipo principal de vehículo", choices=TIPOS, selected="AUTOS"),
    ui.input_select("mes", "Mes de análisis", choices={str(i + 1): m for i, m in enumerate(MESES_NOMBRE)}, selected="1"),
    ui.input_select("anio", "Año específico", choices=[str(a) for a in ANIOS], selected="2021"),
    ui.hr(),
    ui.input_checkbox_group("tipos_seleccionados", "Tipos a mostrar en gráficos", 
                           {t: t for t in TIPOS}, 
                           selected=["AUTOS", "MOTOS", "AUTOBUS DE 2 EJES"]),
    ui.hr(),
    ui.input_numeric("meses_forecast", "Meses a predecir", 1, min=1, max=12),
    width=300
)

# -----------------
# Sección ANALISIS (superior)

analysis_section = ui.TagList(
    ui.h3("Análisis y pronóstico"),

    ui.layout_columns(
        ui.value_box("Total de Vehículos", ui.output_text("total_label"), showcase=ui.tags.i(class_="fas fa-car")),
        ui.value_box("Periodo de Pronóstico", ui.output_text("frecuencia_label"), showcase=ui.tags.i(class_="fas fa-calendar")),
        ui.value_box("Pronóstico Estimado", ui.output_text("forecast_label"), showcase=ui.tags.i(class_="fas fa-chart-line")),
        col_widths=[4, 4, 4],
    ),

    ui.layout_columns(
        ui.card(ui.card_header("Pronóstico"), ui.output_plot("grafico_prediccion", height="500px", width="100%")),
        col_widths=[12],
    ),
    ui.layout_columns(
        ui.card(ui.card_header("Frecuencia por año"), ui.output_plot("grafico_frecuencia", height="500px", width="100%")),
        col_widths=[12],
    ),
)

# ----------------
# Sección VISION (inferior)

vision_section = ui.TagList(
    ui.h3("Visión general por año y tipo"),

    ui.layout_columns(
        ui.card(ui.card_header("Tabla de datos"), ui.output_table("tabla"), full_screen=True),
        ui.card(ui.card_header("Resumen por tipo"), ui.output_ui("resumen_tipo")),
        col_widths=[6, 6],
    ),

    ui.card(ui.card_header("Totales por tipo"), ui.output_plot("grafico_comparativo", height="600px", width="100%")),
)

# ------------------------------
# Sección PREGUNTAS RAPIDAS


quick_section = ui.TagList(
    ui.layout_columns(
        ui.div(
            ui.h4("Autos esperados en Junio 2025"),
            ui.div(ui.output_text("respuesta_autos_junio"), class_="quick-answer"),
            class_="p-3"
        ),
        ui.div(
            ui.h4("Vehículo más transitado en 2023"),
            ui.div(ui.output_text("respuesta_mas_2023"), class_="quick-answer"),
            class_="p-3"
        ),
        ui.div(
            ui.h4("Comportamiento Autobuses 2 ejes 2021‑2025"),
            ui.div(ui.output_text("respuesta_bus2_ejes"), class_="quick-answer"),
            class_="p-3"
        ),
        col_widths=[4, 4, 4],
    ),
)

app_ui = ui.page_fluid(
    ui.tags.head(ui.tags.style(CUSTOM_CSS)),
    ui.tags.link(rel="stylesheet", href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css"),
    ui.div(
        ui.h2("Movimientos mensuales por tipo de vehículo en la red CAPUFE", 
              style="text-align:center; margin: 30px 0; font-size: 28px; color: #005A9C;"),
        class_="container-fluid"
    ),

   ui.navset_tab(
        ui.nav_panel("Dashboard", ui.layout_sidebar(controls_sidebar,
                                                   ui.div(
                                                       analysis_section,
                                                       ui.tags.hr(style="margin: 40px 0;"),
                                                       vision_section,
                                                       class_="px-4 py-3"
                                                   ))),
        ui.nav_panel("Preguntas rápidas", ui.div(quick_section, class_="container mt-4")),
    ),
)

# -----------------------------------------------------------------------------
# SERVER


def server(input, output, session):
    # ------------- datos filtrados por rango -------------
    @reactive.Calc
    def datos_filtrados_por_rango():
        """Filtra el dataframe según el rango de años seleccionado"""
        anio_min, anio_max = input.anio_range()
        return df[(df["FECHA"].dt.year >= anio_min) & (df["FECHA"].dt.year <= anio_max)]
    
    # ------------- datos seleccionados -------------
    @reactive.Calc
    def datos_seleccionados():
        """Filtra por año y mes específicos dentro del rango general"""
        df_rango = datos_filtrados_por_rango()
        tipos = obtener_tipos_seleccionados(input)
        mes = int(input.mes() or "1")
        anio = int(input.anio() or "2021")
        return df_rango[
            (df_rango["FECHA"].dt.year == anio) & 
            (df_rango["FECHA"].dt.month == mes)
        ][["FECHA"] + tipos]

    # ------------------- tabla ----------------------
    @output
    @render.table
    def tabla():
        df_rango = datos_filtrados_por_rango()
        mes = int(input.mes() or "1")
        df_filtrado = df_rango[df_rango["FECHA"].dt.month == mes].copy()
        df_filtrado["Año"] = df_filtrado["FECHA"].dt.year
        df_filtrado["Mes"] = df_filtrado["FECHA"].dt.month.apply(lambda m: MESES_NOMBRE[m - 1])
        cols = ["Año", "Mes"] + obtener_tipos_seleccionados(input)
        df_formatted = df_filtrado[cols].copy()
        for col in cols[2:]:
            df_formatted[col] = df_formatted[col].apply(lambda x: f"{int(x):,}")
        return df_formatted

    # ------------------- gráficos -------------------
    @output
    @render.plot
    def grafico_frecuencia():
        df_rango = datos_filtrados_por_rango()
        tipos = obtener_tipos_seleccionados(input)
        agrup = df_rango.groupby(df_rango["FECHA"].dt.year)[tipos].sum().reset_index()
        
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(agrup))
        width = 0.8 / len(tipos)
        colores = sns.color_palette("husl", len(tipos))
        
        for i, t in enumerate(tipos):
            ax.bar(x + i * width, agrup[t], width, label=t, color=colores[i])
        
        ax.set_xticks(x + width * len(tipos) / 2)
        ax.set_xticklabels(agrup["FECHA"], fontsize=12)
        ax.set_title("Frecuencia de vehículos por año", fontsize=16)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Mejorar la leyenda
        if len(tipos) > 5:
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=11)
        else:
            ax.legend(fontsize=12)
        
        # Formatear números grandes en eje Y
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x):,}"))
        ax.set_ylabel("Cantidad de vehículos", fontsize=12)
        
        plt.tight_layout()
        return fig

    @output
    @render.plot
    def grafico_prediccion():
        df_rango = datos_filtrados_por_rango()
        tipo = input.tipo() or "AUTOS"
        meses = input.meses_forecast() or 1
        
        # Usamos los datos filtrados para la serie histórica
        serie_hist = df_rango.set_index("FECHA")[tipo].resample("MS").sum()

        predicciones = []
        intervalos_inf = []
        intervalos_sup = []

        for i in range(1, meses + 1):
        # Predecir para cada horizonte de tiempo
            pred = predecir_valor(tipo, i)
            predicciones.append(pred["prediccion"])
            intervalos_inf.append(pred["inferior"])
            intervalos_sup.append(pred["superior"])

        # Generar fechas futuras
        fechas_pred = pd.date_range(start=serie_hist.index[-1] + pd.DateOffset(months=1), periods=meses, freq="MS")
       
        
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(serie_hist.index, serie_hist.values, "-o", label="Histórico", markersize=4)
        ax.plot(fechas_pred, predicciones, "r--o", label="Pronóstico", markersize=5)
        ax.fill_between(fechas_pred, intervalos_inf, intervalos_sup, color='red', alpha=0.2, 
                    label="Intervalo de confianza")
        
        ax.set_title(f"Pronóstico para {tipo}", fontsize=16)
        ax.grid(True, linestyle="--", alpha=0.3)
        
        # Añadir etiquetas a los ejes
        ax.set_xlabel("Fecha", fontsize=12)
        ax.set_ylabel("Cantidad de vehículos", fontsize=12)
        
        # Formatear números grandes en eje Y
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x):,}"))
        
        # Añadir leyenda con mejor formato
        ax.legend(fontsize=12, loc='best')
        
        # Formato de fecha para el eje x
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        return fig

    @output
    @render.plot
    def grafico_comparativo():
        df_rango = datos_filtrados_por_rango()
        tipos = obtener_tipos_seleccionados(input)
        totales = {t: int(df_rango[t].sum()) for t in tipos}
        ordenados = sorted(totales.items(), key=lambda x: x[1], reverse=True)
        
        labels, values = zip(*ordenados)
        fig, ax = plt.subplots(figsize=(14, 10))  # Mayor altura para el gráfico horizontal
        
        barras = ax.barh(labels, values, color=sns.color_palette("Set2", len(labels)), height=0.6)
        ax.invert_yaxis()
        
        # Añadir etiquetas con formato de miles
        for i, v in enumerate(values):
            ax.text(v * 1.01, i, f"{v:,}", fontsize=11, va='center')
        
        ax.set_title("Totales por tipo de vehículo (en el rango seleccionado)", fontsize=16)
        ax.set_xlabel("Cantidad de vehículos", fontsize=12)
        
        # Aumentar tamaño de fuente para las etiquetas del eje Y (tipos de vehículos)
        ax.tick_params(axis='y', labelsize=11)
        
        # Formato para números grandes en el eje x
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x):,}"))
        
        # Asegurar que todo quepa bien
        fig.tight_layout(pad=3.0)
        return fig

    # ------------------- resúmenes ------------------
    @output
    @render.ui
    def resumen_tipo():
        df_rango = datos_filtrados_por_rango()
        tipo = input.tipo() or "AUTOS"
        
        total = df_rango[tipo].sum()
        mayor = max(VEHICULOS, key=lambda x: df_rango[x].sum())
        menor = min(VEHICULOS, key=lambda x: df_rango[x].sum())
        
        total_general = sum(df_rango[col].sum() for col in VEHICULOS)
        porcentaje = (total / total_general * 100) if total_general else 0
        
        # Crear un resumen con mejor formato
        return ui.div(
            ui.div(
                ui.h4(f"Resumen para {tipo}"),
                ui.p(f"Total en el rango seleccionado: {total:,.0f} vehículos"),
                ui.p(f"Representa {porcentaje:.2f}% del tráfico total"),
                class_="mb-4"
            ),
            ui.div(
                ui.h5("Estadísticas generales:"),
                ui.p([ui.tags.i(class_="fas fa-arrow-up"), f" Tipo con mayor movimiento: ", ui.tags.b(mayor)]),
                ui.p([ui.tags.i(class_="fas fa-arrow-down"), f" Tipo con menor movimiento: ", ui.tags.b(menor)]),
                class_="mb-2"
            ),
            class_="p-3"
        )

    # ------------------- value boxes ---------------
    @output
    @render.text
    def total_label():
        df_rango = datos_filtrados_por_rango()
        return f"{df_rango[input.tipo() or 'AUTOS'].sum():,.0f}"

    @output
    @render.text
    def frecuencia_label():
        m = input.meses_forecast() or 1
        return f"{m} mes{'es' if m != 1 else ''}"

    @output
    @render.text
    def forecast_label():
        pred = predecir_valor(input.tipo() or "AUTOS", input.meses_forecast() or 1)
        return f"{int(pred['prediccion']):,}"

    # ------------------- preguntas rápidas ----------
    @output
    @render.text
    def respuesta_autos_junio():
        meses = meses_adelante_hasta(2025, 6)
        pred = predecir_valor("AUTOS", meses)
        return f"{int(pred['prediccion']):,} autos"

    @output
    @render.text
    def respuesta_mas_2023():
        tot = totales_por_vehiculo_anual(2023, VEHICULOS)
        vehiculo_max = max(tot, key=tot.get)
        return f"{vehiculo_max} con {tot[vehiculo_max]:,} vehículos"

    @output
    @render.text
    def respuesta_bus2_ejes():
        serie = df.set_index("FECHA")["AUTOBUS DE 2 EJES"]["2021":"2025"]
        tendencia = "creciente" if serie.iloc[-1] > serie.iloc[0] else "decreciente"
        return f"Tendencia {tendencia}, con picos en temporadas vacacionales. El promedio mensual es de {int(serie.mean()):,} autobuses."


app = App(app_ui, server)