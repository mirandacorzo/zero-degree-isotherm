import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium
import os 
import glob 

st.set_page_config(
    page_title="Isoterma 0°C - Análisis & Mapa Interactivo", layout="wide"
)

st.title("🏔️ Isoterma 0°C")
st.markdown(
    "Explorador de desempeño de la corrección de sesgo espacial en la Isoterma"
    " $0^\\circ\\text{C}$ sobre los Andes, mediante dos métodos espaciales"
)

# Rutas de archivos CSV consolidados
FILE_KRIGING = "Resultados/Metodo_Kriging_BiasCorrection/Tabla_Estadisticos_Pre_vs_Post_Kriging_Historico.xlsx"
FILE_IDW = (
    "Resultados/Metodo_1_BiasCorrection/Tabla_Estadisticos_Pre_vs_Post_BC_IDW_Historico.xlsx"
)
FILE_META = "metadata_estaciones.csv"

FILE_CSV_HISTORICO = "Procesados_CSV/series_historicas_consolidadas.parquet"
FILE_CSV_FUTURO = "Procesados_CSV/series_futuras_ssp585.parquet"


def normalizar_codigo(val):
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    digitos = "".join(filter(str.isdigit, s))
    return digitos.zfill(6) if digitos else s


@st.cache_data
def cargar_todo():
    df_meta = (
        pd.read_csv(FILE_META)
        if FILE_META.endswith(".csv")
        else pd.read_excel(FILE_META)
    )
    df_meta.columns = df_meta.columns.str.strip().str.upper()
    df_meta["COD_CLEAN"] = df_meta["ID_ESTACION"].apply(normalizar_codigo)

    xls_k = pd.ExcelFile(FILE_KRIGING)
    k_era5 = pd.read_excel(xls_k, "Kriging_ERA5")
    k_wrf = pd.read_excel(xls_k, "Kriging_WRF")
    k_era5["COD_CLEAN"] = k_era5["CODIGO"].apply(normalizar_codigo)
    k_wrf["COD_CLEAN"] = k_wrf["CODIGO"].apply(normalizar_codigo)

    if pd.io.common.file_exists(FILE_IDW):
        xls_i = pd.ExcelFile(FILE_IDW)
        sheets_i = xls_i.sheet_names
        i_era5 = pd.read_excel(xls_i, sheets_i[0])
        i_wrf = pd.read_excel(xls_i, sheets_i[1])
        i_era5["COD_CLEAN"] = i_era5["CODIGO"].apply(normalizar_codigo)
        i_wrf["COD_CLEAN"] = i_wrf["CODIGO"].apply(normalizar_codigo)
    else:
        i_era5, i_wrf = pd.DataFrame(), pd.DataFrame()

    # Cargar CSVs preprocesados
    df_hist = (
        pd.read_parquet(FILE_CSV_HISTORICO)
        if os.path.exists(FILE_CSV_HISTORICO)
        else pd.DataFrame()
    )
    df_fut = (
        pd.read_parquet(FILE_CSV_FUTURO)
        if os.path.exists(FILE_CSV_FUTURO)
        else pd.DataFrame()
    )

    if not df_hist.empty:
        df_hist["FECHA_DT"] = pd.to_datetime(df_hist["FECHA_KEY"])
    if not df_fut.empty:
        df_fut["FECHA_DT"] = pd.to_datetime(df_fut["FECHA_KEY"])

    return df_meta, k_era5, k_wrf, i_era5, i_wrf, df_hist, df_fut

df_meta, k_era5, k_wrf, i_era5, i_wrf, df_hist_all, df_fut_all = cargar_todo()


# =============================================================================
# FUNCIONES RÁPIDAS DE FILTRADO
# =============================================================================
def extraer_serie_diaria(cod_est, modelo_sel):
    if df_hist_all.empty:
        return pd.DataFrame()
    
    df_st = df_hist_all[df_hist_all["CODIGO"] == cod_est].copy()
    if df_st.empty:
        return pd.DataFrame()

    # Mapear columnas según el modelo seleccionado
    df_res = pd.DataFrame({"FECHA_DT": df_st["FECHA_DT"]})
    if "HGT_0C" in df_st.columns:
        df_res["HGT_0C"] = df_st["HGT_0C"]

    if modelo_sel == "ERA5":
        if "ERA5_PRE_BC" in df_st.columns: df_res["PRE_BC"] = df_st["ERA5_PRE_BC"]
        if "ERA5_POST_KRIGING" in df_st.columns: df_res["POST_KRIGING"] = df_st["ERA5_POST_KRIGING"]
        if "ERA5_POST_IDW" in df_st.columns: df_res["POST_IDW"] = df_st["ERA5_POST_IDW"]
    else:
        if "WRF_PRE_BC" in df_st.columns: df_res["PRE_BC"] = df_st["WRF_PRE_BC"]
        if "WRF_POST_KRIGING" in df_st.columns: df_res["POST_KRIGING"] = df_st["WRF_POST_KRIGING"]
        if "WRF_POST_IDW" in df_st.columns: df_res["POST_IDW"] = df_st["WRF_POST_IDW"]

    return df_res.sort_values("FECHA_DT")


def extraer_serie_ssp585(cod_est):
    if df_fut_all.empty:
        return pd.DataFrame()
    df_st = df_fut_all[df_fut_all["CODIGO"] == cod_est].copy()
    return df_st.sort_values("FECHA_DT") if not df_st.empty else pd.DataFrame()


# SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuración")
modelo_sel = st.sidebar.radio("1. Selecciona Modelo:", ["WRF", "ERA5"])
df_k = k_wrf if modelo_sel == "WRF" else k_era5
df_i = i_wrf if modelo_sel == "WRF" else i_era5

df_k = pd.merge(
    df_k,
    df_meta[["COD_CLEAN", "LAT", "LON", "ALT", "SRC"]],
    on="COD_CLEAN",
    how="left",
)
if not df_i.empty:
    df_i = pd.merge(
        df_i,
        df_meta[["COD_CLEAN", "LAT", "LON", "ALT", "SRC"]],
        on="COD_CLEAN",
        how="left",
    )

modo_vis = st.sidebar.radio(
    "2. Modo de Análisis:",
    [
        "Por Estación Individual",
        "Todas las Estaciones (Vista Regional)",
        "🗺️ Mapa Geoespacial",
    ],
)
metodo_sel = st.sidebar.radio(
    "3. Método Espacial:",
    ["Comparativa (IDW vs Kriging)", "Solo Kriging", "Solo IDW"],
)

st.sidebar.markdown("---")
mostrar_ssp585 = st.sidebar.checkbox(
    "🔮 Escenario Futuro WRF (SSP5-8.5: 2015–2065)",
    value=False,
)

# -----------------------------------------------------------------------------
# VISTA: POR ESTACIÓN INDIVIDUAL
# -----------------------------------------------------------------------------
if modo_vis == "Por Estación Individual":
    st.sidebar.markdown("---")
    lista_estaciones = sorted(df_k["NOMBRE_ESTACION"].unique().tolist())
    estacion_sel = st.sidebar.selectbox(
        "🎯 Selecciona la Estación:", lista_estaciones
    )

    row_k = df_k[df_k["NOMBRE_ESTACION"] == estacion_sel].iloc[0]
    row_i = (
        df_i[df_i["NOMBRE_ESTACION"] == estacion_sel].iloc[0]
        if not df_i.empty and estacion_sel in df_i["NOMBRE_ESTACION"].values
        else None
    )

    cod_est = row_k["CODIGO"]
    alt_est = (
        row_k["ALT"] if "ALT" in row_k and not pd.isna(row_k["ALT"]) else "N/A"
    )
    src_est = (
        row_k["SRC"] if "SRC" in row_k and not pd.isna(row_k["SRC"]) else "N/A"
    )

    p_val_pre = row_k.get("PRE_BC_P_VALUE", 0.0)
    sig_pre_str = (
        "✅ SÍ (p ≤ 0.05)"
        if p_val_pre <= 0.05 or row_k.get("PRE_BC_SIG_95") == "SI"
        else "❌ NO"
    )

    p_val_k = row_k.get("POST_BC_P_VALUE", 0.0)
    sig_k_str = (
        "✅ SÍ (p ≤ 0.05)"
        if p_val_k <= 0.05 or row_k.get("POST_BC_SIG_95") == "SI"
        else "❌ NO"
    )

    if row_i is not None:
        p_val_i = row_i.get("POST_BC_P_VALUE", 0.0)
        sig_i_str = (
            "✅ SÍ (p ≤ 0.05)"
            if p_val_i <= 0.05 or row_i.get("POST_BC_SIG_95") == "SI"
            else "❌ NO"
        )
    else:
        sig_i_str = "N/A"

    st.subheader(
        f"📍 Estación: {estacion_sel} ({cod_est}) — Modelo: {modelo_sel}"
    )

    col_obs1, col_obs2, col_obs3, col_obs4 = st.columns(4)
    col_obs1.metric(
        " Isoterma 0°C Observada (Media)", f"{row_k['OBS_MEDIA_M']:.1f} m s.n.m."
    )
    col_obs2.metric(
        " Altitud Estación",
        f"{alt_est} m s.n.m." if alt_est != "N/A" else "N/A",
    )
    col_obs3.metric(" Días Observados (N)", f"{row_k['N_DIAS']} días")
    col_obs4.metric(" Fuente", f"{src_est}")

    st.divider()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(
        "Isoterma Pre-BC",
        f"{row_k['PRE_BC_MEDIA']:.1f} m",
        delta=f"Sig. 95%: {sig_pre_str}",
    )

    if row_i is not None:
        kpi2.metric(
            "Isoterma Post-IDW",
            f"{row_i['POST_BC_MEDIA']:.1f} m",
            delta=f"Sig. 95%: {sig_i_str}",
        )
    else:
        kpi2.metric("Isoterma Post-IDW", "N/A")

    kpi3.metric(
        "Isoterma Post-Kriging",
        f"{row_k['POST_BC_MEDIA']:.1f} m",
        delta=f"Sig. 95%: {sig_k_str}",
    )
    kpi4.metric(
        "Correlación R (Kriging)",
        f"{row_k['POST_BC_R']:.3f}",
        delta=f"p-val: {p_val_k:.2e}",
    )

    st.divider()

    st.subheader("📈 Serie de Tiempo Diaria Comparativa (Histórico 1980–2014)")
    df_ts = extraer_serie_diaria(normalizar_codigo(cod_est), modelo_sel)

    if not df_ts.empty:
        fig_ts = go.Figure()
        if "HGT_0C" in df_ts.columns:
            fig_ts.add_trace(
                go.Scatter(
                    x=df_ts["FECHA_DT"],
                    y=df_ts["HGT_0C"],
                    name="Observación",
                    line=dict(color="#1f77b4", width=1.3),
                    connectgaps=False,
                )
            )

        if "POST_IDW" in df_ts.columns and metodo_sel in [
            "Comparativa (IDW vs Kriging)",
            "Solo IDW",
        ]:
            fig_ts.add_trace(
                go.Scatter(
                    x=df_ts["FECHA_DT"],
                    y=df_ts["POST_IDW"],
                    name="Post-IDW",
                    line=dict(color="#ff7f0e", width=1.1),
                    connectgaps=False,
                )
            )

        if "POST_KRIGING" in df_ts.columns and metodo_sel in [
            "Comparativa (IDW vs Kriging)",
            "Solo Kriging",
        ]:
            fig_ts.add_trace(
                go.Scatter(
                    x=df_ts["FECHA_DT"],
                    y=df_ts["POST_KRIGING"],
                    name="Post-Kriging",
                    line=dict(color="#00cc96", width=0.9),
                    connectgaps=False,
                )
            )

        fig_ts.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Isoterma 0°C (m s.n.m.)",
            height=400,
            template="plotly_white",
            hovermode="x unified",
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    if mostrar_ssp585:
        st.divider()
        st.subheader(
            f"🔮 Proyección Futura WRF — Escenario SSP5-8.5 (2015–2065) | {estacion_sel}"
        )

        df_fut = extraer_serie_ssp585(normalizar_codigo(cod_est))

        if not df_fut.empty:
            fig_fut = go.Figure()

            if "POST_IDW_SSP585" in df_fut.columns and metodo_sel in [
                "Comparativa (IDW vs Kriging)",
                "Solo IDW",
            ]:
                fig_fut.add_trace(
                    go.Scatter(
                        x=df_fut["FECHA_DT"],
                        y=df_fut["POST_IDW_SSP585"],
                        name="Futuro Post-IDW (SSP5-8.5)",
                        line=dict(color="#d62728", width=0.2),
                        connectgaps=False,
                    )
                )

                df_fut["POST_IDW_MA30"] = (
                    df_fut["POST_IDW_SSP585"]
                    .rolling(window=30, min_periods=1)
                    .mean()
                )

                fig_fut.add_trace(
                    go.Scatter(
                        x=df_fut["FECHA_DT"],
                        y=df_fut["POST_IDW_MA30"],
                        name="Media Móvil 30d Post-IDW",
                        line=dict(color="#8c564b", width=2),
                    )
                )

            if "POST_KRIGING_SSP585" in df_fut.columns and metodo_sel in [
                "Comparativa (IDW vs Kriging)",
                "Solo Kriging",
            ]:
                fig_fut.add_trace(
                    go.Scatter(
                        x=df_fut["FECHA_DT"],
                        y=df_fut["POST_KRIGING_SSP585"],
                        name="Futuro Post-Kriging (SSP5-8.5)",
                        line=dict(color="#9467bd", width=0.2),
                        connectgaps=False,
                    )
                )

                df_fut["POST_KRIGING_MA30"] = (
                    df_fut["POST_KRIGING_SSP585"]
                    .rolling(window=30, min_periods=1)
                    .mean()
                )

                fig_fut.add_trace(
                    go.Scatter(
                        x=df_fut["FECHA_DT"],
                        y=df_fut["POST_KRIGING_MA30"],
                        name="Media Móvil 30d Post-Kriging",
                        line=dict(color="#117a65", width=2),
                    )
                )

            fig_fut.update_layout(
                xaxis_title="Fecha Proyección",
                yaxis_title="Isoterma 0°C (m s.n.m.)",
                height=420,
                template="plotly_white",
                hovermode="x unified",
            )
            st.plotly_chart(fig_fut, use_container_width=True)
        else:
            st.info(
                "No se encontraron archivos de proyección futura para esta estación."
            )

    st.divider()

    c_left, c_right = st.columns([2, 1])
    with c_left:
        st.subheader("📊 Métricas de Error por Método")
        metricas = ["|BIAS| Absoluto", "MAE", "RMSE"]
        val_pre = [
            abs(row_k["PRE_BC_BIAS"]),
            row_k["PRE_BC_MAE"],
            row_k["PRE_BC_RMSE"],
        ]
        val_k = [
            abs(row_k["POST_BC_BIAS"]),
            row_k["POST_BC_MAE"],
            row_k["POST_BC_RMSE"],
        ]

        fig_est = go.Figure()
        fig_est.add_trace(
            go.Bar(
                x=metricas,
                y=val_pre,
                name="Pre-BC Original",
                marker_color="#ef553b",
            )
        )

        if row_i is not None and metodo_sel in [
            "Comparativa (IDW vs Kriging)",
            "Solo IDW",
        ]:
            val_i = [
                abs(row_i["POST_BC_BIAS"]),
                row_i["POST_BC_MAE"],
                row_i["POST_BC_RMSE"],
            ]
            fig_est.add_trace(
                go.Bar(
                    x=metricas,
                    y=val_i,
                    name="Post-IDW",
                    marker_color="#ff7f0e",
                )
            )

        if metodo_sel in ["Comparativa (IDW vs Kriging)", "Solo Kriging"]:
            fig_est.add_trace(
                go.Bar(
                    x=metricas,
                    y=val_k,
                    name="Post-Kriging",
                    marker_color="#00cc96",
                )
            )

        fig_est.update_layout(
            barmode="group",
            yaxis_title="Metros (m)",
            height=350,
            template="plotly_white",
        )
        st.plotly_chart(fig_est, use_container_width=True)

    with c_right:
        st.subheader("🗺️ Ubicación Espacial")
        lat_est, lon_est = row_k.get("LAT"), row_k.get("LON")
        if lat_est and lon_est and not pd.isna(lat_est):
            m_single = folium.Map(
                location=[lat_est, lon_est], zoom_start=9, tiles="OpenStreetMap"
            )
            folium.Marker(
                location=[lat_est, lon_est],
                popup=f"<b>{estacion_sel}</b><br>Altitud: {alt_est} m",
                tooltip=estacion_sel,
                icon=folium.Icon(color="red", icon="info-sign"),
            ).add_to(m_single)
            st_folium(m_single, width=350, height=350)

    st.subheader("📋 Resumen Estadístico & Significancia al 95% de Confianza")
    resumen_dict = {
        "Estado": ["Observación Inicial", "Pre-BC (Original)"],
        "Isoterma Media (m)": [row_k["OBS_MEDIA_M"], row_k["PRE_BC_MEDIA"]],
        "BIAS (m)": [0.0, row_k["PRE_BC_BIAS"]],
        "MAE (m)": [0.0, row_k["PRE_BC_MAE"]],
        "RMSE (m)": [0.0, row_k["PRE_BC_RMSE"]],
        "Correlación (R)": [1.0, row_k["PRE_BC_R"]],
        "P-Value": [0.0, p_val_pre],
        "Significativo (95%)": ["SÍ", sig_pre_str],
    }

    if row_i is not None:
        resumen_dict["Estado"].append("Post-IDW")
        resumen_dict["Isoterma Media (m)"].append(row_i["POST_BC_MEDIA"])
        resumen_dict["BIAS (m)"].append(row_i["POST_BC_BIAS"])
        resumen_dict["MAE (m)"].append(row_i["POST_BC_MAE"])
        resumen_dict["RMSE (m)"].append(row_i["POST_BC_RMSE"])
        resumen_dict["Correlación (R)"].append(row_i["POST_BC_R"])
        resumen_dict["P-Value"].append(p_val_i)
        resumen_dict["Significativo (95%)"].append(sig_i_str)

    resumen_dict["Estado"].append("Post-Kriging")
    resumen_dict["Isoterma Media (m)"].append(row_k["POST_BC_MEDIA"])
    resumen_dict["BIAS (m)"].append(row_k["POST_BC_BIAS"])
    resumen_dict["MAE (m)"].append(row_k["POST_BC_MAE"])
    resumen_dict["RMSE (m)"].append(row_k["POST_BC_RMSE"])
    resumen_dict["Correlación (R)"].append(row_k["POST_BC_R"])
    resumen_dict["P-Value"].append(p_val_k)
    resumen_dict["Significativo (95%)"].append(sig_k_str)

    st.dataframe(pd.DataFrame(resumen_dict))

# -----------------------------------------------------------------------------
# VISTA: REGIONAL
# -----------------------------------------------------------------------------
elif modo_vis == "Todas las Estaciones (Vista Regional)":
    st.subheader(f"📊 Resumen Regional — Modelo: {modelo_sel}")
    c1, c2, c3, c4 = st.columns(4)
    rmse_pre = df_k["PRE_BC_RMSE"].mean()
    rmse_k = df_k["POST_BC_RMSE"].mean()
    rmse_i = (
        df_i["POST_BC_RMSE"].mean()
        if not df_i.empty and "POST_BC_RMSE" in df_i.columns
        else 0
    )
    r_k = df_k["POST_BC_R"].mean()

    c1.metric("RMSE Pre-BC (Original)", f"{rmse_pre:.1f} m")
    c2.metric(
        "RMSE Post-IDW", f"{rmse_i:.1f} m", delta=f"{rmse_i - rmse_pre:.1f} m"
    )
    c3.metric(
        "RMSE Post-Kriging",
        f"{rmse_k:.1f} m",
        delta=f"{rmse_k - rmse_pre:.1f} m",
    )
    c4.metric("R Post-Kriging", f"{r_k:.2f}")

    st.divider()

    st.subheader(f"📈 Comparativa del Error (RMSE) Regional — {modelo_sel}")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df_k["NOMBRE_ESTACION"],
            y=df_k["PRE_BC_RMSE"],
            name="Pre-BC Original",
            marker_color="#ef553b",
        )
    )

    if (
        metodo_sel in ["Comparativa (IDW vs Kriging)", "Solo IDW"]
        and not df_i.empty
    ):
        fig.add_trace(
            go.Bar(
                x=df_i["NOMBRE_ESTACION"],
                y=df_i["POST_BC_RMSE"],
                name="Post-IDW",
                marker_color="#ff7f0e",
            )
        )
    if metodo_sel in ["Comparativa (IDW vs Kriging)", "Solo Kriging"]:
        fig.add_trace(
            go.Bar(
                x=df_k["NOMBRE_ESTACION"],
                y=df_k["POST_BC_RMSE"],
                name="Post-Kriging",
                marker_color="#00cc96",
            )
        )

    fig.update_layout(
        barmode="group",
        xaxis_title="Estación",
        yaxis_title="RMSE (m s.n.m.)",
        height=500,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# VISTA: MAPA GEOESPACIAL
# -----------------------------------------------------------------------------
else:
    st.subheader("MAPA Interactivo de la Red de Estaciones Meteorológicas")
    df_mapa = df_k.dropna(subset=["LAT", "LON"]).copy()

    if not df_mapa.empty:
        lat_cen = df_mapa["LAT"].mean()
        lon_cen = df_mapa["LON"].mean()
        m_red = folium.Map(
            location=[lat_cen, lon_cen], zoom_start=7, tiles="OpenStreetMap"
        )

        for idx, row in df_mapa.iterrows():
            sig_k = (
                "SÍ"
                if row.get("POST_BC_P_VALUE", 0.0) <= 0.05
                or row.get("POST_BC_SIG_95") == "SI"
                else "NO"
            )
            popup_html = f"""
            <div style="font-family: sans-serif; width: 220px;">
                <h4 style="margin-bottom:2px;">{row['NOMBRE_ESTACION']}</h4>
                <b>Código:</b> {row['CODIGO']}<br>
                <b>Isoterma Observada:</b> {row['OBS_MEDIA_M']:.1f} m<br>
                <b>RMSE Kriging:</b> {row['POST_BC_RMSE']:.1f} m<br>
                <b>R Kriging:</b> {row['POST_BC_R']:.3f}<br>
                <b>Significativo (95%):</b> {sig_k}
            </div>
            """
            col_marker = (
                "green"
                if row["POST_BC_R"] >= 0.6
                else "orange"
                if row["POST_BC_R"] >= 0.4
                else "red"
            )

            folium.CircleMarker(
                location=[row["LAT"], row["LON"]],
                radius=7,
                color=col_marker,
                fill=True,
                fill_color=col_marker,
                fill_opacity=0.8,
                popup=popup_html,
                tooltip=f"{row['NOMBRE_ESTACION']} ({row['CODIGO']})",
            ).add_to(m_red)

        st_folium(m_red, width=1100, height=600)
