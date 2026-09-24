import json
import os
import numpy as np
import pandas as pd

# =============================================================================
# RUTAS DE ARCHIVOS DEL PROYECTO
# =============================================================================
FILE_META = "metadata_estaciones.csv"
FILE_KRIGING = "Resultados/Metodo_Kriging_BiasCorrection/Tabla_Estadisticos_Pre_vs_Post_Kriging_Historico.xlsx"
FILE_IDW = "Resultados/Metodo_1_BiasCorrection/Tabla_Estadisticos_Pre_vs_Post_BC_IDW_Historico.xlsx"
FILE_HIST = "Procesados_CSV/series_historicas_consolidadas.parquet"
FILE_FUT = "Procesados_CSV/series_futuras_ssp585.parquet"

os.makedirs("datos/series", exist_ok=True)


def normalizar_codigo(val):
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    digitos = "".join(filter(str.isdigit, s))
    return digitos.zfill(6) if digitos else s


def sanitizar_lista(serie):
    """Redondea floats a 1 decimal y convierte NaN/Inf a None (null en JSON)."""
    if serie is None or serie.empty:
        return []
    res = []
    for val in serie:
        if pd.isna(val) or np.isinf(val):
            res.append(None)
        else:
            res.append(round(float(val), 1))
    return res


print("⏳ Cargando metadatos y estadísticas...")

# 1. Metadatos
df_meta = (
    pd.read_csv(FILE_META)
    if FILE_META.endswith(".csv")
    else pd.read_excel(FILE_META)
)
df_meta.columns = df_meta.columns.str.strip().str.upper()

# Detección inteligente de columnas en metadata_estaciones.csv
col_id = next(
    (c for c in ["ID_ESTACION", "CODIGO", "COD_EST", "COD"] if c in df_meta.columns),
    df_meta.columns[0],
)

# Se agrega "NAM" para identificar el nombre de la estación
col_nom = next(
    (
        c
        for c in ["NAM", "NOMBRE_ESTACION", "ESTACION", "NOMBRE", "NOM_EST", "NOM"]
        if c in df_meta.columns
    ),
    None,
)

col_lat = next((c for c in ["LAT", "LATITUD"] if c in df_meta.columns), None)
col_lon = next((c for c in ["LON", "LONGITUD"] if c in df_meta.columns), None)
col_alt = next(
    (c for c in ["ALT", "ALTITUD", "ELEVACION"] if c in df_meta.columns), None
)
col_src = next((c for c in ["SRC", "FUENTE"] if c in df_meta.columns), None)

df_meta["COD_CLEAN"] = df_meta[col_id].apply(normalizar_codigo)

# 2. Hojas de Estadísticas Kriging
xls_k = pd.ExcelFile(FILE_KRIGING)
k_wrf = pd.read_excel(xls_k, "Kriging_WRF")
k_era5 = pd.read_excel(xls_k, "Kriging_ERA5")
k_wrf["COD_CLEAN"] = k_wrf["CODIGO"].apply(normalizar_codigo)
k_era5["COD_CLEAN"] = k_era5["CODIGO"].apply(normalizar_codigo)

# 3. Hojas de Estadísticas IDW
if os.path.exists(FILE_IDW):
    xls_i = pd.ExcelFile(FILE_IDW)
    sheets_i = xls_i.sheet_names
    i_era5 = pd.read_excel(xls_i, sheets_i[0])
    i_wrf = pd.read_excel(xls_i, sheets_i[1])
    i_era5["COD_CLEAN"] = i_era5["CODIGO"].apply(normalizar_codigo)
    i_wrf["COD_CLEAN"] = i_wrf["CODIGO"].apply(normalizar_codigo)
else:
    i_era5, i_wrf = pd.DataFrame(), pd.DataFrame()

# 4. Series Temporales Parquet
print("⏳ Cargando series temporales en Parquet...")
df_h = (
    pd.read_parquet(FILE_HIST) if os.path.exists(FILE_HIST) else pd.DataFrame()
)
df_f = (
    pd.read_parquet(FILE_FUT) if os.path.exists(FILE_FUT) else pd.DataFrame()
)

if not df_h.empty and "CODIGO" in df_h.columns:
    df_h["CODIGO"] = df_h["CODIGO"].apply(normalizar_codigo)
if not df_f.empty and "CODIGO" in df_f.columns:
    df_f["CODIGO"] = df_f["CODIGO"].apply(normalizar_codigo)

print("⏳ Generando archivos JSON modulares...")
indice_estaciones = {}

for _, row in df_meta.iterrows():
    cod = row["COD_CLEAN"]
    nom = (
        str(row[col_nom]).strip()
        if col_nom and pd.notna(row[col_nom])
        else f"Estación {cod}"
    )

    lat_val = float(row[col_lat]) if col_lat and pd.notna(row[col_lat]) else None
    lon_val = float(row[col_lon]) if col_lon and pd.notna(row[col_lon]) else None
    alt_val = (
        float(row[col_alt]) if col_alt and pd.notna(row[col_alt]) else "N/A"
    )
    src_val = str(row[col_src]) if col_src and pd.notna(row[col_src]) else "N/A"

    # Extraer métricas estadísticas para WRF
    rk_w = k_wrf[k_wrf["COD_CLEAN"] == cod]
    ri_w = (
        i_wrf[i_wrf["COD_CLEAN"] == cod]
        if not i_wrf.empty
        else pd.DataFrame()
    )
    stat_kw = rk_w.iloc[0].to_dict() if not rk_w.empty else {}
    stat_iw = ri_w.iloc[0].to_dict() if not ri_w.empty else {}

    # Extraer métricas estadísticas para ERA5
    rk_e = k_era5[k_era5["COD_CLEAN"] == cod]
    ri_e = (
        i_era5[i_era5["COD_CLEAN"] == cod]
        if not i_era5.empty
        else pd.DataFrame()
    )
    stat_ke = rk_e.iloc[0].to_dict() if not rk_e.empty else {}
    stat_ie = ri_e.iloc[0].to_dict() if not ri_e.empty else {}

    indice_estaciones[cod] = {
        "nombre": nom,
        "lat": lat_val,
        "lon": lon_val,
        "alt": alt_val,
        "src": src_val,
        "stats": {
            "wrf": {
                "obs_media": round(float(stat_kw.get("OBS_MEDIA_M", 0.0) or 0.0), 1),
                "n_dias": int(stat_kw.get("N_DIAS", 0) or 0),
                "pre_rmse": round(float(stat_kw.get("PRE_BC_RMSE", 0.0) or 0.0), 1),
                "krig_rmse": round(float(stat_kw.get("POST_BC_RMSE", 0.0) or 0.0), 1),
                "krig_r": round(float(stat_kw.get("POST_BC_R", 0.0) or 0.0), 3),
                "idw_rmse": round(float(stat_iw.get("POST_BC_RMSE", 0.0) or 0.0), 1),
            },
            "era5": {
                "obs_media": round(float(stat_ke.get("OBS_MEDIA_M", 0.0) or 0.0), 1),
                "n_dias": int(stat_ke.get("N_DIAS", 0) or 0),
                "pre_rmse": round(float(stat_ke.get("PRE_BC_RMSE", 0.0) or 0.0), 1),
                "krig_rmse": round(float(stat_ke.get("POST_BC_RMSE", 0.0) or 0.0), 1),
                "krig_r": round(float(stat_ke.get("POST_BC_R", 0.0) or 0.0), 3),
                "idw_rmse": round(float(stat_ie.get("POST_BC_RMSE", 0.0) or 0.0), 1),
            },
        },
    }

    # Series temporales
    dh = (
        df_h[df_h["CODIGO"] == cod].sort_values("FECHA_KEY")
        if not df_h.empty
        else pd.DataFrame()
    )
    dfu = (
        df_f[df_f["CODIGO"] == cod].sort_values("FECHA_KEY")
        if not df_f.empty
        else pd.DataFrame()
    )

    serie_data = {
        "historico": {
            "fechas": dh["FECHA_KEY"].tolist()
            if not dh.empty and "FECHA_KEY" in dh
            else [],
            "obs": sanitizar_lista(dh.get("HGT_0C")),
            "wrf_pre": sanitizar_lista(dh.get("WRF_PRE_BC")),
            "wrf_kriging": sanitizar_lista(dh.get("WRF_POST_KRIGING")),
            "wrf_idw": sanitizar_lista(dh.get("WRF_POST_IDW")),
            "era5_pre": sanitizar_lista(dh.get("ERA5_PRE_BC")),
            "era5_kriging": sanitizar_lista(dh.get("ERA5_POST_KRIGING")),
            "era5_idw": sanitizar_lista(dh.get("ERA5_POST_IDW")),
        },
        "futuro": {
            "fechas": dfu["FECHA_KEY"].tolist()
            if not dfu.empty and "FECHA_KEY" in dfu
            else [],
            "kriging": sanitizar_lista(dfu.get("POST_KRIGING_SSP585")),
            "idw": sanitizar_lista(dfu.get("POST_IDW_SSP585")),
        },
    }

    # Guardar archivo individual por estación en datos/series/
    with open(f"datos/series/{cod}.json", "w", encoding="utf-8") as f:
        json.dump(serie_data, f, ensure_ascii=False)

# Guardar archivo índice general en datos/estaciones.json
with open("datos/estaciones.json", "w", encoding="utf-8") as f:
    json.dump(indice_estaciones, f, ensure_ascii=False)

print("🎉 ¡Exportación completada con éxito! Archivos livianos generados en 'datos/'.")
