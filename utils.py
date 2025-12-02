import psutil
import dask
import os
import duckdb
from numba import njit
import geopandas as gpd
import math
import xarray as xr
import numpy as np
import pandas as pd
import rasterio
from math import erf, sqrt
from rasterio import features
from shapely.geometry import shape
from dask.distributed import Client, LocalCluster, SSHCluster, progress, wait
from shapely.geometry import Polygon
import warnings
warnings.filterwarnings("ignore")

def createDuckCon(temp_file='/tmp/precip_tracking.duckdb'):
    con = duckdb.connect(temp_file)
    con.execute(f"PRAGMA threads={os.cpu_count()};")
    con.execute("SET memory_limit='90GB';")
    con.execute("SET enable_progress_bar = true")
    con.execute("SET progress_bar_time = 100;")
    con.execute("PRAGMA enable_object_cache;")
    con.execute("PRAGMA preserve_insertion_order=false;")
    con.execute(f"SET temp_directory='/storage/tmp/';")
    con.execute("PRAGMA verify_parallelism;")
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")
    return con

def loadTrackingTable(con, path_files, table_name):
    # Load datasets
    con.execute(f"""
        CREATE OR REPLACE VIEW {table_name} AS
        SELECT *
        FROM read_parquet('{path_files}', union_by_name=true)
        WHERE threshold = 0.1 AND
        timestamp >= '2015-01-01 00:00:00' AND 
        timestamp < '2025-01-01 00:00:00' AND 
        duration >= 120
    """)

def loadSpatialTracking(path_files):
    print("Loading spatial tracking data from: " + path_files)
    # Load xarray dataset
    ds = xr.open_mfdataset(path_files + '*.nc', engine='h5netcdf', 
                           combine='nested', concat_dim='time').sel(lat=slice(-60, 60))
    return ds


def daskClient(n_workers=30, a_memory=80, temp_folder='/mnt/data/tmp', threads_per_worker = 1, node_ips=None, cluster_user=None):
    if n_workers < 1:
        n_workers = 1
    memory_bytes = int(a_memory) * 1024**3
    available_memory = psutil.virtual_memory().available
    if memory_bytes > available_memory:
        memory_bytes = available_memory
    memory_gb = memory_bytes / 1024**3
    memory_gb = (memory_gb / n_workers)
    # Create dask client
    dask.config.set({"temporary_directory": temp_folder})
    dask.config.set({"dataframe.convert-string": False})
    dask.config.set({"distributed.workers.memory.terminate": None})
    # Create dask client with cluster or local
    if node_ips is not None and cluster_user is not None:
        print("--- Dask Cluster ---")
        print('Temp Folder: ' + temp_folder)
        print("Node IPs: " + node_ips)
        print("Cluster User: " + cluster_user)
        nodes = ["localhost"] + node_ips.split(',')
        cluster = SSHCluster(nodes, connect_options={'known_hosts': '~/.ssh/known_hosts',
                            'username': cluster_user,
                            'password': None,
                            'client_keys': None,
                            'server_key': None},
                                worker_options={'nthreads': threads_per_worker,
                                                'memory_limit': memory_bytes},
                                scheduler_options={'port': 8786, 'dashboard_address': ':8787'})
        print('Cluster created')
        print('Number of Workers: ' + str(len(cluster.workers)))
        memory_gb = cluster.workers[0].memory_limit / 1024**3
        print('Memory/Worker: {:.2f}GB'.format(memory_gb))
        print("-------------------\n")
    else:
        print("--- Dask Client ---")
        print('Number of Cores: ' + str(n_workers))
        print('Avaiable Memory: ' + str(available_memory / 1024**3) + 'GB')
        print('Threads: ' + str(n_workers))
        print('Memory/Worker: {:.2f}GB'.format(memory_gb))
        print('Temp Folder: ' + temp_folder)
        print("-------------------\n")   
        cluster = LocalCluster(n_workers=n_workers, processes=True, threads_per_worker=threads_per_worker,
                                memory_limit=str(memory_gb) + 'GB', dashboard_address=':8787')
    client = Client(cluster)
    return client

def daskCompute(darray, client):
    futures = client.compute(darray)
    progress(futures, notebook=False)
    wait(futures)
    carray = client.gather(futures)
    return carray

def ds_stats(da_value, da_weight=None, *, stat="sum", time_dim="time", weighted=False,
    climatology=False,
):
    """
    Reduz 'da_value' para frequência mensal, retornando sempre 'time' como datetime64[ns]
    (label no início do mês, via resample 'MS'). Ponderação (por 'da_weight') é aplicada
    apenas quando 'stat == "mean"' e 'weighted=True'.

    Parâmetros
    ----------
    da_value : xr.DataArray
        Série principal (e.g., precipitação acumulada por pixel). Deve conter 'time_dim'.
    da_weight : xr.DataArray, opcional
        Série de pesos (e.g., nº de ocorrências por pixel). Mesmas dimensões/coords (alinháveis).
    stat : {"sum","mean","max","min","median"}
        Estatística mensal.
    time_dim : str
        Nome da dimensão temporal (padrão: "time").
    weighted : bool
        Se True e stat == "mean", calcula média ponderada mensal por 'da_weight'.
        Para outras estatísticas, 'weighted' é ignorado.
    climatology : bool
        Se True, retorna climatologia mensal (coord 'month' = 1..12) agregando todos os anos.

    Regras de NaN e zeros
    ---------------------
    - NaN em 'da_value' não contribui (é ignorado) nas reduções.
    - Zeros são válidos e entram no cálculo.
    - Para média ponderada, pesos coincidentes com NaN em 'da_value' são tratados como 0.
      Se a soma de pesos no mês/pixel for 0, resulta em NaN.

    Retorna
    -------
    xr.DataArray
        - Se climatology=False: série mensal com 'time' (datetime64[ns], início do mês).
        - Se climatology=True: climatologia com coord 'month' (1..12).
    """
    if time_dim not in da_value.dims:
        raise ValueError(f"`{time_dim}` não está em `da_value.dims`.")

    # Alinhar opcionalmente pesos
    if da_weight is not None:
        da_value, da_weight = xr.align(da_value, da_weight, join="inner")

    # --- Redução mensal garantindo 'time' datetime64[ns] no início do mês ---
    # Para média ponderada, precisamos de num = sum(v*w) e den = sum(w) por mês.
    if stat == "mean" and weighted:
        if da_weight is None:
            raise ValueError("`weighted=True` exige `da_weight`.")
        # Zerar peso onde o valor é NaN para não contaminar o denominador
        w_clean = da_weight.where(da_value.notnull(), other=0.0).fillna(0.0)

        num = (da_value.fillna(0.0) * w_clean).resample({time_dim: "MS"}).sum(skipna=True)
        den = w_clean.resample({time_dim: "MS"}).sum(skipna=True)

        monthly = num / den.where(den != 0)
    else:
        # Redutores não ponderados, todos com skipna=True
        if stat == "sum":
            monthly = da_value.resample({time_dim: "MS"}).sum(skipna=True)
        elif stat == "mean":
            monthly = da_value.resample({time_dim: "MS"}).mean(skipna=True)
        elif stat == "max":
            monthly = da_value.resample({time_dim: "MS"}).max(skipna=True)
        elif stat == "min":
            monthly = da_value.resample({time_dim: "MS"}).min(skipna=True)
        elif stat == "median":
            monthly = da_value.resample({time_dim: "MS"}).median(skipna=True)
        else:
            raise ValueError("`stat` deve ser um de {'sum','mean','max','min','median'}.")

    # monthly possui 'time' como DatetimeIndex (datetime64[ns]) no 1º dia de cada mês (OK)

    if climatology:
        # Agrupa por mês do calendário (1..12) e reduz ao longo do tempo
        # Mantém mesma estatística para a climatologia (consistente ao caso mensal já computado)
        g = monthly.groupby(f"{time_dim}.month")
        if stat == "sum":
            out = g.sum(dim=time_dim, skipna=True)
        elif stat == "mean":
            out = g.mean(dim=time_dim, skipna=True)
        elif stat == "max":
            out = g.max(dim=time_dim, skipna=True)
        elif stat == "min":
            out = g.min(dim=time_dim, skipna=True)
        elif stat == "median":
            out = g.median(dim=time_dim, skipna=True)
        # Coord 'month' (1..12). Aqui não há coord 'time' por definição de climatologia.
        return out

    return monthly



def mean_map_per_event(
    da_value: xr.DataArray,
    da_density: xr.DataArray,
    time_name: str = "time",
    mode: str = "mean_of_ratios",  # "ratio_of_sums" ou "mean_of_ratios"
    min_events_per_step: int = 0,  # ex.: 3 ou 5 para filtrar meses raros
    use_time_weights: bool = False,
):
    """
    Calcula mapa (lat,lon) de média por evento.
    - mode="ratio_of_sums": sum(value)/sum(density)  [event-weighted; estável]
    - mode="mean_of_ratios": mean_t(value/density)   [mês típico; filtrar baixa densidade]
    """
    assert mode in {"ratio_of_sums", "mean_of_ratios"}

    if mode == "ratio_of_sums":
        num = da_value.sum(time_name, skipna=True)
        den = da_density.sum(time_name, skipna=True)
        return (num / den).where(den > 0)

    # mean_of_ratios:
    valid = (da_density >= max(1, min_events_per_step)) & da_value.notnull()
    per_event = (da_value / da_density).where(valid)

    if not use_time_weights:
        return per_event.mean(time_name, skipna=True)

    # ponderação por ∆t (se temporalmente irregular)
    t = pd.to_datetime(per_event[time_name].values)
    if len(t) < 2:
        return per_event.squeeze(drop=True)

    dt = np.diff(t).astype("timedelta64[s]").astype(float)
    dt = np.append(dt, dt[-1])
    w = xr.DataArray(dt, coords={time_name: per_event[time_name]}, dims=(time_name,))
    num = (per_event * w).sum(time_name, skipna=True)
    den = w.where(per_event.notnull()).sum(time_name, skipna=True)
    return num / den


def spatial_weighted_mean_series(da_value, mask=None, lat_name='lat', lon_name='lon'):
    # pesos de área ~ cos(lat) para grids regulares em lon/lat
    lat = da_value.coords[lat_name]
    w_lat = np.cos(np.deg2rad(lat))
    # normaliza shape para broadcast: (lat, lon)
    W = xr.ones_like(da_value, dtype=float)
    W = W * w_lat # broadcast em lat
    # aplica máscara, se fornecida
    if mask is not None:
        da_value = da_value.where(mask)
        W = W.where(mask)
    # soma ponderada e soma de pesos no espaço
    num = (da_value * W).sum(dim=[lat_name, lon_name], skipna=True)
    den = W.sum(dim=[lat_name, lon_name], skipna=True)
    return num / den

def pixel2area(
    da,
    lat_coord=None,
    lon_coord=None,
    earth_radius_km=6371.0,
    units="km2",
):
    """
    Convert pixel-count data to area (km² or m²) for regular lat/lon grids (EPSG:4326).
        
    Parameters
    ----------
    da : xarray.DataArray
        Data in "pixel units" (e.g., 5 = 5 pixels).
    lat_coord, lon_coord : str, optional
        Names of latitude/longitude coords. Auto-detected.
    earth_radius_km : float
        Earth radius (default: 6371.0 km).
    units : {'km2', 'm2'}

    Returns
    -------
    result : xarray.DataArray
        Same shape as `da`, in km² or m².
    """
    # --- Auto-detect lat/lon ---
    def _find_coord(candidates):
        for c in candidates:
            if c in da.coords:
                return c
        raise ValueError(f"Coordinate not found. Tried: {candidates}")

    lat_coord = lat_coord or _find_coord(["lat", "latitude", "y"])
    lon_coord = lon_coord or _find_coord(["lon", "longitude", "x"])

    lat = da.coords[lat_coord]
    lon = da.coords[lon_coord]

    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("This function requires 1D (regular) lat/lon. For 2D, use pixel2km().")

    # --- Detect spacing automatically ---
    dlat_deg = np.abs(np.diff(lat.values)).mean()
    dlon_deg = np.abs(np.diff(lon.values)).mean()

    # Optional: warn if spacing is not uniform
    if not np.allclose(np.diff(lat.values), dlat_deg, rtol=1e-3):
        print("⚠️  Latitude spacing is not perfectly uniform — using mean Δlat.")
    if not np.allclose(np.diff(lon.values), dlon_deg, rtol=1e-3):
        print("⚠️  Longitude spacing is not perfectly uniform — using mean Δlon.")

    # --- Compute area per latitude (1D) ---
    R = earth_radius_km
    dlat_rad = np.deg2rad(dlat_deg)
    dlon_rad = np.deg2rad(dlon_deg)
    lat_rad = np.deg2rad(lat.values)
    
    # Area = R² · dlat · dlon · cos(lat)
    area_lat = (R ** 2) * dlat_rad * dlon_rad * np.cos(lat_rad)  # (nlat,)

    # --- Broadcast to da's shape ---
    # Create shape like da, but 1s everywhere except lat axis
    area_shape = [1] * da.ndim
    lat_axis = da.get_axis_num(lat_coord)
    area_shape[lat_axis] = len(lat)
    area_broadcast = area_lat.reshape(area_shape)

    # Multiply
    result = da * area_broadcast

    # Convert units
    if units == "m2":
        result = result * 1e6
        unit_str = "m2"
    elif units == "km2":
        unit_str = "km2"
    else:
        raise ValueError("units must be 'km2' or 'm2'")

    # Preserve metadata
    result.attrs.update(da.attrs)
    result.attrs["units"] = unit_str
    if "long_name" in da.attrs:
        result.attrs["long_name"] = da.attrs["long_name"] + f" (converted to {unit_str})"

    return result

def point2retPolygon(lat, lon, dx_km, dy_km):
    dlat = dy_km / 111.0
    dlon = dx_km / (111.0 * math.cos(math.radians(lat)))
    poly = Polygon([
        (lon - dlon, lat - dlat),
        (lon - dlon, lat + dlat),
        (lon + dlon, lat + dlat),
        (lon + dlon, lat - dlat),
        (lon - dlon, lat - dlat)
    ])
    return poly

def geo_areas(ds, perc=None, threshold=None, min_area=100000, operation=np.greater_equal):

    # Extrai valores de ds
    data = ds.values
    lat = ds['lat'].values
    lon = ds['lon'].values
    # 1. Definir thresholds
    if perc is not None:
        threshold = np.percentile(data, perc)
    
    # 2. Criar máscaras binárias
    data_mask = operation(data, threshold).astype(np.uint8)
    # 3. Função para extrair polígonos do raster
    def raster_to_polygons(mask, transform=None):
        polygons = []
        for geom, val in features.shapes(mask, mask=mask, transform=transform):
            polygons.append(shape(geom))
        return polygons
    # 4. Criar transform simples (lon/lat uniformes)
    pixel_size_x = (lon[-1] - lon[0]) / (len(lon)-1)
    pixel_size_y = (lat[0] - lat[-1]) / (len(lat)-1)  # nota: topo - base
    transform = rasterio.transform.from_origin(
        west=lon[0] - pixel_size_x/2,  # limite esquerdo
        north=lat[0] + pixel_size_y/2, # latitude do topo + metade do pixel
        xsize=pixel_size_x,
        ysize=pixel_size_y
    )
    # 5. Extrair polígonos
    high_polygons = raster_to_polygons(data_mask, transform)
    # 6. Criar GeoDataFrames
    gdf = gpd.GeoDataFrame({'geometry': high_polygons}, crs='EPSG:4326')
    # Remove geometries where area in km² less then 1
    gdf['area_km2'] = gdf['geometry'].to_crs(epsg=3395).area / 10**6
    gdf = gdf[gdf['area_km2'] >= min_area]
    return gdf

# --- utilidades ---
def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0])
    r = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-r, r + 1, dtype=float)
    k = np.exp(-(x**2) / (2.0 * sigma**2))
    return k / k.sum()

def _gaussian_smooth_separable(arr: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Suavização gaussiana separável (lat, lon) com NumPy."""
    if sigma_cells <= 0:
        return arr
    k = _gaussian_kernel_1d(sigma_cells)
    tmp = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, arr)  # lat
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, tmp)  # lon
    return out

def _edges_from_centers(centers: np.ndarray) -> np.ndarray:
    """Deriva bordas a partir dos centros (assume espaçamento aproximadamente regular)."""
    c = np.asarray(centers)
    d = np.diff(c)
    d0 = d[0] if d.size else 1.0
    left  = c[0] - d0/2
    right = c[-1] + (d[-1] if d.size else d0)/2
    return np.r_[left, c[:-1] + d/2, right]


# --- principal ---
def hotspots_dataset(
    da: xr.DataArray,
    *,
    area_weight: bool = False,          # True => ~ densidade por área (1/cos(lat))
    smooth_sigma_cells: float = 2.0,    # σ em número de células
    normalize: str = "global",          # "none" | "global"
    hotspot_quantile: float = 0.99      # limiar para hotspots (ex.: 0.95, 0.99, 0.995)
) -> xr.Dataset:
    """
    Constrói um Dataset com campo suavizado e máscara de hotspots por percentil.
    Entrada:
        da(lat, lon): DataArray global (contagens/frequências em grade regular lon/lat, graus).
    Saída:
        Dataset com variáveis:
          - value_raw(lat, lon)
          - value_norm(lat, lon)
          - value_smooth(lat, lon)
          - hotspots_mask(lat, lon)  [bool]
        e coordenadas lat/lon + bordas lat_bnds/lon_bnds.
    """
    # --- validações básicas ---
    if not {"lat", "lon"}.issubset(set(da.dims)):
        raise ValueError("O DataArray deve ter dimensões 'lat' e 'lon'.")

    # garante float64 para convolução estável (mantém dtype final depois)
    arr = np.asarray(da.values, dtype=float)
    lat = np.asarray(da["lat"].values, dtype=float)
    lon = np.asarray(da["lon"].values, dtype=float)

    # --- (opcional) ajuste por área: ~ 1/cos(lat) ---
    if area_weight:
        # broadcast: (lat, 1)
        w = 1.0 / np.clip(np.cos(np.deg2rad(lat)), 0.1, None)
        arr_weighted = arr * w[:, None]
    else:
        arr_weighted = arr

    # --- normalização ---
    if normalize == "global":
        m = np.nanmax(arr_weighted)
        denom = m if np.isfinite(m) and m > 0 else 1.0
        value_norm = arr_weighted / denom
    elif normalize == "none":
        value_norm = arr_weighted.copy()
    else:
        raise ValueError("normalize deve ser 'none' ou 'global'.")

    # --- suavização gaussiana separável ---
    value_smooth = _gaussian_smooth_separable(value_norm, smooth_sigma_cells)

    # --- hotspots por percentil global (no campo suavizado) ---
    pos = value_smooth[np.isfinite(value_smooth) & (value_smooth > 0)]
    thr = float(np.quantile(pos, hotspot_quantile)) if pos.size else 0.0
    hotspots_mask = value_smooth >= thr

    # --- bordas da grade ---
    lon_bnds = _edges_from_centers(lon)
    lat_bnds = _edges_from_centers(lat)

    # --- monta Dataset ---
    ds = xr.Dataset(
        data_vars=dict(
            value_raw      =(("lat","lon"), arr,           {"long_name":"raw gridded values", "units":"1"}),
            value_norm     =(("lat","lon"), value_norm,    {"long_name":"normalized values",  "units":"1", "normalize":normalize}),
            value_smooth   =(("lat","lon"), value_smooth,  {"long_name":"smoothed values",    "units":"1", "sigma_cells":smooth_sigma_cells}),
            hotspots_mask  =(("lat","lon"), hotspots_mask, {"long_name":f"hotspots >= q{int(hotspot_quantile*100)}", "threshold":thr}),
        ),
        coords=dict(
            lat=("lat", lat, {"units":"degrees_north"}),
            lon=("lon", lon, {"units":"degrees_east"}),
            lat_bnds=(("bnds","lat"), np.vstack([lat_bnds[:-1], lat_bnds[1:]]), {"units":"degrees_north"}),
            lon_bnds=(("bnds","lon"), np.vstack([lon_bnds[:-1], lon_bnds[1:]]), {"units":"degrees_east"}),
        ),
        attrs=dict(
            title="Hotspot dataset from gridded field",
            area_weight=area_weight,
            smooth_sigma_cells=smooth_sigma_cells,
            normalize=normalize,
            hotspot_quantile=hotspot_quantile,
            hotspot_threshold=thr,
            crs_note="Assume lon/lat em graus (EPSG:4326) com espaçamento ~regular."
        ),
    )
    return ds


# =========================
# Núcleo Mann–Kendall (MK)
# =========================

@njit(cache=True)  # sem fastmath no núcleo MK
def _tie_correction_sum_eps(y, eps=0.0):
    """
    Soma para correção de ties no Mann–Kendall clássico,
    considerando empates quando |Δy| <= eps.
    """
    n = len(y)
    if n <= 1:
        return 0.0
    ys = np.sort(y)
    tie_sum = 0.0
    c = 1
    for i in range(1, n):
        if abs(ys[i] - ys[i-1]) <= eps:
            c += 1
        else:
            if c > 1:
                tie_sum += c * (c - 1) * (2.0 * c + 5.0)
            c = 1
    if c > 1:
        tie_sum += c * (c - 1) * (2.0 * c + 5.0)
    return tie_sum

@njit(cache=True)  # sem fastmath no núcleo MK
def _mk_tau_p_core_eps(y, eps=0.0):
    """
    Mann–Kendall clássico (sem pesos) com correção de ties e epsilon opcional.
    Retorna (tau, p).
    """
    n = len(y)
    s = 0
    for i in range(n - 1):
        yi = y[i]
        for j in range(i + 1, n):
            diff = y[j] - yi
            # empates se |diff| <= eps
            s += (diff > eps) - (diff < -eps)

    tie_sum = _tie_correction_sum_eps(y, eps)
    var_s = (n * (n - 1) * (2.0 * n + 5.0) - tie_sum) / 18.0
    if var_s <= 0.0:
        return 0.0, 1.0

    if s > 0:
        z = (s - 1.0) / sqrt(var_s)
    elif s < 0:
        z = (s + 1.0) / sqrt(var_s)
    else:
        z = 0.0

    tau = s / (0.5 * n * (n - 1.0))
    # p-valor bicaudal via erf
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))
    return tau, p

# =========================
# Theil–Sen (slope)
# =========================

@njit(cache=True, fastmath=True)
def _sens_slope_median_twopass(y, t):
    """
    Theil–Sen com tempo físico. Duas passagens:
      (1) contar pares válidos (dt!=0) para alocar;
      (2) preencher vetor e pegar mediana via np.partition (O(m)).
    """
    n = len(y)
    valid_pairs = 0
    for i in range(n - 1):
        ti = t[i]
        for j in range(i + 1, n):
            dt = t[j] - ti
            if np.isfinite(dt) and dt != 0.0:
                valid_pairs += 1
    if valid_pairs == 0:
        return np.nan

    slopes = np.empty(valid_pairs, dtype=np.float64)
    k = 0
    for i in range(n - 1):
        yi = y[i]
        ti = t[i]
        for j in range(i + 1, n):
            dt = t[j] - ti
            if np.isfinite(dt) and dt != 0.0:
                dy = y[j] - yi
                if np.isfinite(dy):
                    slopes[k] = dy / dt
                else:
                    slopes[k] = np.nan
                k += 1

    if k == 0:
        return np.nan

    slopes = slopes[:k]
    slopes = slopes[np.isfinite(slopes)]  # remove NaNs antes da mediana
    if slopes.size == 0:
        return np.nan

    mid = slopes.size // 2
    np.partition(slopes, mid)
    if slopes.size % 2 == 1:
        return slopes[mid]
    else:
        a = np.partition(slopes, mid - 1)[mid - 1]
        b = slopes[mid]
        return 0.5 * (a + b)

# =========================
# Agregação de duplicatas
# =========================

@njit(cache=True)
def _aggregate_duplicates_by_mean(y, t):
    """
    Agrega valores com o mesmo timestamp (t) por média.
    Pré-condição: t está ordenado de forma não decrescente.
    """
    n = len(t)
    if n == 0:
        return y, t
    # contar grupos
    groups = 1
    for i in range(1, n):
        if t[i] != t[i - 1]:
            groups += 1

    y_out = np.empty(groups, np.float64)
    t_out = np.empty(groups, np.float64)

    idx = 0
    acc = y[0]
    cnt = 1
    cur_t = t[0]
    for i in range(1, n):
        if t[i] == cur_t:
            acc += y[i]
            cnt += 1
        else:
            y_out[idx] = acc / cnt
            t_out[idx] = cur_t
            idx += 1
            cur_t = t[i]
            acc = y[i]
            cnt = 1
    # último grupo
    y_out[idx] = acc / cnt
    t_out[idx] = cur_t
    return y_out, t_out

# =========================
# Pipeline MK + Sen (Numba)
# =========================

@njit(cache=True)  # sem fastmath aqui; comparações importam
def mk_tau_p_sen_numba(y, t, w=None, eps=0.0, aggregate_duplicates=True):
    """
    Pipeline único:
      1) filtra NaN (+ pesos) e ordena por tempo,
      2) (opcional) agrega timestamps duplicados por média,
      3) centragem ponderada (se w),
      4) MK (tau, p) com correção de ties e epsilon opcional,
      5) Sen’s slope (mediana dos slopes vs. tempo físico).
    Retorna (tau, p, slope).
    """
    y = np.asarray(y, np.float64)
    t = np.asarray(t, np.float64)

    if w is None:
        mask = np.isfinite(y) & np.isfinite(t)
        y = y[mask]
        t = t[mask]
        if y.size < 2:
            return np.nan, np.nan, np.nan
    else:
        w = np.asarray(w, np.float64)
        mask = np.isfinite(y) & np.isfinite(t) & np.isfinite(w) & (w > 0.0)
        y = y[mask]
        t = t[mask]
        w = w[mask]
        if y.size < 2:
            return np.nan, np.nan, np.nan
        wsum = w.sum()
        if wsum <= 0.0:
            return np.nan, np.nan, np.nan
        # centragem ponderada (não altera slope)
        y = y - (y * w).sum() / wsum

    # ordenar por tempo (garante t não decrescente)
    order = np.argsort(t)
    y = y[order]
    t = t[order]

    # (opcional) agregar timestamps duplicados exatos
    if aggregate_duplicates and y.size > 1:
        y, t = _aggregate_duplicates_by_mean(y, t)
        if y.size < 2:
            return np.nan, np.nan, np.nan

    # MK (tau, p) com epsilon
    tau, p = _mk_tau_p_core_eps(y, eps)
    # Sen’s slope
    slope = _sens_slope_median_twopass(y, t)
    return tau, p, slope

# =========================
# Utilitários xarray/dask
# =========================

def time_to_physical_time(tcoord: xr.DataArray) -> xr.DataArray:
    """
    Converte datetime64 para contagem de meses desde a primeira observação.
    Se já for numérico, apenas faz cast para float64.
    """
    if np.issubdtype(tcoord.dtype, np.datetime64):
        t_ref = np.datetime64(tcoord.values[0], 'M')  # trunca para mês
        months_since = (tcoord.values.astype('datetime64[M]') - t_ref) / np.timedelta64(1, 'M')
        return xr.DataArray(months_since, coords=tcoord.coords, dims=tcoord.dims)
    return tcoord.astype('float64')

def rechunk(ds, CHUNK_LAT=50, CHUNK_LON=50):
    return ds.chunk({"time": -1, "lat": CHUNK_LAT, "lon": CHUNK_LON})

def mk_sen_field(
    ds: xr.DataArray,
    weight_ds: xr.DataArray | None = None,
    time_coord: str = 'time',
    allow_rechunk: bool = False,
    CHUNK_LAT: int = 50,
    CHUNK_LON: int = 50,
    pval_threshold: float = 0.05,
    eps: float = 0.0,
    aggregate_duplicates: bool = True,
    normalize_slope: bool = True,
) -> xr.Dataset:
    """
    Aplica MK (tau,p) + Sen em cada grade.

    Parâmetros novos:
      - eps: tolerância para considerar empates (|Δy| <= eps).
      - aggregate_duplicates: agrega valores com mesmo timestamp por média.

    Requisitos:
      - 'time' em um único chunk;
      - lat/lon chunkados em blocos menores (ex.: 100x100).
    """
    ds = ds.where(np.isfinite(ds))
    if weight_ds is not None:
        weight_ds = weight_ds.where(weight_ds > 0)

    # Rechunk lat/lon
    ds = rechunk(ds, CHUNK_LAT=CHUNK_LAT, CHUNK_LON=CHUNK_LON)
    if weight_ds is not None:
        weight_ds = rechunk(weight_ds, CHUNK_LAT=CHUNK_LAT, CHUNK_LON=CHUNK_LON)

    # Garantir time em 1 chunk (obrigatório p/ core_dims)
    try:
        n_time_chunks = len(ds.chunks[time_coord])
    except Exception:
        n_time_chunks = 1
    if n_time_chunks > 1:
        ds = ds.chunk({time_coord: -1})
        if weight_ds is not None:
            weight_ds = weight_ds.chunk({time_coord: -1})

    # Vetor temporal físico
    t_num = time_to_physical_time(ds[time_coord])

    dkwargs = {'allow_rechunk': bool(allow_rechunk)}
    ufunc_kwargs = {'eps': float(eps), 'aggregate_duplicates': bool(aggregate_duplicates)}

    if weight_ds is None:
        tau, p, slope = xr.apply_ufunc(
            mk_tau_p_sen_numba,
            ds, t_num,
            input_core_dims=[[time_coord], [time_coord]],
            output_core_dims=[[], [], []],
            vectorize=True,
            dask='parallelized',
            dask_gufunc_kwargs=dkwargs,
            kwargs=ufunc_kwargs,
            output_dtypes=[float, float, float],
        )
    else:
        tau, p, slope = xr.apply_ufunc(
            mk_tau_p_sen_numba,
            ds, t_num, weight_ds,
            input_core_dims=[[time_coord], [time_coord], [time_coord]],
            output_core_dims=[[], [], []],
            vectorize=True,
            dask='parallelized',
            dask_gufunc_kwargs=dkwargs,
            kwargs=ufunc_kwargs,
            output_dtypes=[float, float, float],
        )

    # Normalize by time span in years
    if normalize_slope:
        slope /= (t_num[-1] - t_num[0]) / 12.0  # meses -> anos

    # Máscara de significância diretamente no slope
    trend_slope = slope.where(p < pval_threshold)

    # Calcule p_value with
    p_value = p.where(p < pval_threshold)

    trend = xr.Dataset(
        {
            "p": p,
            "tau": tau,
            "slope": slope,
            "trend": trend_slope,
            "p_value": p_value
        }
    )

    # Atributos sem puxar .values (evita compute de arrays grandes)
    start_str = ds[time_coord].isel({time_coord: 0}).dt.strftime("%Y-%m-%d").item()
    end_str   = ds[time_coord].isel({time_coord: -1}).dt.strftime("%Y-%m-%d").item()
    trend.attrs.update({
        "start_date": start_str,
        "end_date": end_str,
        "description": "Mann–Kendall trend test (two-sided) with Sen’s slope",
        "pv_alpha": pval_threshold,
        "eps_tie": eps,
        "aggregate_duplicates": aggregate_duplicates,
    })

    return trend

def _wrap_lon(lon):
    """Normaliza lon para o intervalo [-180, 180)."""
    lon = ((lon + 180.0) % 360.0) - 180.0
    lon = np.where(np.isclose(lon, 180.0), -180.0, lon)
    return lon

def _grid_index(lon, lat, *, lon_min=-180.0, lon_max=180.0, lat_min=-60.0, lat_max=60.0, step=10.0):
    """
    Converte lon/lat em (col, row_sul→norte, qid_norte0, lon_center, lat_center).
    A numeração do quadrante (qid) começa no NORTE (linha superior) em 0.
    Retorna None se fora da janela.
    """
    lon = float(lon); lat = float(lat)
    lon = ((lon + 180.0) % 360.0) - 180.0
    if not (lon_min <= lon < lon_max) or not (lat_min <= lat < lat_max):
        return None

    ncols = int((lon_max - lon_min) / step)
    nrows = int((lat_max - lat_min) / step)

    col_s = int(np.floor((lon - lon_min) / step))           # 0 .. ncols-1 (oeste→leste)
    row_s = int(np.floor((lat - lat_min) / step))           # 0 .. nrows-1 (SUL→NORTE)

    # reindexa linha para 0 no NORTE (topo)
    row_n = (nrows - 1) - row_s                             # 0 .. nrows-1 (NORTE→SUL)

    # quadrante 0-based iniciando no NORTE, varrendo esquerda→direita (W→E)
    qid = row_n * ncols + col_s                             # 0 .. nrows*ncols-1

    lon_c = lon_min + (col_s + 0.5) * step
    lat_c = lat_min + (row_s + 0.5) * step
    return col_s, row_s, qid, lon_c, lat_c

def _grid_frame(lon_min=-180.0, lon_max=180.0, lat_min=-60.0, lat_max=60.0, step=10.0):
    """
    DataFrame completo da grade com numeração de quadrante começando no NORTE como 0.
    """
    ncols = int((lon_max - lon_min) / step)
    nrows = int((lat_max - lat_min) / step)

    rows = []
    for row_s in range(nrows):                   # SUL→NORTE
        row_n = (nrows - 1) - row_s              # NORTE→SUL
        for col_s in range(ncols):               # OESTE→LESTE
            qid = row_n * ncols + col_s          # NORTE-0
            lon0 = lon_min + col_s * step
            lon1 = lon_min + (col_s + 1) * step
            lat0 = lat_min + row_s * step
            lat1 = lat_min + (row_s + 1) * step
            rows.append({
                "quad_id": qid, "col": col_s, "row_s": row_s, "row_n": row_n,
                "lon_center": lon0 + 0.5 * step, "lat_center": lat0 + 0.5 * step,
                "lon0": lon0, "lon1": lon1, "lat0": lat0, "lat1": lat1
            })
    return pd.DataFrame(rows)

def summarize_quadrants(
    gdf: gpd.GeoDataFrame,
    *,
    start_col="start_geometry",
    end_col="end_geometry",
    season_col=None,
    season=None,
    lon_min=-180.0, lon_max=180.0, lat_min=-60.0, lat_max=60.0,
    step=10.0
) -> pd.DataFrame:
    """
    Retorna colunas:
      quad_id (0-based, começa no NORTE), Qtdy_Start, Qtdy_End,
      Common_Dest (quad_id 0-based), Common_Dest_Count,
      lon_center, lat_center, lon0, lon1, lat0, lat1, col, row_s, row_n
    """
    df = gdf
    if season_col and season is not None:
        df = df[df[season_col] == season].copy()

    # extrai lon/lat iniciais/finais
    s_lon = _wrap_lon(df[start_col].x.values)
    s_lat = df[start_col].y.values
    e_lon = _wrap_lon(df[end_col].x.values)
    e_lat = df[end_col].y.values

    # binagem vetorizada (loop leve sobre arrays)
    def _vec_grid_id(lons, lats):
        ids = np.full(lons.shape[0], -1, dtype=int)
        cols = np.full_like(ids, -1)
        rows_s = np.full_like(ids, -1)
        for i, (lo, la) in enumerate(zip(lons, lats)):
            idx = _grid_index(lo, la, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max, step=step)
            if idx is not None:
                c, r_s, q, _, _ = idx
                ids[i] = q; cols[i] = c; rows_s[i] = r_s
        return ids, cols, rows_s

    q_s, _, _ = _vec_grid_id(s_lon, s_lat)
    q_e, _, _ = _vec_grid_id(e_lon, e_lat)

    valid = (q_s >= 0) & (q_e >= 0)
    q_s = q_s[valid]
    q_e = q_e[valid]

    # contagens
    starts = pd.Series(q_s, name="quad_id").value_counts(sort=False).rename_axis("quad_id").reset_index(name="Qtdy_Start")
    ends   = pd.Series(q_e, name="quad_id").value_counts(sort=False).rename_axis("quad_id").reset_index(name="Qtdy_End")

    # matriz OD para Common Dest
    od = pd.DataFrame({"q_s": q_s, "q_e": q_e}).value_counts().rename("count").reset_index()
    common_dest = (od.sort_values(["q_s", "count"], ascending=[True, False])
                     .groupby("q_s", as_index=False).first()
                     .rename(columns={"q_s":"quad_id", "q_e":"Common_Dest", "count":"Common_Dest_Count"}))

    # grade completa e merge
    grid = _grid_frame(lon_min, lon_max, lat_min, lat_max, step)
    out = (grid
           .merge(starts, on="quad_id", how="left")
           .merge(ends,   on="quad_id", how="left")
           .merge(common_dest, on="quad_id", how="left"))

    out["Qtdy_Start"] = out["Qtdy_Start"].fillna(0).astype(int)
    out["Qtdy_End"]   = out["Qtdy_End"].fillna(0).astype(int)
    out["Common_Dest"] = out["Common_Dest"].fillna(-1).astype(int)
    out["Common_Dest_Count"] = out["Common_Dest_Count"].fillna(0).astype(int)

    cols = ["quad_id","col","row_s","row_n","lon_center","lat_center",
            "lon0","lon1","lat0","lat1","Qtdy_Start","Qtdy_End","Common_Dest","Common_Dest_Count"]
    return out[cols]