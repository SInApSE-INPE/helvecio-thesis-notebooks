from __future__ import annotations
import numpy as np
import xarray as xr

# Import atrasado dentro dos workers é OK, mas fazemos no módulo para falhar cedo se faltou instalar.
try:
    import pymannkendall as pmk
except Exception as _e:
    pmk = None

# ============================================================
# Utilitários de tempo e normalização da inclinação (Sen)
# ============================================================

def _samples_per_year_from_time(tcoord: xr.DataArray) -> float:
    """
    Estima amostragem por ano para converter a slope do PMK (por amostra)
    em slope por ano.

    Estratégia:
      - Se datetime e espaçamento "quase mensal" -> 12.0
      - Se datetime genérico, usa (N-1) / (anos de span)
      - Se numérico, assume unidade já é o passo temporal desejado e retorna 1.0
        (ou seja, você controla a escala de 'ds' nesse caso)
    """
    if np.issubdtype(tcoord.dtype, np.datetime64):
        # Número de anos no span
        t0 = tcoord.isel({tcoord.dims[0]: 0}).values
        t1 = tcoord.isel({tcoord.dims[0]: -1}).values
        # anos decimais
        span_years = (t1 - t0) / np.timedelta64(1, "D") / 365.2425
        n = tcoord.sizes[tcoord.dims[0]]
        if n < 2:
            return 1.0
        # Tenta detectar série mensal: diferença mediana ~ 1 mês
        diffs = np.diff(tcoord.values.astype("datetime64[ms]")).astype("timedelta64[ms]").astype(np.int64)
        if diffs.size > 0:
            med_days = np.median(diffs) / (1000.0 * 60.0 * 60.0 * 24.0)
            if 25.0 <= med_days <= 35.0:  # ~1 mês
                return 12.0
        if span_years > 0:
            return (n - 1) / span_years
        return 1.0
    # Eixo temporal numérico: sem suposições
    return 1.0


# ============================================================
# Chamada ao pymannkendall em 1D (núcleo escalar)
# ============================================================

def _pmk_call_1d(y: np.ndarray,
                 *,
                 test: str = "original",
                 alpha: float = 0.05,
                 period: int | None = None,
                 lag: int | None = None
                 ) -> tuple[float, float, float, float, float, float, float, int, int]:
    """
    Chamada robusta ao pymannkendall para uma série 1D.
    Retorna apenas números:
      p, z, tau, s, var_s, slope, intercept, h, trend_code
    Em casos degenerados (constante/denom=0), retorna 'sem tendência':
      p=1, z=0, tau=0, s=0, var_s=0, slope=0, intercept=nan, h=0, trend_code=0
    """
    if pmk is None:
        raise ImportError("pymannkendall não está instalado. Faça: pip install pymannkendall")

    # 1) Limpeza
    y = np.asarray(y)
    y = y[np.isfinite(y)]
    n = y.size

    # 2) Requisitos mínimos
    if n < 3:
        return 1.0, 0.0, 0.0, 0.0, 0.0, np.nan, np.nan, 0, 0

    # 3) Série constante (ou quase) → denom=0 no PMK
    #    (compara n° de valores únicos; para dados inteiros, isso é bem estável)
    if np.unique(y).size < 2:
        return 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(np.nan), 0, 0

    # 4) Seleção do teste
    tname = test.lower()
    try:
        if tname in ("original", "mk", "mann-kendall"):
            res = pmk.original_test(y, alpha=alpha)

        elif tname in ("seasonal", "seasonal_test", "seasonal-mk", "mk-seasonal"):
            if period is None:
                raise ValueError("Para 'seasonal', forneça 'period' (ex.: 12 para mensal).")
            # Checagem rápida: pelo menos 2 obs por estação em média
            # (evita muitos casos triviais que levam a denom=0)
            if n < 2 * int(period):
                # ainda assim podemos tentar, mas se quebrar, caímos no except e devolvemos neutro
                pass
            res = pmk.seasonal_test(y, period=int(period), alpha=alpha)

        elif tname in ("hamed-rao", "hamed_rao", "hamed_rao_modification_test", "hr"):
            if lag is None:
                res = pmk.hamed_rao_modification_test(y, alpha=alpha)
            else:
                res = pmk.hamed_rao_modification_test(y, alpha=alpha, lag=int(lag))

        elif tname in ("yue-wang", "yue_wang", "yue_wang_modification_test", "yw"):
            if lag is None:
                res = pmk.yue_wang_modification_test(y, alpha=alpha)
            else:
                res = pmk.yue_wang_modification_test(y, alpha=alpha, lag=int(lag))

        elif tname in ("sci", "sens_slope_test", "sens-slope"):
            res = pmk.sens_slope_test(y, alpha=alpha)

        else:
            raise ValueError(f"Teste '{test}' não reconhecido.")

    except ZeroDivisionError:
        # Caso clássico: denom=0 dentro do PMK (todos empates ou série constante por estação)
        return 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(np.nan), 0, 0
    except Exception:
        # Qualquer outra exceção numérica incomum → retorna neutro
        return 1.0, 0.0, 0.0, 0.0, 0.0, np.nan, np.nan, 0, 0

        # 5) Extrai resultados do namedtuple (sempre numéricos)
    trend_map = {"increasing": 1, "decreasing": -1}
    trend_code = trend_map.get(getattr(res, "trend", "").lower(), 0)
    h = 1 if bool(getattr(res, "h", False)) else 0

    p        = float(getattr(res, "p", np.nan))
    z        = float(getattr(res, "z", np.nan))
    tau      = float(getattr(res, "Tau", np.nan))
    s        = float(getattr(res, "s", np.nan))
    var_s    = float(getattr(res, "var_s", np.nan))
    slope    = float(getattr(res, "slope", np.nan))
    intercept= float(getattr(res, "intercept", np.nan))

    # 5a) Fallback para inclinação/intercepto se o HR não fornecer
    if not np.isfinite(slope):
        try:
            _sen = pmk.sens_slope_test(y, alpha=alpha)
            slope     = float(getattr(_sen, "slope", np.nan))
            intercept = float(getattr(_sen, "intercept", np.nan))
        except Exception:
            pass  # se ainda falhar, deixaremos NaN e trataremos abaixo

    # 6) Pós-validação: neutralize apenas em casos realmente degenerados
    #    (p, z, tau todos não finitos E slope também não finita, ou série muito curta já tratada antes)
    inval_core = (not np.isfinite(p)) and (not np.isfinite(z)) and (not np.isfinite(tau))
    if inval_core and (not np.isfinite(slope)):
        return 1.0, 0.0, 0.0, 0.0, 0.0, np.nan, np.nan, 0, 0

    return p, z, tau, s, var_s, slope, intercept, h, trend_code

# ============================================================
# Função pública para mapas (xarray)
# ============================================================

def _samples_per_year_from_time(tcoord: xr.DataArray) -> float:
    """
    Estima amostragem por ano para converter a slope do PMK (por amostra)
    em slope por ano. Funciona com DataArray 1D ao longo de 'time'.
    """
    if np.issubdtype(tcoord.dtype, np.datetime64):
        # anos de span (robusto a calendários)
        t0 = tcoord.isel({tcoord.dims[0]: 0}).values
        t1 = tcoord.isel({tcoord.dims[0]: -1}).values
        span_years = (t1 - t0) / np.timedelta64(1, "D") / 365.2425
        n = tcoord.sizes[tcoord.dims[0]]
        if n < 2:
            return 1.0
        # heurística para mensal
        diffs = np.diff(tcoord.values.astype("datetime64[ms]")).astype("timedelta64[ms]").astype(np.int64)
        if diffs.size > 0:
            med_days = np.median(diffs) / (1000.0 * 60.0 * 60.0 * 24.0)
            if 25.0 <= med_days <= 35.0:
                return 12.0
        if span_years > 0:
            return (n - 1) / span_years
        return 1.0
    return 1.0


def dask_mann_kendall(
    data: xr.DataArray | xr.Dataset,
    *,
    var: str | None = None,         # <- novo: quando vier Dataset, escolha a variável
    test: str = "original",
    alpha: float = 0.05,
    period: int | None = None,
    lag: int | None = None,
    time_coord: str = "time",
    engine: str = "ufunc",
    CHUNK_SPATIAL: int = 100,       # chunk opcional para dims não temporais (se desejar)
    allow_rechunk: bool = False,
    normalize_slope: bool = False,
) -> xr.Dataset:
    """
    Calcula mapas de tendência com pymannkendall (pymannkendall) a partir de um
    DataArray ou Dataset. Quando um Dataset é fornecido, use 'var' para escolher
    a variável (ou detecta automaticamente se houver exatamente uma).
    """
    if pmk is None:
        raise ImportError("pymannkendall não está instalado. Faça: pip install pymannkendall")

    # --- Seleção/validação do DataArray de entrada ---
    if isinstance(data, xr.Dataset):
        if var is None:
            if len(data.data_vars) != 1:
                raise ValueError(
                    f"Dataset possui {len(data.data_vars)} variáveis. Informe var=<nome>."
                )
            da = next(iter(data.data_vars))
            ds = data[da]
        else:
            if var not in data:
                raise ValueError(f"A variável '{var}' não existe no Dataset. Opções: {list(data.data_vars)}")
            ds = data[var]
    elif isinstance(data, xr.DataArray):
        ds = data
    else:
        raise TypeError("mk_pmk_field aceita xr.DataArray ou xr.Dataset.")

    if time_coord not in ds.dims:
        raise ValueError(f"Dimensão temporal '{time_coord}' não encontrada em ds.dims={ds.dims}")

    # --- Limpeza de NaNs só na série-alvo ---
    ds = ds.where(np.isfinite(ds))

    # --- Chunking: garante time em 1 chunk; restante mantém como está (opcionalmente aplica CHUNK_SPATIAL) ---
    # Se já existe chunking, mantemos; só forçamos time a -1.
    chunk_map = {time_coord: -1}
    # Se quiser forçar um chunk espacial padrão (opcional):
    for d in ds.dims:
        if d == time_coord:
            continue
        # aplica chunk apenas se ainda não há chunk ou se usuário quiser homogenizar
        chunk_map[d] = CHUNK_SPATIAL
    # Nota: se não quiser alterar dimensões espaciais existentes, use apenas {time_coord: -1}
    ds = ds.chunk(chunk_map)

    # --- Garante time em 1 chunk (paranoia defensiva) ---
    try:
        n_time_chunks = len(ds.chunks[ds.get_axis_num(time_coord)])
    except Exception:
        n_time_chunks = 1
    if n_time_chunks > 1:
        ds = ds.chunk({time_coord: -1})

    # --- Fator de normalização para slope (por ano) ---
    samples_per_year = _samples_per_year_from_time(ds[time_coord])

    # --- Implementação via apply_ufunc (engine ufunc) ---
    if engine == "ufunc":
        def _pmk_wrapper(y):
            # y: (T,) numpy
            p, z, tau, s, var_s, slope, intercept, h, trend_code = _pmk_call_1d(
                np.asarray(y),
                test=test,
                alpha=float(alpha),
                period=None if period is None else int(period),
                lag=None if lag is None else int(lag),
            )
            if normalize_slope and np.isfinite(slope):
                slope = slope * samples_per_year
            return (np.float32(p), np.float32(z), np.float32(tau), np.float32(s),
                    np.float32(var_s), np.float32(slope), np.float32(intercept),
                    np.uint8(h), np.int8(trend_code))

        p, z, tau, s, var_s, slope, intercept, h, trend_code = xr.apply_ufunc(
            _pmk_wrapper,
            ds,
            input_core_dims=[[time_coord]],
            output_core_dims=[[], [], [], [], [], [], [], [], []],
            vectorize=True,
            dask="parallelized",
            dask_gufunc_kwargs={"allow_rechunk": bool(allow_rechunk)},
            output_dtypes=[np.float32, np.float32, np.float32, np.float32,
                           np.float32, np.float32, np.float32, np.uint8, np.int8],
        )

    elif engine == "blocks":
        # --- função aplicada por bloco (recebe um DataArray) ---
        def _block_func(block: xr.DataArray) -> xr.Dataset:
            # ordem: time primeiro
            b = block.transpose(time_coord, ...)
            arr = b.data
            if hasattr(arr, "compute"):
                arr = arr.compute()
            arr = np.asarray(arr)

            # dims/shape
            T = arr.shape[0]
            space_dims  = tuple(d for d in b.dims if d != time_coord)
            space_shape = tuple(b.sizes[d] for d in space_dims)
            P = int(np.prod(space_shape)) if space_shape else 1

            # (T, P)
            arr2d = arr.reshape(T, P)

            # saídas planas
            out  = np.full((7, P), np.nan, dtype=np.float32)  # p,z,tau,s,var_s,slope,intercept
            hout = np.zeros(P, dtype=np.uint8)                # h
            tout = np.zeros(P, dtype=np.int8)                 # trend_code

            for idx in range(P):
                y = arr2d[:, idx]
                y = y[np.isfinite(y)]
                if y.size < 3:
                    continue
                p_, z_, tau_, s_, var_s_, slope_, inter_, h_, tcode_ = _pmk_call_1d(
                    y,
                    test=test,
                    alpha=float(alpha),
                    period=None if period is None else int(period),
                    lag=None if lag is None else int(lag),
                )
                if normalize_slope and np.isfinite(slope_):
                    slope_ = slope_ * samples_per_year

                out[0, idx] = np.float32(p_)
                out[1, idx] = np.float32(z_)
                out[2, idx] = np.float32(tau_)
                out[3, idx] = np.float32(s_)
                out[4, idx] = np.float32(var_s_)
                out[5, idx] = np.float32(slope_)
                out[6, idx] = np.float32(inter_)
                hout[idx]   = np.uint8(h_)
                tout[idx]   = np.int8(tcode_)

            # reconstrói no shape espacial
            data_vars = {
                "p":         (space_dims, out[0].reshape(space_shape)),
                "z":         (space_dims, out[1].reshape(space_shape)),
                "tau":       (space_dims, out[2].reshape(space_shape)),
                "s":         (space_dims, out[3].reshape(space_shape)),
                "var_s":     (space_dims, out[4].reshape(space_shape)),
                "slope":     (space_dims, out[5].reshape(space_shape)),
                "intercept": (space_dims, out[6].reshape(space_shape)),
                "h":         (space_dims, hout.reshape(space_shape)),
                "trend_code":(space_dims, tout.reshape(space_shape)),
            }
            # coords: SOMENTE espaciais (sem 'time')
            coords = {d: block.coords[d] for d in space_dims if d in block.coords}
            return xr.Dataset(data_vars, coords=coords)

        # ----------------- TEMPLATE DASK, SEM 'time' -----------------
        space_dims  = tuple(d for d in ds.dims if d != time_coord)
        # base espacial dask (herda chunks espaciais da ENTRADA)
        base_da = xr.zeros_like(ds.isel({time_coord: 0}))

        # garante dask chunks nas dims espaciais
        if not hasattr(base_da.data, "chunks") or base_da.data.chunks is None:
            # se necessário, aplique um chunk padrão espacial (ex.: 100)
            default = CHUNK_SPATIAL
            chunk_map = {d: default for d in space_dims}
            base_da = base_da.chunk(chunk_map)

        # constrói template a partir do base_da, tipando as variáveis
        tmpl = xr.Dataset(
            {
                "p":         base_da.astype(np.float32),
                "z":         base_da.astype(np.float32),
                "tau":       base_da.astype(np.float32),
                "s":         base_da.astype(np.float32),
                "var_s":     base_da.astype(np.float32),
                "slope":     base_da.astype(np.float32),
                "intercept": base_da.astype(np.float32),
                "h":         base_da.astype(np.uint8),
                "trend_code":base_da.astype(np.int8),
            },
            coords={d: ds.coords[d] for d in space_dims if d in ds.coords}
        )

        # *** CRÍTICO ***: remove QUALQUER coord extra (incluindo 'time') do template
        # (map_blocks exige que o resultado contenha exatamente as coord-vars do template)
        drop_coords = [c for c in list(tmpl.coords) if c not in space_dims]
        if drop_coords:
            tmpl = tmpl.drop_vars(drop_coords)

        # chama map_blocks com template consistente
        res = xr.map_blocks(_block_func, ds, template=tmpl)

        # extrai as variáveis
        p, z, tau, s, var_s, slope, intercept, h, trend_code = (
            res["p"], res["z"], res["tau"], res["s"], res["var_s"],
            res["slope"], res["intercept"], res["h"], res["trend_code"]
        )


    else:
        raise ValueError("engine deve ser 'ufunc' ou 'blocks'.")

    # --- Monta Dataset final com atributos coerentes ---
    trend = xr.Dataset(
        dict(
            p=p, z=z, tau=tau, s=s, var_s=var_s,
            slope=slope, intercept=intercept,
            h=h, trend_code=trend_code
        )
    )

    # Datas de início/fim
    tcoord = ds[time_coord]
    if np.issubdtype(tcoord.dtype, np.datetime64):
        start_str = tcoord.isel({time_coord: 0}).dt.strftime("%Y-%m-%d").item()
        end_str   = tcoord.isel({time_coord: -1}).dt.strftime("%Y-%m-%d").item()
    else:
        start_str = str(tcoord.isel({time_coord: 0}).item())
        end_str   = str(tcoord.isel({time_coord: -1}).item())

    trend.attrs.update({
        "description": f"pymannkendall '{test}' test applied per grid",
        "alpha": float(alpha),
        "period": None if period is None else int(period),
        "lag": None if lag is None else int(lag),
        "start_date": start_str,
        "end_date": end_str,
        "slope_units": "per year" if normalize_slope and np.issubdtype(tcoord.dtype, np.datetime64) else "per sample",
        "engine": engine,
        "source_var": ds.name if hasattr(ds, "name") and ds.name is not None else (var if isinstance(data, xr.Dataset) else None)
    })


    # Remove todos os atributos com valor None
    for var in trend.variables:
        trend[var].attrs = {
            k: v for k, v in trend[var].attrs.items() if v is not None
        }
    trend.attrs = {
        k: v for k, v in trend.attrs.items() if v is not None
    }

    return trend
