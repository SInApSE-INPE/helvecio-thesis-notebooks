import numpy as np, pandas as pd, pathlib, folium
from matplotlib import cm, colors
import matplotlib
import branca.colormap as bcm
import xarray as xr
import glob
import base64
from PIL import Image
import io
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import json

def read_gsmap(path):
    ds = xr.open_dataset(path, engine='netcdf4', decode_times=False)
    ds = ds['hourlyPrecipRate']
    ds = ds.sel(Latitude=slice(-90,90)).data[0]
    ds[ds < 0.1] = np.nan
    return ds

def read_imerg(path):
    ds = xr.open_dataset(path, engine='netcdf4')
    data = ds['precipitation'][0]
    data = np.rot90(data, k=1)
    data[data < 0.1] = np.nan
    return data[::-1]

def to_rgba(data, vmin=0, vmax=20, cmap_name='turbo'):
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgba = (cmap(norm(data)) * 255).astype('uint8')
    return np.flipud(rgba)

def rgba_to_base64(rgba_array, quality=75, scale_factor=1.0, format='WEBP'):
    """Converte array RGBA para string base64 com otimizações de tamanho
    
    Parameters:
    -----------
    rgba_array : np.ndarray
        Array RGBA a ser convertido
    quality : int
        Qualidade da compressão (1-100). Menor = arquivo menor
    scale_factor : float
        Fator de escala da imagem (0.5 = metade da resolução)
    format : str
        Formato de imagem: 'WEBP', 'PNG', ou 'JPEG'
    """
    img = Image.fromarray(rgba_array)
    
    # Reduz resolução se necessário
    if scale_factor != 1.0:
        new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    buffer = io.BytesIO()
    
    if format == 'WEBP':
        img.save(buffer, format='WEBP', quality=quality, method=6)
    elif format == 'JPEG':
        # Converte RGBA para RGB para JPEG
        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
        rgb_img.save(buffer, format='JPEG', quality=quality, optimize=True)
    else:  # PNG
        img.save(buffer, format='PNG', optimize=True, compress_level=9)
    
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def get_timestamp_from_file(filepath, timestamp_format=None, position=None):
    """
    Extrai timestamp do nome do arquivo de forma flexível
    
    Parameters:
    -----------
    filepath : str
        Caminho do arquivo
    timestamp_format : str
        Formato strftime para extrair timestamp (ex: '%Y%m%d.%H%M')
    position : list or tuple, optional
        [start, end] - Posições na string do filename para extrair timestamp
        Se fornecido, usa filename[start:end] antes de aplicar o formato
    
    Returns:
    --------
    str : Timestamp formatado como 'YYYY-MM-DD HH:MM UTC'
    """
    filename = pathlib.Path(filepath).stem
    
    if position is not None:
        # Extrai substring nas posições especificadas
        timestamp_str = filename[position[0]:position[1]]
    else:
        # Tenta extrair a parte do timestamp baseado no formato
        # Para GSMaP: gsmap_mvk.20200101.0000.v8.0000.0 -> pega as partes 1 e 2
        parts = filename.split('.')
        if len(parts) >= 3 and timestamp_format == '%Y%m%d.%H%M':
            # Assume formato GSMaP: nome.YYYYMMDD.HHMM.resto
            timestamp_str = f"{parts[1]}.{parts[2]}"
        else:
            timestamp_str = filename
    
    # Parse o timestamp
    ts = pd.to_datetime(timestamp_str, format=timestamp_format)
    return ts.strftime('%Y-%m-%d %H:%M UTC')


def match_timestamps(gsmap_files, imerg_files, 
                     gsmap_format='%Y%m%d.%H%M', gsmap_position=None,
                     imerg_format='%Y%m%d-S%H%M%S', imerg_position=None,
                     gsmap_repeat=2):
    """
    Sincroniza arquivos GSMaP e IMERG por timestamp
    
    Parameters:
    -----------
    gsmap_files : list
        Lista de caminhos dos arquivos GSMaP
    imerg_files : list
        Lista de caminhos dos arquivos IMERG
    gsmap_format : str
        Formato strftime do timestamp GSMaP
    gsmap_position : list or tuple, optional
        [start, end] para extrair timestamp do filename GSMaP
    imerg_format : str
        Formato strftime do timestamp IMERG
    imerg_position : list or tuple, optional
        [start, end] para extrair timestamp do filename IMERG
    gsmap_repeat : int
        Número de vezes para repetir cada arquivo GSMaP (padrão: 2)
        Útil quando GSMaP tem resolução temporal diferente do IMERG
        Ex: GSMaP 60min, IMERG 30min -> gsmap_repeat=2
    
    Returns:
    --------
    tuple : (gsmap_synced, imerg_synced, timestamps)
        Listas sincronizadas de arquivos e timestamps
    """
    print("Sincronizando arquivos por timestamp...")
    
    # Extrai timestamps de cada arquivo
    gsmap_data = []
    for f in gsmap_files:
        try:
            ts_str = get_timestamp_from_file(f, gsmap_format, gsmap_position)
            ts = pd.to_datetime(ts_str, format='%Y-%m-%d %H:%M UTC')
            gsmap_data.append((f, ts, ts_str))
        except Exception as e:
            print(f"Aviso: Não foi possível processar {pathlib.Path(f).name}: {e}")
    
    imerg_data = []
    for f in imerg_files:
        try:
            ts_str = get_timestamp_from_file(f, imerg_format, imerg_position)
            ts = pd.to_datetime(ts_str, format='%Y-%m-%d %H:%M UTC')
            imerg_data.append((f, ts, ts_str))
        except Exception as e:
            print(f"Aviso: Não foi possível processar {pathlib.Path(f).name}: {e}")
    
    # Se gsmap_repeat > 1, duplica cada arquivo GSMaP
    if gsmap_repeat > 1:
        expanded_gsmap = []
        for f, ts, ts_str in gsmap_data:
            for _ in range(gsmap_repeat):
                expanded_gsmap.append((f, ts, ts_str))
        gsmap_data = expanded_gsmap
        print(f"GSMaP expandido {gsmap_repeat}x: {len(gsmap_data)} entradas")
    
    # Ordena ambas as listas por timestamp
    gsmap_data.sort(key=lambda x: x[1])
    imerg_data.sort(key=lambda x: x[1])
    
    print(f"Arquivos GSMaP (original): {len(gsmap_files)}")
    print(f"Arquivos GSMaP (após expansão): {len(gsmap_data)}")
    print(f"Arquivos IMERG: {len(imerg_files)}")
    
    # Usa o tamanho mínimo para sincronização
    n_frames = min(len(gsmap_data), len(imerg_data))
    
    if n_frames == 0:
        print("AVISO: Nenhum arquivo para sincronizar!")
        return [], [], []
    
    # Cria listas sincronizadas pegando os primeiros n_frames de cada
    gsmap_synced = [gsmap_data[i][0] for i in range(n_frames)]
    imerg_synced = [imerg_data[i][0] for i in range(n_frames)]
    timestamps = [imerg_data[i][2] for i in range(n_frames)]  # Usa timestamp do IMERG (mais preciso)
    
    print(f"Frames sincronizados: {n_frames}")
    
    return gsmap_synced, imerg_synced, timestamps

def load_geojson_for_timestamp(timestamp_str, gsmap_track_dir, imerg_track_dir, layer_type):
    """
    Carrega arquivo GeoJSON para um timestamp específico
    
    Parameters:
    -----------
    timestamp_str : str
        Timestamp no formato 'YYYY-MM-DD HH:MM UTC'
    gsmap_track_dir : str
        Diretório base dos tracks GSMaP (ex: /storage/tracks/gsmap_v8_2015-2024/01mmhr_v8/track/geometry/)
    imerg_track_dir : str
        Diretório base dos tracks IMERG
    layer_type : str
        Tipo de camada: 'vector_field', 'trajectory', ou 'boundary'
    
    Returns:
    --------
    tuple : (gsmap_geojson, imerg_geojson) - Ambos como dicts Python ou None se não encontrado
    """
    # Converte timestamp para formato de arquivo: YYYYMMDD_HHMM
    ts = pd.to_datetime(timestamp_str, format='%Y-%m-%d %H:%M UTC')
    filename = ts.strftime('%Y%m%d_%H%M') + '.GeoJSON'
    
    gsmap_path = pathlib.Path(gsmap_track_dir) / layer_type / filename
    imerg_path = pathlib.Path(imerg_track_dir) / layer_type / filename
    
    gsmap_data = None
    imerg_data = None
    
    # Carrega GSMaP GeoJSON
    if gsmap_path.exists():
        try:
            with open(gsmap_path, 'r') as f:
                gsmap_data = json.load(f)
        except Exception as e:
            print(f"Aviso: Erro ao carregar {gsmap_path}: {e}")
    
    # Carrega IMERG GeoJSON
    if imerg_path.exists():
        try:
            with open(imerg_path, 'r') as f:
                imerg_data = json.load(f)
        except Exception as e:
            print(f"Aviso: Erro ao carregar {imerg_path}: {e}")
    
    return gsmap_data, imerg_data

def process_frame(args):
    """Processa um frame individual - usado para paralelização"""
    i, gsmap_file, imerg_file, timestamp, vmin, vmax, quality, scale_factor, img_format = args
    
    try:
        # Carrega os dados
        gsm_data = read_gsmap(gsmap_file)
        img_data = read_imerg(imerg_file)
        
        # Converte para RGBA
        gsm_rgba = to_rgba(gsm_data, vmin, vmax, 'turbo')
        img_rgba = to_rgba(img_data, vmin, vmax, 'turbo')
        
        # Converte para base64 com otimizações
        gsmap_b64 = rgba_to_base64(gsm_rgba, quality=quality, scale_factor=scale_factor, format=img_format)
        imerg_b64 = rgba_to_base64(img_rgba, quality=quality, scale_factor=scale_factor, format=img_format)
        
        return {
            'index': i,
            'gsmap_img': gsmap_b64,
            'imerg_img': imerg_b64,
            'timestamp': timestamp
        }
    except Exception as e:
        print(f"Erro ao processar frame {i}: {e}")
        return None

def create_animated_dual_map(gsmap_files, imerg_files, timestamps,
                              output_file="gsmap_imerg_animated.html",
                              extent=[-180, 180, -60, 60], vmin=0, vmax=20, n_workers=None,
                              gsmap_track_dir=None, imerg_track_dir=None,
                              enable_tracks=False,
                              image_quality=65, image_scale=0.75, image_format='WEBP'):
    """
    Cria um HTML interativo com slider temporal para comparar GSMaP e IMERG
    
    Parameters:
    -----------
    gsmap_files : list
        Lista de caminhos dos arquivos GSMaP (já sincronizados)
    imerg_files : list
        Lista de caminhos dos arquivos IMERG (já sincronizados)
    timestamps : list
        Lista de timestamps correspondentes (strings formatadas)
    output_file : str
        Nome do arquivo HTML de saída
    extent : list
        [lon_min, lon_max, lat_min, lat_max]
    vmin, vmax : float
        Limites da escala de cores
    n_workers : int, optional
        Número de processos paralelos. Se None, usa cpu_count()-1
    gsmap_track_dir : str, optional
        Diretório base dos tracks GSMaP (/path/to/geometry/)
    imerg_track_dir : str, optional
        Diretório base dos tracks IMERG (/path/to/geometry/)
    enable_tracks : bool
        Se True, carrega e adiciona camadas GeoJSON de tracks
    image_quality : int
        Qualidade de compressão das imagens (1-100). Valores menores = arquivo menor
        Recomendado: 50-75 para balanço qualidade/tamanho
    image_scale : float
        Fator de escala espacial (0.5-1.0). Valores menores = arquivo menor
        Recomendado: 0.5-0.75 para redução significativa
    image_format : str
        Formato das imagens: 'WEBP' (melhor compressão), 'PNG', ou 'JPEG'
    """
    if len(gsmap_files) != len(imerg_files) != len(timestamps):
        raise ValueError("gsmap_files, imerg_files e timestamps devem ter o mesmo tamanho!")
    
    lon_min, lon_max, lat_min, lat_max = extent
    bounds = [[lat_min, lon_min], [lat_max, lon_max]]
    center = [(lat_min + lat_max) / 2, (lon_min + lon_max) / 2]
    
    n_frames = len(gsmap_files)
    print(f"Processando {n_frames} frames usando multiprocessing...")
    
    # Define número de workers
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    print(f"Usando {n_workers} processos paralelos")
    
    # Prepara argumentos para processamento paralelo
    args_list = [(i, gsmap_files[i], imerg_files[i], timestamps[i], vmin, vmax, 
                  image_quality, image_scale, image_format) for i in range(n_frames)]
    
    # Processa frames em paralelo
    results = []
    print(f"Configuração de compressão: formato={image_format}, qualidade={image_quality}, escala={image_scale}")
    with Pool(processes=n_workers) as pool:
        for result in tqdm(pool.imap(process_frame, args_list), total=n_frames, desc="Processando frames"):
            if result is not None:
                results.append(result)
    
    # Ordena resultados pelo índice (caso estejam fora de ordem)
    results.sort(key=lambda x: x['index'])
    
    # Extrai listas separadas
    gsmap_images = [r['gsmap_img'] for r in results]
    imerg_images = [r['imerg_img'] for r in results]
    result_timestamps = [r['timestamp'] for r in results]
    
    # Carrega dados GeoJSON se habilitado
    gsmap_tracks_data = {'vector_field': [], 'trajectory': [], 'boundary': []}
    imerg_tracks_data = {'vector_field': [], 'trajectory': [], 'boundary': []}
    
    if enable_tracks and gsmap_track_dir and imerg_track_dir:
        print("Carregando arquivos GeoJSON...")
        for ts in tqdm(result_timestamps, desc="Carregando GeoJSON"):
            for layer_type in ['vector_field', 'trajectory', 'boundary']:
                gsmap_geo, imerg_geo = load_geojson_for_timestamp(
                    ts, gsmap_track_dir, imerg_track_dir, layer_type
                )
                gsmap_tracks_data[layer_type].append(gsmap_geo)
                imerg_tracks_data[layer_type].append(imerg_geo)
    
    # Cria o HTML customizado com dois mapas Leaflet sincronizados
    html_template = f'''
<!DOCTYPE html>
<html>
<head>
    <title>GSMaP vs IMERG - Comparação Temporal</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: Arial, sans-serif; }}
        
        .container {{ display: flex; flex-direction: column; height: 100vh; }}
        
        .maps-container {{
            display: flex;
            flex: 1;
            gap: 10px;
            padding: 10px;
            background: #f0f0f0;
        }}
        
        .map-wrapper {{
            flex: 1;
            position: relative;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }}
        
        .map {{ height: 100%; width: 100%; }}
        
        .map-title {{
            position: absolute;
            top: 10px;
            left: 50px;
            z-index: 1000;
            background: rgba(255,255,255,0.9);
            padding: 8px 15px;
            border-radius: 5px;
            font-weight: bold;
            font-size: 14px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }}
        
        .controls {{
            background: white;
            padding: 15px 20px;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        }}
        
        .controls-inner {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .timestamp-display {{
            display: flex;
            justify-content: space-around;
            align-items: center;
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
            gap: 20px;
        }}
        
        .timestamp-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .timestamp-label {{
            font-size: 14px;
            color: #666;
            font-weight: normal;
        }}
        
        .slider-container {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        
        .slider {{
            flex: 1;
            height: 8px;
            -webkit-appearance: none;
            background: #ddd;
            border-radius: 4px;
            outline: none;
        }}
        
        .slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            background: #4CAF50;
            border-radius: 50%;
            cursor: pointer;
        }}
        
        .btn {{
            padding: 10px 20px;
            font-size: 14px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.3s;
        }}
        
        .btn-play {{ background: #4CAF50; color: white; }}
        .btn-play:hover {{ background: #45a049; }}
        .btn-play.playing {{ background: #f44336; }}
        
        .speed-control {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        
        .speed-control select {{
            padding: 8px;
            border-radius: 5px;
            border: 1px solid #ddd;
        }}
        
        .frame-info {{
            font-size: 12px;
            color: #666;
            min-width: 80px;
            text-align: right;
        }}
        
        .colorbar {{
            position: fixed;
            top: 50%;
            transform: translateY(-50%);
            right: 20px;
            background: rgba(255,255,255,0.95);
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            z-index: 1000;
        }}
        
        .colorbar-gradient {{
            width: 30px;
            height: 200px;
            background: linear-gradient(to top, 
                #30123b, #4662d7, #36aaf9, #1ae4b6, 
                #72fe5e, #c8ef34, #faba39, #f66b19, 
                #ca2a04, #7a0403);
            border-radius: 3px;
        }}
        
        .colorbar-labels {{
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            height: 200px;
            margin-left: 5px;
            font-size: 11px;
        }}
        
        .colorbar-title {{
            text-align: center;
            font-size: 11px;
            margin-top: 5px;
            font-weight: bold;
        }}
        
        .colorbar-inner {{
            display: flex;
        }}
        
        .layer-controls {{
            position: fixed;
            top: 120px;
            left: 10px;
            background: rgba(255,255,255,0.95);
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            z-index: 1000;
            font-size: 12px;
            max-width: 200px;
        }}
        
        .layer-controls h4 {{
            margin: 0 0 10px 0;
            font-size: 13px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }}
        
        .layer-section {{
            margin-bottom: 10px;
        }}
        
        .layer-section h5 {{
            margin: 5px 0;
            font-size: 11px;
            color: #666;
            font-weight: bold;
        }}
        
        .layer-toggle {{
            display: flex;
            align-items: center;
            margin: 3px 0;
            cursor: pointer;
        }}
        
        .layer-toggle input {{
            margin-right: 6px;
            cursor: pointer;
        }}
        
        .layer-toggle label {{
            cursor: pointer;
            user-select: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="maps-container">
            <div class="map-wrapper">
                <div class="map-title" id="title-gsmap">GSMaP</div>
                <div id="map1" class="map"></div>
            </div>
            <div class="map-wrapper">
                <div class="map-title" id="title-imerg">IMERG</div>
                <div id="map2" class="map"></div>
            </div>
        </div>
        
        <div class="controls">
            <div class="controls-inner">
                <div class="timestamp-display">
                    <div class="timestamp-item">
                        <span class="timestamp-label">GSMaP:</span>
                        <span id="timestamp-gsmap"></span>
                    </div>
                    <div class="timestamp-item">
                        <span class="timestamp-label">IMERG:</span>
                        <span id="timestamp-imerg"></span>
                    </div>
                </div>
                <div class="slider-container">
                    <button class="btn btn-play" id="playBtn" onclick="togglePlay()">▶ Play</button>
                    <input type="range" class="slider" id="timeSlider" min="0" max="{n_frames - 1}" value="0" oninput="updateFrame(this.value)">
                    <div class="speed-control">
                        <label>Speed:</label>
                        <select id="speedSelect" onchange="updateSpeed()">
                            <option value="2000">0.5x</option>
                            <option value="1000" selected>1x</option>
                            <option value="500">2x</option>
                            <option value="250">4x</option>
                        </select>
                    </div>
                    <div class="frame-info" id="frameInfo"></div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="colorbar">
        <div class="colorbar-inner">
            <div class="colorbar-gradient"></div>
            <div class="colorbar-labels">
                <span>{vmax}</span>
                <span>{(vmax+vmin)//2}</span>
                <span>{vmin}</span>
            </div>
        </div>
        <div class="colorbar-title">mm/h</div>
    </div>
    
    <div class="layer-controls" id="layerControls" style="display: {'block' if enable_tracks else 'none'};">
        <h4>Camadas</h4>
        <div class="layer-section">
            <h5>GSMaP</h5>
            <div class="layer-toggle">
                <input type="checkbox" id="toggle-gsmap-vector" checked onchange="toggleLayer('gsmap', 'vector_field')">
                <label for="toggle-gsmap-vector">Vector Field</label>
            </div>
            <div class="layer-toggle">
                <input type="checkbox" id="toggle-gsmap-traj" checked onchange="toggleLayer('gsmap', 'trajectory')">
                <label for="toggle-gsmap-traj">Trajectory</label>
            </div>
            <div class="layer-toggle">
                <input type="checkbox" id="toggle-gsmap-bound" checked onchange="toggleLayer('gsmap', 'boundary')">
                <label for="toggle-gsmap-bound">Boundary</label>
            </div>
        </div>
        <div class="layer-section">
            <h5>IMERG</h5>
            <div class="layer-toggle">
                <input type="checkbox" id="toggle-imerg-vector" checked onchange="toggleLayer('imerg', 'vector_field')">
                <label for="toggle-imerg-vector">Vector Field</label>
            </div>
            <div class="layer-toggle">
                <input type="checkbox" id="toggle-imerg-traj" checked onchange="toggleLayer('imerg', 'trajectory')">
                <label for="toggle-imerg-traj">Trajectory</label>
            </div>
            <div class="layer-toggle">
                <input type="checkbox" id="toggle-imerg-bound" checked onchange="toggleLayer('imerg', 'boundary')">
                <label for="toggle-imerg-bound">Boundary</label>
            </div>
        </div>
    </div>

    <script>
        // Dados das imagens em base64
        const gsmapImages = {gsmap_images};
        const imergImages = {imerg_images};
        const timestamps = {result_timestamps};
        const n_frames = gsmapImages.length;
        
        // Dados GeoJSON
        const gsmapTracksData = {json.dumps(gsmap_tracks_data)};
        const imergTracksData = {json.dumps(imerg_tracks_data)};
        const tracksEnabled = {str(enable_tracks).lower()};
        
        const bounds = [[{lat_min}, {lon_min}], [{lat_max}, {lon_max}]];
        const center = [{center[0]}, {center[1]}];
        
        // Inicializa os mapas
        const map1 = L.map('map1', {{ zoomControl: true }}).setView(center, 2);
        const map2 = L.map('map2', {{ zoomControl: true }}).setView(center, 2);
        
        // Camadas base
        const oceanTiles1 = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            attribution: 'Esri Ocean'
        }}).addTo(map1);
        
        const oceanTiles2 = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            attribution: 'Esri Ocean'
        }}).addTo(map2);
        
        // Camadas de fronteiras
        const boundaries1 = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            attribution: 'Esri Boundaries'
        }});
        
        const boundaries2 = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            attribution: 'Esri Boundaries'
        }});
        
        // Overlays de imagem
        let gsmapOverlay = null;
        let imergOverlay = null;
        
        // Camadas GeoJSON
        let gsmapLayers = {{
            vector_field: null,
            trajectory: null,
            boundary: null
        }};
        let imergLayers = {{
            vector_field: null,
            trajectory: null,
            boundary: null
        }};
        
        // Estado de visibilidade das camadas
        let layerVisibility = {{
            gsmap: {{ vector_field: true, trajectory: true, boundary: true }},
            imerg: {{ vector_field: true, trajectory: true, boundary: true }}
        }};
        
        // Sincronização dos mapas
        map1.on('move', function() {{
            map2.setView(map1.getCenter(), map1.getZoom(), {{ animate: false }});
        }});
        
        map2.on('move', function() {{
            map1.setView(map2.getCenter(), map2.getZoom(), {{ animate: false }});
        }});
        
        // Adiciona grade de coordenadas
        function addGraticule(map) {{
            const latInterval = 30;
            const lonInterval = 60;
            
            // Linhas de latitude
            for (let lat = -60; lat <= 60; lat += latInterval) {{
                L.polyline([[lat, -180], [lat, 180]], {{
                    color: 'gray',
                    weight: 0.8,
                    opacity: 0.6,
                    dashArray: '5,5'
                }}).addTo(map);
                
                // Label
                L.marker([lat, -175], {{
                    icon: L.divIcon({{
                        className: 'coord-label',
                        html: '<div style="font-size:10px;color:#333;font-weight:bold;background:rgba(255,255,255,0.7);padding:1px 3px;border-radius:2px;">' + lat + '°</div>',
                        iconSize: [50, 20],
                        iconAnchor: [0, 10]
                    }})
                }}).addTo(map);
            }}
            
            // Linhas de longitude
            for (let lon = -180; lon <= 180; lon += lonInterval) {{
                L.polyline([[-60, lon], [60, lon]], {{
                    color: 'gray',
                    weight: 0.8,
                    opacity: 0.6,
                    dashArray: '5,5'
                }}).addTo(map);
                
                // Label
                L.marker([-55, lon], {{
                    icon: L.divIcon({{
                        className: 'coord-label',
                        html: '<div style="font-size:10px;color:#333;font-weight:bold;background:rgba(255,255,255,0.7);padding:1px 3px;border-radius:2px;">' + lon + '°</div>',
                        iconSize: [50, 20],
                        iconAnchor: [25, 0]
                    }})
                }}).addTo(map);
            }}
        }}
        
        addGraticule(map1);
        addGraticule(map2);
        
        // Adiciona fronteiras por cima
        boundaries1.addTo(map1);
        boundaries2.addTo(map2);
        
        // Função para criar popup com propriedades do GeoJSON
        function createPopupContent(properties) {{
            let html = '<div style="max-height:200px;overflow-y:auto;">';
            html += '<table style="font-size:11px;border-collapse:collapse;">';
            for (let key in properties) {{
                html += '<tr><td style="padding:2px;font-weight:bold;">' + key + ':</td>';
                html += '<td style="padding:2px;">' + properties[key] + '</td></tr>';
            }}
            html += '</table></div>';
            return html;
        }}
        
        // Função para estilizar GeoJSON features
        function getFeatureStyle(layerType) {{
            if (layerType === 'vector_field') {{
                return {{
                    color: '#FF6B00',
                    weight: 2,
                    opacity: 0.8
                }};
            }} else if (layerType === 'trajectory') {{
                return {{
                    color: '#0066FF',
                    weight: 3,
                    opacity: 0.9,
                    dashArray: '5, 5'
                }};
            }} else if (layerType === 'boundary') {{
                return {{
                    color: '#FF0066',
                    weight: 2,
                    opacity: 0.7,
                    fillOpacity: 0.1,
                    fillColor: '#FF0066'
                }};
            }}
            return {{ color: '#000', weight: 2 }};
        }}
        
        // Função para adicionar camada GeoJSON ao mapa
        function addGeoJSONLayer(map, geojsonData, layerType, mapType) {{
            if (!geojsonData || !geojsonData.features || geojsonData.features.length === 0) {{
                return null;
            }}
            
            const layer = L.geoJSON(geojsonData, {{
                style: getFeatureStyle(layerType),
                onEachFeature: function(feature, layer) {{
                    if (feature.properties) {{
                        layer.bindPopup(createPopupContent(feature.properties));
                    }}
                }}
            }});
            
            // Adiciona ao mapa se a visibilidade estiver ativada
            if (layerVisibility[mapType][layerType]) {{
                layer.addTo(map);
            }}
            
            return layer;
        }}
        
        // Função para alternar visibilidade das camadas
        function toggleLayer(mapType, layerType) {{
            layerVisibility[mapType][layerType] = !layerVisibility[mapType][layerType];
            
            const map = mapType === 'gsmap' ? map1 : map2;
            const layers = mapType === 'gsmap' ? gsmapLayers : imergLayers;
            
            if (layers[layerType]) {{
                if (layerVisibility[mapType][layerType]) {{
                    layers[layerType].addTo(map);
                }} else {{
                    map.removeLayer(layers[layerType]);
                }}
            }}
        }}
        
        // Variáveis de controle de animação
        let currentFrame = 0;
        let isPlaying = false;
        let playInterval = null;
        let speed = 1000;
        
        function updateFrame(frameIndex) {{
            currentFrame = parseInt(frameIndex);
            
            // Remove overlays anteriores
            if (gsmapOverlay) map1.removeLayer(gsmapOverlay);
            if (imergOverlay) map2.removeLayer(imergOverlay);
            
            // Remove camadas GeoJSON anteriores
            if (tracksEnabled) {{
                for (let layerType of ['vector_field', 'trajectory', 'boundary']) {{
                    if (gsmapLayers[layerType]) {{
                        map1.removeLayer(gsmapLayers[layerType]);
                        gsmapLayers[layerType] = null;
                    }}
                    if (imergLayers[layerType]) {{
                        map2.removeLayer(imergLayers[layerType]);
                        imergLayers[layerType] = null;
                    }}
                }}
            }}
            
            // Adiciona novos overlays
            gsmapOverlay = L.imageOverlay('data:image/{image_format.lower()};base64,' + gsmapImages[currentFrame], bounds, {{
                opacity: 0.85,
                interactive: false
            }}).addTo(map1);
            
            imergOverlay = L.imageOverlay('data:image/{image_format.lower()};base64,' + imergImages[currentFrame], bounds, {{
                opacity: 0.85,
                interactive: false
            }}).addTo(map2);
            
            // Adiciona novas camadas GeoJSON
            if (tracksEnabled) {{
                for (let layerType of ['vector_field', 'trajectory', 'boundary']) {{
                    if (gsmapTracksData[layerType][currentFrame]) {{
                        gsmapLayers[layerType] = addGeoJSONLayer(
                            map1, 
                            gsmapTracksData[layerType][currentFrame],
                            layerType,
                            'gsmap'
                        );
                    }}
                    if (imergTracksData[layerType][currentFrame]) {{
                        imergLayers[layerType] = addGeoJSONLayer(
                            map2,
                            imergTracksData[layerType][currentFrame],
                            layerType,
                            'imerg'
                        );
                    }}
                }}
            }}
            
            // Garante que as fronteiras fiquem por cima
            boundaries1.bringToFront();
            boundaries2.bringToFront();
            
            // Atualiza UI
            document.getElementById('timestamp-gsmap').textContent = timestamps[currentFrame];
            document.getElementById('timestamp-imerg').textContent = timestamps[currentFrame];
            document.getElementById('timeSlider').value = currentFrame;
            document.getElementById('frameInfo').textContent = (currentFrame + 1) + ' / ' + n_frames;
            document.getElementById('title-gsmap').textContent = 'GSMaP - ' + timestamps[currentFrame];
            document.getElementById('title-imerg').textContent = 'IMERG - ' + timestamps[currentFrame];
        }}
        
        function togglePlay() {{
            const btn = document.getElementById('playBtn');
            if (isPlaying) {{
                clearInterval(playInterval);
                btn.textContent = '▶ Play';
                btn.classList.remove('playing');
                isPlaying = false;
            }} else {{
                btn.textContent = '⏸ Pause';
                btn.classList.add('playing');
                isPlaying = true;
                playInterval = setInterval(function() {{
                    currentFrame = (currentFrame + 1) % n_frames;
                    updateFrame(currentFrame);
                }}, speed);
            }}
        }}
        
        function updateSpeed() {{
            speed = parseInt(document.getElementById('speedSelect').value);
            if (isPlaying) {{
                clearInterval(playInterval);
                playInterval = setInterval(function() {{
                    currentFrame = (currentFrame + 1) % n_frames;
                    updateFrame(currentFrame);
                }}, speed);
            }}
        }}
        
        // Inicializa com o primeiro frame
        updateFrame(0);
    </script>
</body>
</html>
'''
    
    # Salva o HTML
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    # Calcula estatísticas do arquivo
    file_size = pathlib.Path(output_file).stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    avg_frame_size_kb = (file_size / n_frames) / 1024
    
    print(f"\n{'='*60}")
    print(f"Arquivo salvo: {output_file}")
    print(f"Tamanho total: {file_size_mb:.2f} MB")
    print(f"Número de frames: {n_frames}")
    print(f"Tamanho médio por frame: {avg_frame_size_kb:.2f} KB")
    print(f"Formato: {image_format}, Qualidade: {image_quality}, Escala: {image_scale}")
    print(f"{'='*60}\n")
    
    return output_file


# =============================================================================
# EXECUÇÃO
# =============================================================================
if __name__ == "__main__":
    GSMAP_PATH = '/storage/precipitation/jaxa/v8/2020/01/01/'
    IMERG_PATH = '/storage/precipitation/imerg/final_v7/2020/01/01/'

    GSMAP_FILES = sorted(glob.glob(GSMAP_PATH + '*.nc'))
    IMERG_FILES = sorted(glob.glob(IMERG_PATH + '*.nc4'))
    
    # Configuração de formatos de timestamp
    # GSMaP: gsmap_mvk.20200101.0000.v8.0000.0.nc
    # Timestamp está nas posições 10-23 da string: "20200101.0000"
    gsmap_format = '%Y%m%d.%H%M'
    gsmap_position = [10, 23]  # "gsmap_mvk.20200101.0000" -> posições do timestamp
    
    # IMERG: 3B-HHR.MS.MRG.3IMERG.20200101-S000000-E002959.0000.V07B.HDF5
    # Timestamp está nas posições 21 a 37: "20200101-S000000"
    imerg_format = '%Y%m%d-S%H%M%S'
    imerg_position = [21, 37]
    
    # Sincroniza arquivos por timestamp
    # gsmap_repeat=2 porque GSMaP é 60min e IMERG é 30min
    # Cada arquivo GSMaP será usado para 2 frames IMERG consecutivos
    gsmap_synced, imerg_synced, timestamps = match_timestamps(
        GSMAP_FILES, 
        IMERG_FILES,
        gsmap_format=gsmap_format,
        gsmap_position=gsmap_position,
        imerg_format=imerg_format,
        imerg_position=imerg_position,
        gsmap_repeat=2  # Duplica cada arquivo GSMaP
    )
    
    if len(gsmap_synced) == 0:
        print("Erro: Nenhum arquivo sincronizado. Verifique os formatos de timestamp.")
        exit(1)
    
    # Cria o HTML animado com todos os frames sincronizados
    # Configurações otimizadas para reduzir tamanho do arquivo:
    # - image_quality: 50-75 (padrão 65) - menor = arquivo menor
    # - image_scale: 0.5-1.0 (padrão 0.75) - menor = arquivo menor
    # - image_format: 'WEBP' (melhor), 'PNG', ou 'JPEG'
    create_animated_dual_map(
        gsmap_synced,
        imerg_synced,
        timestamps,
        output_file="gsmap_imerg_animated.html",
        extent=[-180, 180, -60, 60],
        vmin=0, 
        vmax=20,
        n_workers=None,  # None = usa cpu_count()-1, ou especifique um número
        image_quality=50,  # Ajuste 50-80 para balancear qualidade/tamanho
        image_scale=1,  # Ajuste 0.5-1.0 para balancear resolução/tamanho
        image_format='WEBP'  # WEBP oferece melhor compressão que PNG
    )