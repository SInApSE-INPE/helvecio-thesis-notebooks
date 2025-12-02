import glob
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import io
import os
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from matplotlib.animation import FFMpegWriter
import matplotlib as mpl
mpl.rcParams['animation.embed_limit'] = 1000_000_000  # 1GB


# # Set path for the tracking table files
# gsmap_trackingtable = '/mnt/data/tracks/gsmap_v8_2015-2024/01mmhr_v8/track/trackingtable/*.parquet'
# imerg_trackingtable = '/mnt/data/tracks/imerg_final_v7_2015-2024/01mmhr_v7/track/trackingtable/*.parquet'

gsmap_trackingtable = '/prj/cptec/helvecio.leal/tracks/gsmap/track/trackingtable/*.parquet'
imerg_trackingtable = '/prj/cptec/helvecio.leal/tracks/imerg/track/trackingtable/*.parquet'



# Mout tracks_df using parquets and sorting by timestamp
gsmap_trackingfiles = glob.glob(gsmap_trackingtable)
imerg_trackingfiles = glob.glob(imerg_trackingtable)
gsmap_tracking_timestamps = [pd.to_datetime(f.split('/')[-1][:-8] , format='%Y%m%d_%H%M') for f in gsmap_trackingfiles]
imerg_tracking_timestamps = [pd.to_datetime(f.split('/')[-1][:-8] , format='%Y%m%d_%H%M') for f in imerg_trackingfiles]
gsmap_track_df = pd.DataFrame({'path':gsmap_trackingfiles}, index=gsmap_tracking_timestamps)
imerg_track_df = pd.DataFrame({'path':imerg_trackingfiles}, index=imerg_tracking_timestamps)
# Merge the dataframes and handle the time interval differences
tracks_df = pd.merge(imerg_track_df, gsmap_track_df, left_index=True, right_index=True, how='left', suffixes=('_imerg', '_gsmap'))
tracks_df = tracks_df.sort_index()
# Forward fill the gsmap paths to handle missing 30-minute intervals
# tracks_df['path_gsmap'] = tracks_df['path_gsmap'].ffill()


def load_data(timestamp, df, bbox):
    selected_df = df.loc[timestamp]
    gsmap_path = selected_df['path_gsmap']
    imerg_path = selected_df['path_imerg']

    # Se algum dos paths for NaN, retorna GeoDataFrame vazio
    if pd.isna(gsmap_path):
        gsmap_gdf = gpd.GeoDataFrame(columns=['uid', 'geometry', 'trajectory', 'duration', 'status', 'threshold_level'], geometry='geometry', crs="EPSG:4326")
    else:
        gsmap_gdf = pd.read_parquet(gsmap_path)

    if pd.isna(imerg_path):
        imerg_gdf = gpd.GeoDataFrame(columns=['uid', 'geometry', 'trajectory', 'duration', 'status', 'threshold_level'], geometry='geometry', crs="EPSG:4326")
    else:
        imerg_gdf = pd.read_parquet(imerg_path)

    # Filtra threshold_level e duração mínima
    gsmap_gdf = gsmap_gdf[(gsmap_gdf['threshold_level'] == 0) & (gsmap_gdf['duration'] >= 120)]
    imerg_gdf = imerg_gdf[(imerg_gdf['threshold_level'] == 0) & (imerg_gdf['duration'] >= 120)]

    # Converte para GeoDataFrame
    gsmap_gdf = gpd.GeoDataFrame(gsmap_gdf, geometry=gpd.GeoSeries.from_wkt(gsmap_gdf['geometry']), crs="EPSG:4326")
    gsmap_gdf['trajectory'] = gpd.GeoSeries.from_wkt(gsmap_gdf['trajectory'], crs="EPSG:4326")
    imerg_gdf = gpd.GeoDataFrame(imerg_gdf, geometry=gpd.GeoSeries.from_wkt(imerg_gdf['geometry']), crs="EPSG:4326")
    imerg_gdf['trajectory'] = gpd.GeoSeries.from_wkt(imerg_gdf['trajectory'], crs="EPSG:4326")

    # Limita à bbox
    gsmap_gdf = gsmap_gdf.cx[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    imerg_gdf = imerg_gdf.cx[bbox[0]:bbox[1], bbox[2]:bbox[3]]

    cols_needed = ['uid', 'geometry', 'trajectory', 'duration', 'status', 'threshold_level']
    gsmap_gdf = gsmap_gdf[cols_needed]
    imerg_gdf = imerg_gdf[cols_needed]

    return timestamp, gsmap_gdf, imerg_gdf

def render_frame_as_array(frame_data, bbox, gsmap_cum, imerg_cum, times, frame_idx):
    ts, gsmap_gdf, imerg_gdf = frame_data

    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    ax_line = fig.add_subplot(gs[1, :])

    for ax in [ax1, ax2]:
        ax.add_feature(cfeature.LAND, zorder=0, facecolor="lightgray")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.set_extent(bbox, crs=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.5, linestyle='--')
        if ax == ax1:
            gl.right_labels = False
            gl.bottom_labels = True
        else:
            gl.left_labels = False
            gl.bottom_labels = True

        gl.top_labels = False

    # gsmap_gdf.plot(ax=ax1, color="blue", alpha=0.5)
    # imerg_gdf.plot(ax=ax2, color="red", alpha=0.5)

    # # Só plote se não estiver vazio
    if not gsmap_gdf.empty:
        gsmap_gdf.plot(ax=ax1, color="blue", alpha=0.5)
        gsmap_gdf['trajectory'].plot(ax=ax1, color="black", alpha=0.8, linewidth=0.5)
    if not imerg_gdf.empty:
        imerg_gdf.plot(ax=ax2, color="red", alpha=0.5)
        imerg_gdf['trajectory'].plot(ax=ax2, color="black", alpha=0.8, linewidth=0.5)

    # add trajectory plot
    if not gsmap_gdf.empty:
        gsmap_gdf['trajectory'].plot(ax=ax1, color="black", alpha=0.8, linewidth=0.5)
    if not imerg_gdf.empty:
        imerg_gdf['trajectory'].plot(ax=ax2, color="black", alpha=0.8, linewidth=0.5)

    ax1.set_title(f"a) GSMaP {ts} instantaneous PS: {len(gsmap_gdf)}", loc='left')
    ax2.set_title(f"b) IMERG {ts} instantaneous PS: {len(imerg_gdf)}", loc='left')

    ax_line.plot(times[:frame_idx+1], gsmap_cum[:frame_idx+1], color="blue", label="GSMaP")
    ax_line.plot(times[:frame_idx+1], imerg_cum[:frame_idx+1], color="red", label="IMERG")

    # gsmap_series = pd.Series(gsmap_cum, index=times)
    # imerg_series = pd.Series(imerg_cum, index=times)
    # window = 6  # 6 horas
    # gsmap_rolling = gsmap_series.rolling(window=window, min_periods=1).mean()
    # imerg_rolling = imerg_series.rolling(window=window, min_periods=1).mean()
    # ax_line.plot(times[:frame_idx+1], gsmap_rolling[:frame_idx+1], color="blue", linestyle="--", label="GSMaP 6h mean")
    # ax_line.plot(times[:frame_idx+1], imerg_rolling[:frame_idx+1], color="red", linestyle="--", label="IMERG 6h mean")


    ax_line.set_xlim(times[0], times[-1])
    # ax_line.set_ylim(0, max(gsmap_cum[-1], imerg_cum[-1]) * 1.1)
    ax_line.set_xlabel("Time")
    ax_line.set_ylabel("Cumulative count (PS)")
    ax_line.legend(loc='upper left', fontsize='small', ncol=2)
    ax_line.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_line.set_title("c) Temporal series of precipitation systems", loc='left')

    # Set xticks format for ax_line
    ax_line.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m-%d'))

    # save figure into a png in folter animation_frames
    os.makedirs("animation_frames", exist_ok=True)
    tstamp = ts.strftime("%Y%m%d_%H%M")
    fig.savefig(f"animation_frames/frame_{tstamp}.png")

    # Converte figura para array usando print_to_buffer (compatível com versões recentes)
    buf, (w, h) = fig.canvas.print_to_buffer()
    img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))[:, :, :3]  # RGB
    plt.close(fig)
    return img


def render_frame_with_index(args):
    idx, frame_data, bbox, gsmap_cum, imerg_cum, times = args
    img = render_frame_as_array(frame_data, bbox, gsmap_cum, imerg_cum, times, idx)
    return idx, img


def create_animation_mp4_parallel(df, start_period, end_period, bbox=[-180, 180, -60, 60],
                                  interval=500, n_jobs=-1, output_file="animation.mp4", fps=10):

    timestamps = df.loc[start_period:end_period].index

    # Carrega dados em paralelo
    results = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(load_data, ts, df, bbox): ts for ts in timestamps}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading data"):
            results.append(future.result())
    results = sorted(results, key=lambda x: x[0])
    times = [r[0] for r in results]

    # Contagem acumulada
    seen_gsmap, seen_imerg = set(), set()
    gsmap_acum, imerg_acum = set(), set()
    gsmap_cum, imerg_cum = [], []
    for _, gsmap_gdf, imerg_gdf in results:
        # # Instantaneous count
        # gsmap_count = (gsmap_gdf['status'] == 'NEW').sum()
        # imerg_count = (imerg_gdf['status'] == 'NEW').sum()
        # gsmap_cum.append(gsmap_count)
        # imerg_cum.append(imerg_count)
        # seen_gsmap = np.max(gsmap_cum)
        # seen_imerg = np.max(imerg_cum)

        # Acumulative unique count
        seen_gsmap.update(set(gsmap_gdf[gsmap_gdf['status'] == 'NEW']['uid']))
        seen_imerg.update(set(imerg_gdf[imerg_gdf['status'] == 'NEW']['uid']))
        gsmap_cum.append(len(seen_gsmap))
        imerg_cum.append(len(seen_imerg))

    # Renderiza frames em paralelo
    args_list = [(idx, r, bbox, gsmap_cum, imerg_cum, times) for idx, r in enumerate(results)]
    frames_out = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(render_frame_with_index, args) for args in args_list]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Rendering frames"):
            frames_out.append(f.result())

    # Ordena frames pelo índice
    frames_out = [img for idx, img in sorted(frames_out, key=lambda x: x[0])]

    # Salva direto em MP4
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('off')
    writer = FFMpegWriter(fps=fps, codec="h264", bitrate=800, extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, output_file, dpi=80):
        for img in tqdm(frames_out, desc="Saving frames"):
            ax.imshow(img)
            writer.grab_frame()
            ax.clear()
    plt.close(fig)
    print(f"Animation saved to {output_file}")

    print("Final GSMaP cum:", gsmap_cum[-1], "Max GSMaP plot:", max(gsmap_cum))
    print("Final IMERG cum:", imerg_cum[-1], "Max IMERG plot:", max(imerg_cum))
    print("Tamanho times:", len(times), "Tamanho gsmap_cum:", len(gsmap_cum))

if __name__ == "__main__":
    start_period = "2020/11/01 00:00:00"
    end_period   = "2020/12/01 00:30:00"
    bbox = [-74.0, -34.0, -18.0, 5.0]

    os.makedirs("anim_tracks", exist_ok=True)

    create_animation_mp4_parallel(
        tracks_df,
        start_period=start_period,
        end_period=end_period,
        bbox=bbox,
        interval=50,
        n_jobs=30,
        output_file=f"anim_tracks/track_dur_S{start_period.replace('/', '').replace(':', '').replace(' ','_')}_E{end_period.replace('/', '').replace(':', '').replace(' ','_')}.mp4",
        fps=20
    )
