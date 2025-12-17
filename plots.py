import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geobr
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
from cartopy.feature import ShapelyFeature
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LogNorm, TwoSlopeNorm, Normalize, BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.path import Path
import string
import gc
import xarray as xr
import geopandas as gpd
import pandas as pd
from IPython.display import HTML
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from matplotlib.ticker import NullFormatter
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.patches import Wedge, Rectangle
from shapely.geometry import Point
from shapely import wkt

def plot_comparison_maps(datasets, titles=None, nrows=3, ncols=2, figsize=(18, 18), 
                         cmaps=None, vlims=None, cbar_labels=None, cbar_extend='both',
                         cbar_shrink=0.2, cbar_pad=0.03, ylim=(-60, 60), 
                         title_prefixes=None, hspace=-0.75, wspace=0.005,
                         add_features=True, save_path=None, cbar_ticks_notation=None,
                         cbar_ticks=None, cmap_num=None):
        
    # Verificar se os dados são suficientes
    n_plots = nrows * ncols
    if len(datasets) != n_plots:
        raise ValueError(f"Número de datasets ({len(datasets)}) deve ser igual a nrows*ncols ({n_plots})")
    
    # Configurar defaults para parâmetros opcionais
    if titles is None:
        titles = [f"Plot {i+1}" for i in range(n_plots)]
    
    if title_prefixes is None:
        title_prefixes = list(string.ascii_lowercase[:n_plots])
    
    if cmaps is None:
        cmaps = ['viridis'] * n_plots
    elif isinstance(cmaps, str):
        cmaps = [cmaps] * n_plots
    
    if vlims is None:
        vlims = [(None, None)] * n_plots
    
    if cbar_labels is None:
        cbar_labels = [""] * n_plots
    
    if isinstance(cbar_extend, str):
        cbar_extend = [cbar_extend] * n_plots
    
    if cbar_ticks_notation is None:
        cbar_ticks_notation = [False] * n_plots
    elif isinstance(cbar_ticks_notation, bool):
        cbar_ticks_notation = [cbar_ticks_notation] * n_plots
    
    if cbar_ticks is None:
        cbar_ticks = [None] * n_plots
    
    if isinstance(cmap_num, int):
        cmap_num = [cmap_num] * n_plots
    
    # Criar figura e subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, 
                             subplot_kw={'projection': ccrs.PlateCarree()},
                             gridspec_kw={'hspace': hspace, 'wspace': wspace})
    
    # Garantir que axes seja sempre um array 2D
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plotar cada dataset
    plot_objects = []
    for i, (data, title, prefix, cmap, vlim, label, extend, notation, ticks, num_levels) in enumerate(
            zip(datasets, titles, title_prefixes, cmaps, vlims, cbar_labels, cbar_extend, 
                cbar_ticks_notation, cbar_ticks, cmap_num if cmap_num is not None else [None] * n_plots)):
        row, col = i // ncols, i % ncols
        ax = axes[row, col]
        
        # Definir parâmetros do colorbar
        cbar_kwargs = {
            'label': label, 
            'shrink': cbar_shrink, 
            'pad': cbar_pad, 
            'extend': extend
        }
        
        # Processar o colormap para criar versão discreta se necessário
        plot_cmap = cmap
        if num_levels is not None:
            base_cmap = plt.get_cmap(cmap)
            colors = [base_cmap(i) for i in np.linspace(0, 1, num_levels)]
            plot_cmap = mcolors.LinearSegmentedColormap.from_list(f'discrete_{cmap}', colors, N=num_levels)
        
        # Plotar dados
        im = data.plot(ax=ax, cmap=plot_cmap, vmin=vlim[0], vmax=vlim[1], 
                       add_colorbar=True, cbar_kwargs=cbar_kwargs)
        plot_objects.append(im)
        
        # Aplicar ticks personalizados se fornecidos
        if ticks is not None:
            cbar = im.colorbar
            cbar.set_ticks(ticks)
        
        # Aplicar notação científica aos ticks da colorbar se necessário
        if notation:
            cbar = im.colorbar
            valores_kmil = [v / 1e3 for v in cbar.get_ticks()]
            labels = [f"{v:.0f}" for v in valores_kmil]
            cbar.ax.set_yticklabels(labels)
            cbar.set_label(label)
        
        # Adicionar título
        ax.set_title(f"{prefix}) {title}", loc='left', y=1)
        
        # Adicionar características cartográficas
        if add_features:
            ax.coastlines(linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.set_extent([-180, 180, ylim[0], ylim[1]], crs=ccrs.PlateCarree())
            ax.set_xticks(range(-180, 181, 30), crs=ccrs.PlateCarree())
            ax.set_yticks(range(-90, 91, 30), crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
            ax.yaxis.set_major_formatter(LatitudeFormatter())
            ax.tick_params(labelsize=8)
            ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                         linewidth=1, color='gray', alpha=0.5, linestyle='--')
            ax.set_xlim([-180, 180])
            ax.set_ylim([ylim[0], ylim[1]])
            ax.set_ylabel('')
            ax.set_xlabel('')

    # Salvar figura se caminho for fornecido
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig, axes, plot_objects

def countSystemsArea(df, ax1=None, dpi=120):
    sns.set(style="whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    # check if duration cotains 0 in first value, increment +1 in duration column of df
    # if df['duration'].iloc[0] == 0:
    #     df['duration'] = df['duration'] + 60
    df_plot = df.copy()
    # Convert to hours
    # df_plot['duration'] = df_plot['duration'] / 60
    df_plot['mean_size'] = df_plot['mean_size'] / (0.1 * 0.1)  # Convertendo de pixels para km²
    df_plot['std_size'] = df_plot['std_size'] / (0.1 * 0.1)    # Convertendo de pixels para km²
    lowest_duration = df_plot[df_plot['duration'] <= 24]
    largest_duration = df_plot[df_plot['duration'] > 24]
    # print(lowest_duration)

    if ax1 is None:
        fig, ax1 = plt.subplots(figsize=(12, 5), dpi=dpi)
    else:
        fig = ax1.figure

    # Plot settings
    ax1.plot(
        lowest_duration.duration,
        lowest_duration['uid_count'].values,
        color='steelblue',
        marker='o',
        linestyle='--',
        linewidth=2,
        label='Tracked PSs',
        markerfacecolor='white',
        markersize=6,
        zorder=3
    )
    ax1.set_xticks(lowest_duration.duration, labels=lowest_duration.duration.astype(str), rotation=45, fontsize=12)
    ax1.set_yscale('log')
    ax1.set_ylabel('Count of Tracked PSs (log)', fontsize=12)
    # Set yticks font size
    ax1.tick_params(axis='y', labelsize=12)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.set_ylim(lowest_duration['uid_count'].min() / 2.1, lowest_duration['uid_count'].max() * 2.1)
    for j, count in enumerate(lowest_duration['uid_count'].values):
        percent = (count / lowest_duration['uid_count'].sum()) * 100 if lowest_duration['uid_count'].sum() > 0 else 0
        ax1.text(
            lowest_duration.duration.iloc[j] + 0.7,
            count * 1.05,
            f'{percent:.2f}%',
            ha='center', va='bottom',
            fontsize=10,
            color='steelblue',
            rotation=45,
        )
    ax2 = ax1.twinx()
    ax2.errorbar(
        lowest_duration.duration,
        lowest_duration['mean_size'],
        yerr=lowest_duration['std_size'],
        fmt='^',
        color='gray',
        linewidth=1.5,
        markersize=8,
        capsize=3,
        alpha=0.3,
        zorder=2
    )
    ax2.set_ylabel('Mean Size x10³ km²', fontsize=12, color='gray')
    ax2.yaxis.set_major_locator(plt.MaxNLocator(nbins=20, integer=True))
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.spines["right"].set_position(("axes", 1.05))
    ax2.tick_params(axis='y', colors='gray')
    # Get interval at column duration at df at first and second row
    interval = df_plot['duration'].iloc[1] - df_plot['duration'].iloc[0]
    if interval == 1:
        ax1.axvspan(-0.5, 2.25, color='lightcoral', alpha=0.2)
        ax1.axvspan(2.25, lowest_duration['duration'].max() + 0.5, color='lightgreen', alpha=0.2)
    else:
        ax1.axvspan(-0.5, 2.25, color='lightcoral', alpha=0.2)
        ax1.axvspan(2.25, lowest_duration['duration'].max() + 0.5, color='lightgreen', alpha=0.2)
    total_eventos = lowest_duration['uid_count'].sum()
    if interval == 1:
        total_short = df_plot[df_plot['duration'] < 3]['uid_count'].sum()
        total_long = df_plot[df_plot['duration'] >= 3]['uid_count'].sum()
    else:
        total_short = df_plot[df_plot['duration'] < 3]['uid_count'].sum()
        total_long = df_plot[df_plot['duration'] >= 3]['uid_count'].sum()
    perc_short = (total_short / total_eventos) * 100 if total_eventos > 0 else 0
    perc_long = (total_long / total_eventos) * 100 if total_eventos > 0 else 0
    bbox_props = dict(boxstyle='square', fc='white', ec='black', alpha=0.9)

    # Configure {} positions
    if interval == 1:
        weigh = 2.5
        weigh2 = 26
        position1 = (0.046, 1.04)
        position2 = (0.548, 1.04)
    else:
        weigh = 2.5
        weigh2 = 26
        position1 = (0.046, 1.04)
        position2 = (0.548, 1.04)

    total_short = f'{total_short:,}'.replace('.', ',')
    ax1.annotate(f'{perc_short:.1f}% < 3H - Total: {total_short}',
                 xy=position1, xycoords='axes fraction',
                 xytext=(0, 15), textcoords='offset points',
                 ha='center', va='bottom', fontsize=12,
                 bbox=bbox_props,
                 arrowprops=dict(arrowstyle=f'-[, widthB={weigh}, lengthB=1', lw=1.5, color='black'))
    total_long = f'{total_long:,}'.replace('.', ',')
    ax1.annotate(f'{perc_long:.1f}% ≥ 3H - Total: {total_long}',
                 xy=position2, xycoords='axes fraction',
                 xytext=(0, 15), textcoords='offset points',
                 ha='center', va='bottom', fontsize=12,
                 bbox=bbox_props,
                 arrowprops=dict(arrowstyle=f'-[, widthB={weigh2}, lengthB=1',
                                 lw=1.5, color='black'))
    count_largest = int(largest_duration['uid_count'].sum())
    perc_largest = (count_largest / total_eventos) * 100 if total_eventos > 0 else 0
    ax1.set_xlim(0, lowest_duration['duration'].max() + 0.5)
    ax1.set_xlabel('Duration (hours)', fontsize=12)
    count_largest = f'{count_largest:,}'.replace('.', ',')
    ax1.annotate(f'Number of PS with Duration > 24H = {count_largest} ({perc_largest:.2f}%)',
                 xy=(.5, 0.02), xycoords='axes fraction',
                 ha='center', va='bottom', fontsize=12,
                 bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='black', lw=1.5, alpha=0.9),
                 arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    # Caso possível os valores do xticks sejam ajustados para inteiro, por exemplo, se for 1.0 deve ser 1, mas se for 1.5 deve ser 1.5
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x) if x.is_integer() else x))
    # Diminuir o tamanho da fonte dos xticks
    ax1.tick_params(axis='x', labelsize=10.2)
    # Legendas
    ax1.legend(loc=[0.5, 0.9], fontsize=12)
    ax2.legend(['Mean Size'], loc=[0.75, 0.9], fontsize=12)
    plt.subplots_adjust(top=0.8)
    plt.tight_layout()
    ax2.set_ylim(-101000, 205000)

    # Ajuste os yticks do eixo ax2 para ter virgula
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}'))

    if ax1 is None:
        return fig


def map(array, title=None, figsize=(15, 15), ax=None, dpi=120,
        cbar_shrink=0.4, cbar_pad=0.03, cbar_aspect=30, cbar_orientation='vertical', cbar_fraction=0.03,
        cbar_extend='neither', cbar_ticks=None, cbar_format='%.0f', cbar_label=None, log=False,
        cmap='tab20b', cmap_num=None, cmap_norm=None,
        cbar=True, ylim=[-90, 90], xlim=[-180, 180], cbar_ticks_notation=False,
        min_val=None, max_val=None, region=[-180, 180, -90, 90], alpha=1, lat_density=False, 
        lat_mean=False, lat_mean_ticks=None, lat_xlabel=None, lat_ylabel=None,
        lat_density_diff=False, array1=None, array2=None, cbar_ticks_notation_resumed=False,
        sec_array=None, sec_title=None, save_path=None):

    gc.collect()
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
    else:
        fig = ax.figure

    # Map drawing
    ax.coastlines(resolution='50m', color='black', linewidth=0.5, alpha=1)
    ax.add_feature(cfeature.OCEAN, alpha=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    geobr.read_country().boundary.plot(ax=ax, linewidth=0.5, alpha=0.3, color='black')

    # Set ticks and labels
    ax.set_xticks(range(-180, 181, 30), crs=ccrs.PlateCarree())
    ax.set_yticks(range(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=9)

    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                 linewidth=1, color='gray', alpha=0.5, linestyle='--')

    # Set limits
    array = array.sel(lat=slice(region[2], region[3]), lon=slice(region[0], region[1]))

    # Fit min values is nan based on min_val
    if min_val is not None:
        array = array.where(array >= min_val, np.nan)

    if log:
        vmin = array.where(array > 0).min().values if min_val is None else min_val
        vmax = array.max().values if max_val is None else max_val
        if vmin <= 0:
            vmin = 1e-3
        cmap_norm = LogNorm(vmin=vmin, vmax=vmax)
        exponents = np.arange(np.floor(np.log10(vmin)), np.ceil(np.log10(vmax)) + 1)
        cbar_ticks = 10 ** exponents
        cbar_format = None
        max_val = vmax if max_val is None else max_val
        min_val = vmin if min_val is None else min_val

    if cmap_num is not None and log is False:
        base_cmap = plt.get_cmap(cmap)
        colors = [base_cmap(i) for i in np.linspace(0, 1, cmap_num)]
        cmap = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=cmap_num)
        cmap_norm = mcolors.Normalize(vmin=min_val, vmax=max_val)

    # Figure plotting with colorbar
    if cbar:
        fg = array.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(),
                cmap=cmap, vmax=max_val,
                norm=cmap_norm,
                alpha=alpha,
                add_colorbar=cbar,
                cbar_kwargs={
                    'shrink': cbar_shrink,
                    'pad': cbar_pad,
                    'aspect': cbar_aspect,
                    'orientation': cbar_orientation,
                    'fraction': cbar_fraction,
                    'extend': cbar_extend,
                    'ticks': cbar_ticks,
                    'format': cbar_format,
                    'label': cbar_label
                })
    else:
        fg = array.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(),
                cmap=cmap, vmax=max_val,
                norm=cmap_norm,
                alpha=alpha)
        fg.colorbar.remove()

    # aplica notação científica, se requisitado
    if cbar_ticks_notation and cbar_ticks:
        labels = [f"{tick:.0f}" for tick in cbar_ticks]
        fg.colorbar.set_ticklabels(labels)

    plt.title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    if cbar_label:
        fg.colorbar.ax.set_ylabel(cbar_label, fontsize=11)
        fg.colorbar.ax.tick_params(labelsize=10)
    
    if min_val is not None:
        cbar = fg.colorbar
        labels = cbar.get_ticks().tolist()
        cbar.set_ticks([min_val] + cbar.get_ticks()[1:].tolist())

    if cbar_ticks_notation_resumed:
        cbar = fg.colorbar
        valores_kmil = [v / 1e3 for v in cbar.get_ticks()]
        labels = [f"{v:.0f}" for v in valores_kmil]
        cbar.ax.set_yticklabels(labels, fontsize=8)
        # cbar.set_label(cbar_label, fontsize=10)
        if min_val is not None:
            cbar.set_ticks([min_val] + cbar.get_ticks()[1:].tolist())
            cbar.ax.set_yticklabels([f"{min_val/1e3:.0f}"] + labels[1:], fontsize=10)

    if sec_array is not None:
        with mpl.rc_context({'hatch.color': 'black','hatch.linewidth': 1.2}):
            cs = sec_array.plot.contourf(
                ax=ax,
                levels=[0, 0.05],
                colors=['none'],
                hatches=['///////'],
                add_colorbar=False,
                transform=ccrs.PlateCarree()
            )
    
        plt.text(0.02, 0.1, sec_title, fontsize=12, color='black',
                 ha='left', va='top', transform=ax.transAxes,
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.3'))

        # Remove ylabels and xlabels
        ax.set_ylabel('')
        ax.set_xlabel('')


    if lat_density:
        lat_density_data = array.sel(lat=slice(region[2], region[3])).sum(dim='lon')
        lat_coords = lat_density_data['lat'].values
        density_values = lat_density_data.values
        if lat_mean:
            density_values = density_values / len(array['lon'])
        ax_inset = inset_axes(ax,
                              width="8%",    
                              height="100%",  
                              loc='upper left',
                              bbox_to_anchor=(-0.1, 0, 1, 1),
                              bbox_transform=ax.transAxes,
                              borderpad=0)

        ax_inset.plot(density_values, lat_coords, color='black')
        ax_inset.set_ylim(ylim)
        if lat_ylabel:
            ax_inset.set_ylabel(lat_ylabel)
        
        ax_inset.grid(True, linestyle='--', alpha=0.3)
        # Preenchimento abaixo da curva
        ax_inset.fill_betweenx(lat_coords, 1, density_values, color='black', alpha=0.15)
        ax_inset.plot(density_values, lat_coords, color='black')
        # Adiciona no máximo 3 xticks
        if lat_mean_ticks is None:
            xticks = np.linspace(0, density_values.max(), num=4)
        else:
            xticks = lat_mean_ticks
        ax_inset.set_xticks(xticks)
        # add label axis x
        if lat_xlabel:
            ax_inset.set_xlabel(lat_xlabel, fontsize=8)
        def human_format(x, pos):
            if x >= 1e6:
                return f'{x*1e-6:.1f}M'
            elif x >= 1e3:
                return f'{x*1e-3:.0f}k'
            else:
                return str(int(x))
        ax_inset.xaxis.set_major_formatter(FuncFormatter(human_format))
        ax_inset.tick_params(axis='x', 
               labelsize=8, 
               rotation=270, 
               pad=10)
        plt.setp(ax_inset.get_xticklabels(), ha='right')
        # Forçar mesmos yticks da figura principal
        yticks = np.linspace(ylim[0], ylim[1], num=5)
        ax_inset.set_yticks(yticks)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.tick_params(axis='y', left=True)

    if lat_density_diff:
        # Calcula densidade por latitude das duas séries
        lat_density_1 = array1.sel(lat=slice(region[2], region[3])).sum(dim='lon')
        lat_density_2 = array2.sel(lat=slice(region[2], region[3])).sum(dim='lon')
        lat_density_o = array.sel(lat=slice(region[2], region[3])).sum(dim='lon')

        lat_coords1 = lat_density_1['lat'].values
        lat_coords2 = lat_density_2['lat'].values
        lat_coords_o = lat_density_o['lat'].values

        # Descobre as latitudes comuns (interseção)
        lat_coordsd = np.intersect1d(lat_coords1, lat_coords2)
        lat_coordsd = np.intersect1d(lat_coordsd, lat_coords_o)

        # Para alinhar as densidades aos latitudes comuns, indexamos com np.isin
        density_values_1 = lat_density_1.sel(lat=lat_coordsd).values
        density_values_2 = lat_density_2.sel(lat=lat_coordsd).values
        density_values_o = lat_density_o.sel(lat=lat_coordsd).values

        # Inserir eixo à esquerda
        ax_inset = inset_axes(ax,
                            width="8%",    
                            height="100%",  
                            loc='upper left',
                            bbox_to_anchor=(-0.1, 0, 1, 1),
                            bbox_transform=ax.transAxes,
                            borderpad=0)

        # Plot das duas densidades (usando os lat_coords originais)
        ax_inset.plot(density_values_1, lat_coordsd, color='black', label='GSMAP')
        ax_inset.plot(density_values_2, lat_coordsd, color='red', linestyle='--', label='IMERG')
        # Plot da diferença (também latitudes comuns)
        ax_inset.plot(density_values_o, lat_coordsd, color='green', linestyle=':', label='Diff')
        ax_inset.set_ylim(ylim)
        ax_inset.set_ylabel("Latitude")
        ax_inset.grid(True, linestyle='--', alpha=0.3)
        # Formatador do eixo X
        xticks = np.linspace(0, max(density_values_1.max(), density_values_2.max()), num=5)
        ax_inset.set_xticks(xticks)
        def human_format(x, pos):
            if x >= 1e6:
                return f'{x*1e-6:.1f}M'
            elif x >= 1e3:
                return f'{x*1e-3:.0f}k'
            else:
                return str(int(x))
        ax_inset.xaxis.set_major_formatter(FuncFormatter(human_format))
        ax_inset.tick_params(axis='x', 
               labelsize=10, 
               rotation=270, 
               pad=10)
        plt.setp(ax_inset.get_xticklabels(), ha='right')
        # Sincroniza yticks com mapa principal
        if lat_mean_ticks is None:
            yticks = np.linspace(ylim[0], ylim[1], num=5)
        else:
            yticks = lat_mean_ticks
        ax_inset.set_yticks(yticks)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.tick_params(axis='y', left=False)
        ax_inset.legend(loc='upper right', fontsize=8, frameon=False, bbox_to_anchor=(.05, .95), borderaxespad=0.)

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)

    if ax is None:
        return fig

def quiver(u, v, sparsity, title, cbar_label, cmap, cmap_num, cbar_extend, cbar_fraction, ylim,
            time_delta = 60, scale=1, figsize=(15, 10), alpha=0.5, grid_alpha=0.5, max_speed=50, 
            cbar_ticks=None, cbar_orientation='vertical', cbar_pad=0.02, cbar_aspect=20, ax=None, cbar=True,
            cbar_shrink=0.8, cbar_format='%.1f', vel_unit='km/h', norm_comp=False, fix_norm=None, save_path=None):
    
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
    else:
        fig = ax.figure
    
    # fig = plt.figure(figsize=figsize)
    # ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution='50m', color='black', linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, alpha=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.set_xticks(range(-180, 181, 30), crs=ccrs.PlateCarree())
    ax.set_yticks(range(-90, 91, 30), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
    
    # Add geobr
    geobr.read_country().boundary.plot(ax=ax, linewidth=0.5, alpha=0.3, color='black')
    
    # Downsample the data
    u = u[::sparsity, ::sparsity]
    v = v[::sparsity, ::sparsity]
    
    # Convert to desired velocity unit
    pixel_size_km = 111.32
    if vel_unit == 'km/h':
        factor = pixel_size_km * 60 / time_delta
    elif vel_unit == 'm/s':
        factor = pixel_size_km * 1000 / (time_delta * 60)
    else:
        factor = 1.0
    # Scale u and v
    u = u * factor
    v = v * factor

    # Calculate the magnitude
    mag = np.sqrt(u**2 + v**2)
    mag = np.where(mag > max_speed, max_speed, mag)
    cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(0, 1, cmap_num))
    cmap = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=cmap_num)
    norm = mcolors.Normalize(vmin=0, vmax=max_speed)
    colors = cmap(norm(mag.flatten()))

    # Normalize u and v
    if norm_comp:
        u = u / mag
        v = v / mag
        if fix_norm is not None:
            u = u * fix_norm
            v = v * fix_norm

    # Plot the quiver
    fi = ax.quiver(u.lon, u.lat, u, v, color=colors, cmap=cmap, edgecolor='black', linewidth=0.5, 
                    norm=norm, transform=ccrs.PlateCarree(), angles='xy', scale_units='xy', scale=scale)
    
    if cbar:
        # Add colorbar
        cbar = plt.colorbar(
            fi, ax=ax, orientation=cbar_orientation, fraction=cbar_fraction,
            pad=cbar_pad, aspect=cbar_aspect, shrink=cbar_shrink, extend=cbar_extend,
            label='Wind Speed (' + vel_unit + ')', ticks=cbar_ticks, format=cbar_format
        )
        pos = cbar.ax.get_position()
        new_y0 = min(pos.y0 + 0.01, 0.7)  # avoid going completely out of the figure
        cbar.ax.set_position([pos.x0, new_y0, pos.width, pos.height])
        cbar.set_label(cbar_label, fontsize=12)

    plt.title(title)
    plt.ylim(ylim)

    # Save the figure
    if save_path is not None:
        plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0.1)

    if ax is None:
        return fig

def add_seasonal_backgrounds(ax_list, start_date, end_date, y_max_values):
    """
    Add seasonal background patches to multiple axes
    
    Parameters:
    ax_list: list of matplotlib axes
    start_date: start date for the time series
    end_date: end date for the time series  
    y_max_values: list of maximum y values for each axis
    """
    # Define seasonal configurations
    season_colors = {'DJF': '#e6f3ff', 'MAM': '#e6ffe6', 'JJA': '#fff7e6', 'SON': '#ffe6f0'}
    season_map = {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM',
                  6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON'}
    
    # Create monthly date range
    full_date_range = pd.date_range(start_date, end_date, freq='MS')
    
    # Add seasonal background patches
    for i, ax in enumerate(ax_list):
        for j in range(len(full_date_range)):
            current_date = full_date_range[j]
            season = season_map[current_date.month]
            
            patch_start = current_date
            if j == len(full_date_range) - 1:
                patch_end = current_date + pd.DateOffset(months=1)
            else:
                patch_end = full_date_range[j + 1]
            
            ax.add_patch(mpatches.Rectangle((patch_start, 0), patch_end - patch_start, 
                                          y_max_values[i], facecolor=season_colors[season], 
                                          alpha=1, zorder=0))
    
    return season_colors



def animate_cumulative(gsmap_trk, imerg_trk, gsmap_name="GSMap", imerg_name="IMERG", color_max=10):

    gsmap_trk = gpd.GeoDataFrame(gsmap_trk, geometry=gpd.GeoSeries.from_wkt(gsmap_trk['geometry']), crs="EPSG:4326")
    imerg_trk = gpd.GeoDataFrame(imerg_trk, geometry=gpd.GeoSeries.from_wkt(imerg_trk['geometry']), crs="EPSG:4326")

    # Função segura para concatenar arrays e valores escalares
    def safe_concat(series):
        arrays = []
        for x in series:
            if x is None:
                continue
            if isinstance(x, (list, np.ndarray, pd.Series)):
                if len(x) > 0:
                    arrays.append(np.asarray(x))
            else:
                arrays.append(np.array([x]))
        if len(arrays) == 0:
            return np.array([], dtype=np.int32)
        return np.concatenate(arrays).astype(np.int32)
    # Contadores cumulativos
    cumulative_count_gsmap = np.zeros((1200, 3600), dtype=np.int32)
    cumulative_count_imerg = np.zeros((1800, 3600), dtype=np.int32)
    # Timestamps únicos
    timestamps = sorted(set(gsmap_trk.timestamp.unique()).union(imerg_trk.timestamp.unique()))
    # Criar figura e eixos
    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    ax_line = fig.add_subplot(gs[1, :])
    xmin, ymin, xmax, ymax = pd.concat([gsmap_trk, imerg_trk]).total_bounds
    for ax in [ax1, ax2]:
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
        ax.set_extent([xmin, xmax, ymin, ymax], crs=ccrs.PlateCarree())
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.2, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
    # Arrays de lat/lon
    lats_gsmap = np.arange(-60, 60, 0.1)
    longs_gsmap = np.arange(-180, 180, 0.1)
    lats_imerg = np.arange(-90, 90, 0.1)
    longs_imerg = np.arange(-180, 180, 0.1)
    # Linhas acumuladas
    df1_counts = gsmap_trk.groupby('timestamp')['uid'].nunique().reindex(timestamps, fill_value=0).cumsum()
    df2_counts = imerg_trk.groupby('timestamp')['uid'].nunique().reindex(timestamps, fill_value=0).cumsum()
    # Cores
    norm = Normalize(vmin=1, vmax=color_max)
    cmap_gsmap = plt.cm.Blues
    cmap_imerg = plt.cm.Reds
    # Calcular vmax final para colorbar
    temp_gsmap = np.zeros_like(cumulative_count_gsmap)
    temp_imerg = np.zeros_like(cumulative_count_imerg)
    for _, cluster in gsmap_trk.groupby('uid'):
        arr_y = safe_concat(cluster['array_y'].explode())
        arr_x = safe_concat(cluster['array_x'].explode())
        if arr_y.size > 0:
            unique_coords = np.unique(np.column_stack((arr_y, arr_x)), axis=0)
            np.add.at(temp_gsmap, (unique_coords[:,0], unique_coords[:,1]), 1)
    for _, cluster in imerg_trk.groupby('uid'):
        arr_y = safe_concat(cluster['array_y'].explode())
        arr_x = safe_concat(cluster['array_x'].explode())
        if arr_y.size > 0:
            unique_coords = np.unique(np.column_stack((arr_y, arr_x)), axis=0)
            np.add.at(temp_imerg, (unique_coords[:,0], unique_coords[:,1]), 1)
    # Inicializar imagens
    img1 = ax1.imshow(np.zeros((len(lats_gsmap), len(longs_gsmap))),
                      extent=[xmin, xmax, ymin, ymax],
                      origin='lower',  cmap=cmap_gsmap, norm=norm)
    img2 = ax2.imshow(np.zeros((len(lats_imerg), len(longs_imerg))),
                      extent=[xmin, xmax, ymin, ymax],
                      origin='lower', cmap=cmap_imerg, norm=norm)
    # Colorbars fixas
    fig.colorbar(img1, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04, label=f"{gsmap_name} cumulative count")
    fig.colorbar(img2, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04, label=f"{imerg_name} cumulative count")
    line1, = ax_line.plot([], [], color='blue', label=f'{gsmap_name} Cumulative Clusters')
    line2, = ax_line.plot([], [], color='red', label=f'{imerg_name} Cumulative Clusters')
    ax_line.set_xlim(timestamps[0], timestamps[-1])
    ax_line.set_ylim(0, max(df1_counts.max(), df2_counts.max())*1.1)
    ax_line.set_ylabel('Cumulative Count')
    ax_line.legend()
    ax_line.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax_line.grid(True, linestyle='--', alpha=0.7)
    ax_line.set_title("c) Cumulative Cluster Count Over Time", loc='left', y=1)

    def update(frame):
        timestamp = timestamps[frame]
        df1_ = gsmap_trk[gsmap_trk.timestamp == timestamp]
        df2_ = imerg_trk[imerg_trk.timestamp == timestamp]
        # Acumular clusters GSMap
        for _, cluster in df1_.groupby('uid'):
            arr_y = safe_concat(cluster['array_y'].explode())
            arr_x = safe_concat(cluster['array_x'].explode())
            if arr_y.size > 0:
                unique_coords = np.unique(np.column_stack((arr_y, arr_x)), axis=0)
                np.add.at(cumulative_count_gsmap, (unique_coords[:,0], unique_coords[:,1]), 1)
        # Acumular clusters IMERG
        for _, cluster in df2_.groupby('uid'):
            arr_y = safe_concat(cluster['array_y'].explode())
            arr_x = safe_concat(cluster['array_x'].explode())
            if arr_y.size > 0:
                unique_coords = np.unique(np.column_stack((arr_y, arr_x)), axis=0)
                np.add.at(cumulative_count_imerg, (unique_coords[:,0], unique_coords[:,1]), 1)
        # DataArray com transparência
        cum_gsmap_xr = xr.DataArray(cumulative_count_gsmap, dims=["lat","lon"], coords=[lats_gsmap,longs_gsmap])
        cum_imerg_xr = xr.DataArray(cumulative_count_imerg, dims=["lat","lon"], coords=[lats_imerg,longs_imerg])
        cum_gsmap_xr = cum_gsmap_xr.sel(lat=slice(ymin,ymax), lon=slice(xmin,xmax)).where(cum_gsmap_xr>0)
        cum_imerg_xr = cum_imerg_xr.sel(lat=slice(ymin,ymax), lon=slice(xmin,xmax)).where(cum_imerg_xr>0)
        img1.set_data(cum_gsmap_xr.values)
        img2.set_data(cum_imerg_xr.values)
        ax1.set_title(f"a) {gsmap_name} Cumulative PS up to {timestamp}", loc='left', y=1, x=-0.09)
        ax2.set_title(f"b) {imerg_name} Cumulative PS up to {timestamp}", loc='left', y=1, x=-0.09)
        # Remover contornos antigos
        [c.remove() for c in ax1.collections[2:]]
        [c.remove() for c in ax2.collections[2:]]
        # Contornos atuais
        if not df1_.empty:
            df1_.geometry.boundary.plot(ax=ax1, color='blue', linewidth=1)
        if not df2_.empty:
            df2_.geometry.boundary.plot(ax=ax2, color='red', linewidth=1)
        # Atualizar gráfico de linha
        line1.set_data(timestamps[:frame+1], df1_counts.values[:frame+1])
        line2.set_data(timestamps[:frame+1], df2_counts.values[:frame+1])
        return img1, img2, line1, line2

    ani = FuncAnimation(fig, update, frames=len(timestamps), blit=False, repeat=False)
    plt.close(fig)  # Isso evita que a figura estática apareça
    return HTML(ani.to_jshtml())


def trajectory_map(
    summary_df,
    *,
    extent=(-180,180,-60,60),
    step=10.0,
    annotate_empty=True,
    mode="full",       # "full" (setas e símbolos) | "balance" (somente cores)
    show_counts=False, # se True, exibe S e E dentro dos quadrantes no modo balance
    # preenchimento dos quadrantes
    quad_face_alpha=0.22,
    quad_edgecolor=None,
    show_quad_id=False,
    balance_top_frac=0.85,
    balance_mid_offset_frac=0.28,
    # contorno dos quadrantes
    quad_edge_lw=0.0,
    # textos (gerais)
    q_fontsize=9.0,
    d_fontsize=9.0,
    q_top_pad_frac=0.15,
    d_bot_pad_frac=0.15,
    text_color="#111",
    # textos no modo balance
    balance_fontsize=8.5,
    balance_color="#000",
    # seta (modo full)
    draw_arrows=True,
    arrow_len_frac=0.35,
    arrow_margin_frac=0.06,
    arrow_lw=0.9,
    arrow_color="#222",
    arrow_alpha=0.95,
    arrow_head_w=3.5,
    arrow_head_l=4.5,
    # ajuste: recuar origem apenas para “vertical para cima”
    up_start_offset_frac=0.35,   # fração da semialtura do quadrante
    angle_tol_deg=10.0,          # tolerância angular em torno de 90° (vertical)
    draw_self_dest_dot=False,    # se True, marca um ponto quando não há destino válido (<0)
    # continentes
    coastline_lw=1.2,
    land_edgecolor="#333",
    land_edge_lw=0.9,
    figsize=(16,9),
    # ---- NOVOS PARÂMETROS (Concordance) ----
    draw_concordance_marker=True,          # desenha marcador ● colorido (apenas no modo "full")
    conc_right_of_q=True,                  # True: posição à direita do rótulo ⊙; False: à esquerda
    conc_position="centroid",      # "top" | "bottom" | "centroid"
    conc_use_df_center=False,      # True: usa colunas lon_center/lat_center se existirem
    conc_xy_offset=(0.0, 0.0),     # deslocamentos absolutos em graus (lon, lat)
    conc_fontsize=6.0,                    # tamanho do marcador ●
    conc_green="#2ca02c",                  # cor para concordância (1)
    conc_red="#d62728",                    # cor para discordância (-1)
    ax=None,
):
    def _wrap_dlon(dlon):
        return ((dlon + 180.0) % 360.0) - 180.0

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
    else:
        fig = ax.figure
    proj = ccrs.PlateCarree()
    ax.set_extent(extent, crs=proj)

    # Base cartográfica
    land = cfeature.LAND.with_scale("110m")
    ocean = cfeature.OCEAN.with_scale("110m")
    ax.add_feature(ocean, facecolor="#e9f1fa", edgecolor="none", zorder=0)
    ax.add_feature(land,  facecolor="#f6f6f6", edgecolor=land_edgecolor, linewidth=land_edge_lw, zorder=0.5)
    ax.coastlines("110m", linewidth=coastline_lw, color="#111", zorder=1.0)
    ax.set_xticks(range(-180, 181, 30), crs=ccrs.PlateCarree())
    ax.set_yticks(range(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlim(extent[0], extent[1])
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                 linewidth=1, color='gray', alpha=0, linestyle='--')
    ax.tick_params(labelsize=9)

    # linhas da grade
    lon_min, lon_max, lat_min, lat_max = extent
    for x in np.arange(lon_min, lon_max+1e-9, step):
        ax.plot([x, x], [lat_min, lat_max], transform=proj, color="#888", lw=0.45, alpha=0.55, zorder=1.1)
    for y in np.arange(lat_min, lat_max+1e-9, step):
        ax.plot([lon_min, lon_max], [y, y], transform=proj, color="#888", lw=0.45, alpha=0.55, zorder=1.1)

    # cores e preenchimento
    BLUE = "#1f77b4"; PINK = "#e75480"; TIE = "#d9d9d9"
    for _, r in summary_df.iterrows():
        s = int(r["Qtdy_Start"]); e = int(r["Qtdy_End"])
        if not annotate_empty and (s == 0 and e == 0):
            continue
        face = (BLUE if s > e else (PINK if e > s else TIE))
        ax.add_patch(Rectangle((float(r["lon0"]), float(r["lat0"])),
                               width=float(r["lon1"]-r["lon0"]),
                               height=float(r["lat1"]-r["lat0"]),
                               transform=proj, facecolor=face,
                               edgecolor=quad_edgecolor, linewidth=quad_edge_lw,
                               alpha=quad_face_alpha, zorder=2.0))

    if mode == "balance":
        for _, r in summary_df.iterrows():
            s = int(r["Qtdy_Start"]); e = int(r["Qtdy_End"])
            if not annotate_empty and (s == 0 and e == 0):
                continue

            # centro geométrico da célula (sem depender de lon_center/lat_center do DF)
            lon0, lon1 = float(r["lon0"]), float(r["lon1"])
            lat0, lat1 = float(r["lat0"]), float(r["lat1"])
            cx = 0.5 * (lon0 + lon1)
            cy = 0.5 * (lat0 + lat1)
            h  = (lat1 - lat0)

            # (opcional) marcador ⊙ quad_id no topo (com fração exata da altura)
            y_top = lat0 + balance_top_frac * h
            y_mid = cy + balance_mid_offset_frac * h
            if show_quad_id:
                ax.text(cx, y_mid, f"⊙{int(r['quad_id'])}",
                        ha="center", va="center",
                        fontsize=q_fontsize, color=text_color,
                        transform=proj, zorder=3.1,
                        path_effects=[pe.withStroke(linewidth=1.4, foreground="white")])

            # (opcional) contagens S/E centradas nas metades
            if show_counts:
                ax.text(cx, y_top, f"S:{s}", ha="center", va="center",
                        fontsize=balance_fontsize, color=balance_color,
                        transform=proj, zorder=3.0,
                        path_effects=[pe.withStroke(linewidth=1.2, foreground="white")])
                ax.text(cx, cy - balance_mid_offset_frac * h, f"E:{e}", ha="center", va="center",
                        fontsize=balance_fontsize, color=balance_color,
                        transform=proj, zorder=3.0,
                        path_effects=[pe.withStroke(linewidth=1.2, foreground="white")])
        if ax is None:
            return fig

    else:

        # modo full — setas
        if draw_arrows:
            for _, r in summary_df.iterrows():
                qs = int(r["quad_id"])
                qd = int(r["Common_Dest"])

                cx, cy = float(r["lon_center"]), float(r["lat_center"])
                lon0, lon1 = float(r["lon0"]), float(r["lon1"])
                lat0, lat1 = float(r["lat0"]), float(r["lat1"])
                w = (lon1 - lon0)
                h = (lat1 - lat0)

                if qd < 0:
                    if draw_self_dest_dot:
                        ax.scatter([cx],[cy], s=10, transform=proj, color=arrow_color, alpha=arrow_alpha, zorder=2.4)
                    continue

                # não desenha quando origem e destino são o mesmo quadrante
                if qd == qs:
                    continue

                # centro do destino
                dest = summary_df.loc[summary_df["quad_id"] == qd]
                if dest.empty:
                    if draw_self_dest_dot:
                        ax.scatter([cx],[cy], s=10, transform=proj, color=arrow_color, alpha=arrow_alpha, zorder=2.4)
                    continue
                dx, dy = float(dest["lon_center"].values[0]), float(dest["lat_center"].values[0])

                # direção robusta (antimeridiano) + métrica local
                dlon = _wrap_dlon(dx - cx); dx_adj = cx + dlon
                coslat = np.cos(np.deg2rad(cy))
                vx_adj = (dx_adj - cx) * max(coslat, 1e-6)
                vy_adj = (dy - cy)
                norm = np.hypot(vx_adj, vy_adj)
                if norm == 0:
                    continue

                # vetor unitário em lon/lat
                theta = np.arctan2(vy_adj, vx_adj)
                ux = np.cos(theta) / max(coslat, 1e-6)
                uy = np.sin(theta)

                # limites úteis
                mx = arrow_margin_frac * (lon1 - lon0)
                my = arrow_margin_frac * (lat1 - lat0)
                xmin, xmax = lon0 + mx, lon1 - mx
                ymin, ymax = lat0 + my, lat1 - my
                half_w = 0.5 * (xmax - xmin)
                half_h = 0.5 * (ymax - ymin)

                # comprimento alvo
                L_fixed = arrow_len_frac * min(half_w, half_h)

                # ===== AJUSTE: só quando a direção é realmente vertical-para-cima =====
                # ângulo (graus) medido a partir do eixo +x; vertical seria ~90°
                ang_deg = np.degrees(theta) % 360.0
                # normaliza para 0..180 (simetria), e compara distância a 90°
                if ang_deg > 180.0: 
                    ang_deg = 360.0 - ang_deg
                is_strictly_up = (uy > 0) and (abs(ang_deg - 90.0) <= angle_tol_deg)

                sx, sy = cx, cy
                if is_strictly_up:
                    sy = np.clip(cy - up_start_offset_frac * half_h, ymin, ymax)

                # clipping
                t_vals = []
                if ux > 0: t_vals.append((xmax - sx)/ux)
                if ux < 0: t_vals.append((xmin - sx)/ux)
                if uy > 0: t_vals.append((ymax - sy)/uy)
                if uy < 0: t_vals.append((ymin - sy)/uy)
                t_bound = min([t for t in t_vals if np.isfinite(t) and t > 0], default=L_fixed)
                L = max(0.0, min(L_fixed, t_bound))

                ex, ey = sx + ux * L, sy + uy * L

                p = Path([(sx, sy), (ex, ey)], [Path.MOVETO, Path.LINETO])
                patch = FancyArrowPatch(
                    path=p, transform=proj,
                    arrowstyle=f"-|>,head_length={arrow_head_l},head_width={arrow_head_w}",
                    mutation_scale=1.0, lw=arrow_lw,
                    color=arrow_color, alpha=arrow_alpha, zorder=2.3,
                )
                ax.add_patch(patch)

        # rótulos Q (topo) e D (base)
        for _, r in summary_df.iterrows():
            s = int(r["Qtdy_Start"]); e = int(r["Qtdy_End"])
            if not annotate_empty and (s == 0 and e == 0):
                continue
            cx, cy = float(r["lon_center"]), float(r["lat_center"])
            h = float(r["lat1"]) - float(r["lat0"])
            q_y = float(r["lat1"]) - q_top_pad_frac * h
            d_y = float(r["lat0"]) + d_bot_pad_frac * h
            ax.text(cx, q_y, f"⊙{int(r['quad_id'])}", ha="center", va="center",
                    fontsize=q_fontsize, color=text_color, transform=proj, zorder=3.1,
                    path_effects=[pe.withStroke(linewidth=1.4, foreground="white")])
            d_val = '-' if int(r["Common_Dest"]) < 0 else int(r["Common_Dest"])
            ax.text(cx, d_y, f"⊗{d_val}", ha="center", va="center",
                    fontsize=d_fontsize, color=text_color, transform=proj, zorder=3.1,
                    path_effects=[pe.withStroke(linewidth=1.4, foreground="white")])
            
            # ---- MARCADOR DE CONCORDÂNCIA (● verde/vermelho) ----
            if draw_concordance_marker:
                # tenta ler como int; qualquer falha ou NaN => trata como 0 (neutro)
                conc_raw = r.get("Concordance", 0)
                try:
                    conc_val = int(conc_raw)
                except Exception:
                    conc_val = 0

                if conc_val == 1:
                    mcolor = conc_green
                elif conc_val == -1:
                    mcolor = conc_red
                else:
                    mcolor = None  # neutro => não desenha

                if mcolor is not None:
                    # largura/altura do quadrante
                    lon0, lon1 = float(r["lon0"]), float(r["lon1"])
                    lat0, lat1 = float(r["lat0"]), float(r["lat1"])
                    w = (lon1 - lon0); h = (lat1 - lat0)

                    # pontos de referência
                    cx_geom = 0.5 * (lon0 + lon1)
                    cy_geom = 0.5 * (lat0 + lat1)
                    cx_df   = float(r.get("lon_center", cx_geom))
                    cy_df   = float(r.get("lat_center", cy_geom))

                    # escolha da posição
                    if conc_position == "centroid":
                        mx, my = (cx_df, cy_df) if conc_use_df_center else (cx_geom, cy_geom)
                    elif conc_position == "top":
                        mx, my = cx_geom, (lat1 - q_top_pad_frac * h)
                    elif conc_position == "bottom":
                        mx, my = cx_geom, (lat0 + d_bot_pad_frac * h)
                    else:
                        mx, my = cx_geom, cy_geom  # fallback

                    # pequeno ajuste opcional (em graus)
                    mx += conc_xy_offset[0]
                    my += conc_xy_offset[1]

                    ax.text(mx, my, "●", ha="center", va="center",
                            fontsize=conc_fontsize, color=mcolor,
                            transform=proj, zorder=3.2,
                            path_effects=[pe.withStroke(linewidth=1.2, foreground="white")])

        if ax is None:
            return fig


def plot_ps_initiation_dissipation_dominance(
    gdf,
    extent=None,
    base_map=None,
    base_edgecolor='black',
    base_linewidth=1.0,
    base_alpha=0.8,
    title='GSMaP',
    title_sequency=('a)', 'b)', 'c)'),
    q1_km=None,
    counts_log=True,
    vmin=None,
    vmax=None,
    # ---- parâmetros para Dominance ----
    dominance_mode='logratio',      # 'logratio' (padrão) ou 'asymmetry'
    dominance_log_base=10.0,        # base do log quando mode='logratio'
    alpha_smoothing=1,            # suavização bayesiana (Jeffreys=0.5; Laplace=1.0)
    dominance_min_total=0,          # mascarar células com (Hs+He) < esse valor (0 = desativado)
):

    # -----------------------------
    # 1) Dados base e extensão
    # -----------------------------
    xs = gdf['start_geometry'].x.values
    ys = gdf['start_geometry'].y.values
    xe = gdf['end_geometry'].x.values
    ye = gdf['end_geometry'].y.values

    if extent is None:
        pad = 1.0
        xmin = float(np.nanmin([xs.min(), xe.min()])) - pad
        xmax = float(np.nanmax([xs.max(), xe.max()])) + pad
        ymin = float(np.nanmin([ys.min(), ye.min()])) - pad
        ymax = float(np.nanmax([ys.max(), ye.max()])) + pad
        extent = [xmin, xmax, ymin, ymax]
    else:
        xmin, xmax, ymin, ymax = extent

    # -----------------------------
    # 2) Grade climatológica pelo Q1
    # -----------------------------
    if q1_km is None:
        q1_km = float(np.nanpercentile(gdf['start_end_haversine_km'].values, 25))
    deg_per_km = 1.0 / 111.32
    dlat = max(q1_km * deg_per_km, 0.1)
    phi0 = (ymin + ymax) / 2.0
    dlon = dlat / max(np.cos(np.deg2rad(phi0)), 1e-6)

    xbins = np.arange(extent[0], extent[1] + dlon, dlon)
    ybins = np.arange(extent[2], extent[3] + dlat, dlat)

    # Histogramas 2D (atenção: (y, x))
    H_start, _, _ = np.histogram2d(ys, xs, bins=[ybins, xbins])
    H_end,   _, _ = np.histogram2d(ye, xe, bins=[ybins, xbins])

    # -----------------------------
    # 3) Contagens: escala e colormaps
    # -----------------------------
    if vmax is not None:
        vmax_counts = float(vmax)
    else:
        vmax_counts = float(np.nanmax(np.stack([H_start, H_end])))

    if counts_log:
        norm_counts = LogNorm(vmin=1, vmax=max(vmax_counts, 1))
        ticks_counts = [1, 3, 10, 30, 100, 300, 1000]
        Hs_plot = H_start
        He_plot = H_end
        cmap_start = plt.get_cmap('YlGnBu')
        cmap_end   = plt.get_cmap('YlOrRd')
    else:
        vmin_lin = 1
        vmax_lin = vmax_counts
        norm_counts = Normalize(vmin=vmin_lin, vmax=max(vmax_lin, 1))
        Hs_plot = H_start.astype(float).copy()
        He_plot = H_end.astype(float).copy()
        Hs_plot[Hs_plot == 0] = np.nan
        He_plot[He_plot == 0] = np.nan
        cmap_start = plt.get_cmap('YlGnBu').copy(); cmap_start.set_bad((1,1,1,0))
        cmap_end   = plt.get_cmap('YlOrRd').copy();  cmap_end.set_bad((1,1,1,0))
        locator = MaxNLocator(nbins=6, integer=True, min_n_ticks=3)
        ticks_counts = locator.tick_values(vmin_lin, max(vmax_lin, 1)).tolist()
        ticks_counts = sorted(set(int(t) for t in ticks_counts if t >= 1))
        if 1 not in ticks_counts:
            ticks_counts = [1] + ticks_counts

    # -----------------------------
    # 4) Dominance (revisado)
    # -----------------------------
    Hs = H_start.astype(float)
    He = H_end.astype(float)
    T  = Hs + He

    # máscara por baixa amostra total (opcional)
    mask_low = (T < dominance_min_total) if dominance_min_total > 0 else np.zeros_like(T, dtype=bool)

    if dominance_mode.lower() == 'asymmetry':
        # D = (Hs - He)/(Hs + He + alpha)
        D = (Hs - He) / (T + alpha_smoothing)
        D[mask_low] = np.nan
        # normalização simétrica [-1,1] "bonita"
        V = 1.0  # limite teórico
        norm_llr = TwoSlopeNorm(vmin=-V, vcenter=0.0, vmax=V)
        ticks_llr = np.linspace(-1, 1, 5)
        cmap_llr = 'coolwarm_r'
        cb_label = 'Dominance — (Init. - Diss.) / (Init. + Diss.)'

    else:  # 'logratio'
        # D = log_base( (Hs + alpha) / (He + alpha) )
        base = float(dominance_log_base)
        alpha = float(alpha_smoothing)
        # cálculo robusto
        ratio = (Hs + alpha) / (He + alpha)
        D = np.log(ratio) / np.log(base)
        # limpa células com T=0 (sem informação)
        D[T == 0] = np.nan
        D[mask_low] = np.nan

        # amplitude robusta por quantil e arredondamento "nice"
        finite = np.isfinite(D)
        q = float(np.nanquantile(np.abs(D[finite]), 0.98)) if np.any(finite) else 1.0

        def round_up_nice(x):
            if x <= 0: return 1.0
            exp = np.floor(np.log10(x))
            frac = x / (10**exp)
            for n in [1, 2, 2.5, 5]:
                if frac <= n:
                    return n * (10**exp)
            return 10**(exp + 1)

        V = round_up_nice(max(q, 0.1))
        norm_llr = TwoSlopeNorm(vmin=-V, vcenter=0.0, vmax=V)
        ticks_llr = np.linspace(-V, V, 5)
        cmap_llr   = 'coolwarm_r'
        cb_label = f'Dominance — log{int(base)}( (Init.+α)/(Diss.+α) )'

    # -----------------------------
    # 5) Layout
    # -----------------------------
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(15, 6.8))
    gs  = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[20, 1.2], wspace=0.08, hspace=0.18)

    ax1  = fig.add_subplot(gs[0, 0], projection=proj)
    ax2  = fig.add_subplot(gs[0, 1], projection=proj)
    ax3  = fig.add_subplot(gs[0, 2], projection=proj)
    cax1 = fig.add_subplot(gs[1, 0])
    cax2 = fig.add_subplot(gs[1, 1])
    cax3 = fig.add_subplot(gs[1, 2])

    def base_and_grid(ax, show_left=False, show_right=False, show_top=False):
        ax.set_extent(extent, crs=proj)
        ax.add_feature(cfeature.LAND, facecolor='0.95', zorder=0, alpha=.5)
        if base_map is not None and len(base_map) > 0:
            try:
                base_map.to_crs(epsg=4326).boundary.plot(
                    ax=ax, edgecolor=base_edgecolor, linewidth=base_linewidth,
                    alpha=base_alpha, zorder=1
                )
            except Exception:
                pass
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                          linewidth=0.5, color='gray', alpha=0.25, linestyle='--')
        if hasattr(gl, 'top_labels'):   gl.top_labels = show_top
        if hasattr(gl, 'xlabels_top'):  gl.xlabels_top = show_top
        if hasattr(gl, 'left_labels'):  gl.left_labels = show_left
        if hasattr(gl, 'ylabels_left'): gl.ylabels_left = show_left
        if hasattr(gl, 'right_labels'):  gl.right_labels = show_right
        if hasattr(gl, 'ylabels_right'): gl.ylabels_right = show_right
        return gl

    base_and_grid(ax1, show_left=True,  show_right=False, show_top=False)
    base_and_grid(ax2, show_left=False, show_right=False, show_top=False)
    base_and_grid(ax3, show_left=False, show_right=True,  show_top=False)

    # -----------------------------
    # 6) Mapas
    # -----------------------------
    m1 = ax1.pcolormesh(xbins, ybins, Hs_plot, cmap=cmap_start, norm=norm_counts, shading='auto')
    m2 = ax2.pcolormesh(xbins, ybins, He_plot, cmap=cmap_end,   norm=norm_counts, shading='auto')
    m3 = ax3.pcolormesh(xbins, ybins, D,       cmap=cmap_llr,   norm=norm_llr,   shading='auto')
    m3.set_alpha(1.0)

    for m in (m1, m2, m3):
        m.set_zorder(2)

    def draw_overlays(ax):
        ax.add_feature(cfeature.COASTLINE.with_scale('110m'),
                       linewidth=0.6, edgecolor='0.3', zorder=10)
        ax.add_feature(cfeature.BORDERS.with_scale('110m'),
                       linewidth=0.5, edgecolor='0.4', zorder=10)
        if base_map is not None and len(base_map) > 0:
            try:
                base_map.to_crs(epsg=4326).boundary.plot(
                    ax=ax, edgecolor=base_edgecolor, linewidth=base_linewidth,
                    alpha=base_alpha, zorder=11
                )
            except Exception:
                pass
    for ax in (ax1, ax2, ax3):
        draw_overlays(ax)

    # Títulos
    ax1.set_title(f"{title_sequency[0]} {title} - Initiation",  loc='left', y=1)
    ax2.set_title(f"{title_sequency[1]} {title} - Dissipation", loc='left', y=1)
    ax3.set_title(f"{title_sequency[2]} {title} - Dominance",           loc='left', y=1)

    # -----------------------------
    # 7) Colorbars
    # -----------------------------
    cb1 = fig.colorbar(m1, cax=cax1, orientation='horizontal', ticks=ticks_counts, extend='max')
    cb1.set_label(f'Initiation — PS counts per cell')
    cb2 = fig.colorbar(m2, cax=cax2, orientation='horizontal', ticks=ticks_counts, extend='max')
    cb2.set_label(f'Dissipation — PS counts per cell')
    cb3 = fig.colorbar(m3, cax=cax3, orientation='horizontal', ticks=ticks_llr, extend='both')
    cb3.set_label(cb_label)

    # Aproximar barras
    for cax in (cax1, cax2, cax3):
        pos = cax.get_position()
        cax.set_position([pos.x0, pos.y0 + 0.15, pos.width, pos.height * 0.75])

    # Nota da grade (km)
    cell_km = dlat * 111.32
    for ax in (ax1, ax2, ax3):
        ax.text(0.01, 0.02, f'Q1 Grid ≈ {cell_km:.1f} km',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(fc='none', ec='none', alpha=0.6))

    fig.suptitle('', y=0.98, fontsize=12)
    plt.show()
    return fig



def build_monthly_distance(df,
                           time_col="end_timestamp",
                           dist_col="start_end_haversine_km",
                           filter_max_km=None):
    # cópias e conversões
    d = df[[time_col, dist_col]].copy()
    d[time_col] = pd.to_datetime(d[time_col], utc=True, errors="coerce")
    d = d.dropna(subset=[time_col, dist_col])
    d = d.sort_values(time_col)

    # filtro opcional de valores extremos (ex.: > 2000 km)
    if filter_max_km is not None:
        d = d.loc[d[dist_col] <= float(filter_max_km)]

    # índice temporal no início do mês (MS) — robusto para resampling
    d = d.set_index(d[time_col].dt.to_period("M").dt.to_timestamp())

    # agregações mensais
    def q25(x): return np.nanpercentile(x, 25)
    def q75(x): return np.nanpercentile(x, 75)

    monthly = d.resample("MS")[dist_col].agg(["mean", "median", q25, q75, "max", "std" , "count"]).rename(
        columns={"q25": "p25", "q75": "p75", "max": "max", "std":"std" , "count": "n", "p90": "p99"}
    )

    # IC 95% (normal approx.) da média mensal
    # atenção: usa desvio-padrão amostral mensal; se n pequeno, prefira bootstrap
    stdm = d.resample("MS")[dist_col].std()
    monthly["ci95"] = 1.96 * (stdm / np.sqrt(monthly["n"].replace(0, np.nan)))
    monthly["p99"] = d.resample("MS")[dist_col].apply(lambda x: np.nanpercentile(x, 99))

    return monthly

def add_climatology_and_anomalies(monthly):
    m = monthly.copy()
    m["month"] = m.index.month
    clim = m.groupby("month")["mean"].mean()
    m["clim"] = m["month"].map(clim)
    m["anom"] = m["mean"] - m["clim"]
    # média móvel de 12 meses para suavizar ruído
    m["mean_roll12"] = m["mean"].rolling(12, min_periods=6).mean()
    m["anom_roll12"] = m["anom"].rolling(12, min_periods=6).mean()
    return m

from scipy.stats import theilslopes
import numpy as np

try:
    import pymannkendall as mk
    _HAS_MK = True
except Exception:
    _HAS_MK = False

def estimate_trend(ts):
    """
    ts: pandas Series com DatetimeIndex em frequência mensal (MS).
    Retorna slope_per_decade (km/década), intercept, pvalue (MK se disponível).
    """
    s = ts.dropna()
    if s.empty or len(s) < 8:
        return np.nan, np.nan, np.nan

    # Construir eixo temporal em UNIDADES DE MÊS (inteiros: 0,1,2,...)
    # Isto evita timedeltas com 'M' e é apropriado para séries mensais regulares.
    first = s.index[0]
    x = (s.index.year - first.year) * 12 + (s.index.month - first.month)
    x = x.astype(float).values
    y = s.values.astype(float)

    # Theil–Sen robusto
    slope, intercept, _, _ = theilslopes(y, x)  # km por mês
    slope_per_decade = slope * 120.0            # 120 meses ≈ 10 anos

    # Mann–Kendall (não-paramétrico) apenas para p-valor (opcional)
    pval = np.nan
    if _HAS_MK:
        res = mk.original_test(s, alpha=0.05)
        pval = getattr(res, "p", np.nan)

    return slope_per_decade, intercept, pval


def plot_monthly_displacement_evolution(
    monthly,
    title_prefix="GSMaP – Amazon",
    prefix_letter=("a)", "b) ")
):
    m = add_climatology_and_anomalies(monthly)
    slope_dec, _, pval = estimate_trend(m["anom"])

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1.0], hspace=0.25)

    # Painel A — níveis absolutos
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.fill_between(m.index, m["p25"], m["p75"], alpha=0.35, label="IQR (P25–P75)")
    ax1.plot(m.index, m["mean"], lw=1.5, label="Monthly mean (km)")
    ci_lo = m["mean"] - m["ci95"]; ci_hi = m["mean"] + m["ci95"]
    ax1.fill_between(m.index, ci_lo, ci_hi, alpha=0.75, label="95% CI of mean")
    ax1.set_ylabel("Start–End distance (km)")
    ax1.set_title(f"{prefix_letter[0]} {title_prefix}: Monthly displacement (absolute)", loc="left")
    ax1.grid(True, alpha=0.25)

    ax1b = ax1.twinx()
    ax1b.bar(m.index, m["n"], width=25, alpha=0.25, label="Event count", align="center")
    ax1b.set_ylabel("Count")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left")

    # Painel B — anomalias dessazonalizadas
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.axhline(0, lw=1, alpha=0.5)
    ax2.plot(m.index, m["anom"], lw=1.0, label="Monthly anomaly (km)")
    ax2.plot(m.index, m["anom_roll12"], lw=2.0, label="12-mo rolling anomaly")

    trend_txt = f"Theil–Sen: {slope_dec:+.1f} km/decade"
    if not np.isnan(pval): trend_txt += f" (MK p = {pval:.3f})"
    ax2.text(0.01, 0.9, trend_txt, transform=ax2.transAxes)

    ax2.set_ylabel("Anomaly (km)")
    ax2.set_title(f"{prefix_letter[1]}{title_prefix}: Deseasonalized trend", loc="left")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="lower left")

    # Eixo x: anos com rótulo e meses como marcadores
    ax2.xaxis.set_major_locator(YearLocator())
    ax2.xaxis.set_major_formatter(DateFormatter('%Y'))
    ax2.xaxis.set_minor_locator(MonthLocator())
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.tick_params(axis='x', which='major', length=6)
    ax2.tick_params(axis='x', which='minor', length=3)

    # ===== Anotar máximo e mínimo, com texto à direita =====
    # índices de máximo e mínimo
    idx_max = m["anom"].idxmax()
    idx_min = m["anom"].idxmin()
    y_max = float(m.loc[idx_max, "anom"]) - 1
    y_min = float(m.loc[idx_min, "anom"]) + 1

    # converter datas para números (dias) para aplicar offset horizontal
    x_max = mdates.date2num(pd.to_datetime(idx_max))
    x_min = mdates.date2num(pd.to_datetime(idx_min))

    # offset horizontal padrão (em dias) e ajuste de borda direita
    xspan_days = (mdates.date2num(m.index.max()) - mdates.date2num(m.index.min()))
    xoff = max(10.0, 0.02 * xspan_days)  # 10 dias ou 2% do span, o que for maior
    right_lim = ax2.get_xlim()[1]
    xtext_max = min(x_max + xoff, right_lim - 0.5 * xoff)
    xtext_min = min(x_min + xoff, right_lim - 0.5 * xoff)

    # deslocamento vertical sutil para evitar sobreposição de seta/linha
    yspan = (np.nanmax(m["anom"]) - np.nanmin(m["anom"])) if np.isfinite(m["anom"]).all() else 1.0
    yoff = -0.04 * yspan if np.isfinite(yspan) and yspan > 0 else 0.5

    # rótulos
    lbl_max = f"Max anomaly: {pd.to_datetime(idx_max).strftime('%b/%Y')} {y_max:+.1f} km"
    lbl_min = f"Min anomaly: {pd.to_datetime(idx_min).strftime('%b/%Y')} {y_min:+.1f} km"

    # anotar máximo (texto à direita)
    ax2.annotate(
        lbl_max,
        xy=(mdates.num2date(x_max), y_max),
        xycoords='data',
        xytext=(mdates.num2date(xtext_max), y_max + yoff),
        textcoords='data',
        arrowprops=dict(arrowstyle="->", lw=1.2, alpha=0.95),
        ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
    )

    # anotar mínimo (texto à direita)
    ax2.annotate(
        lbl_min,
        xy=(mdates.num2date(x_min), y_min),
        xycoords='data',
        xytext=(mdates.num2date(xtext_min), y_min - yoff),
        textcoords='data',
        arrowprops=dict(arrowstyle="->", lw=1.2, alpha=0.95),
        ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
    )

    # estética final
    for ax in (ax1, ax2):
        ax.set_xlabel("")
    plt.setp(ax1.get_xticklabels(), visible=False)

    return fig, (ax1, ax2)


def plot_windrose_grid(
    df,
    step: float = 2.0,
    extent: list | tuple | None = None,
    dir_bins: int = 16,
    dist_stat: str = "median",
    cell_margin_frac: float = 0.06,
    cmap: str = "viridis",
    projection=None,
    show_coastlines: bool = True,
    show_borders: bool = True,
    grid_alpha: float = 0.12,
    grid_edgecolor: str = "k",
    grid_linewidth: float = 0.4,
    figsize=(10, 8),
    title: str | None = "Windroses of start→end directions and distance dispersion",
    radius_mode: str = "frequency",
    gamma: float = 1.0,
    min_radius_frac: float = 0.06,
    color_bins: list | tuple | None = None,  # e.g., [100, 200, 300, 400, 500, 600]
    min_events_per_cell: int = 1,
    base_map=None,
    base_map_style: dict | None = None,
    draw_gridlabels: bool = True,
    show_top: bool = False,
    show_left: bool = True,
    show_right: bool = False,
    show_bottom: bool = True,
    gridline_kwargs: dict | None = None,
):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.colors import BoundaryNorm, Normalize
    from matplotlib.cm import get_cmap, ScalarMappable
    from matplotlib.patches import Rectangle, Wedge
    from shapely.geometry import Point, Polygon, MultiPolygon, LineString, MultiLineString
    import numpy as np
    import pandas as pd
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from shapely import wkt

    # Configurações visuais globais
    mpl.rcParams.update({
        'font.size': 9,
        'axes.titleweight': 'bold',
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'grid.alpha': 0.3
    })

    # Projeção padrão
    if projection is None:
        projection = ccrs.PlateCarree()

    required_cols = {"start_geometry", "end_geometry", "start_end_haversine_km"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Normaliza geometrias (Point ou WKT)
    def to_point(obj):
        if isinstance(obj, Point): 
            return obj
        if isinstance(obj, str):   
            return wkt.loads(obj)
        if hasattr(obj, "geom_type") and obj.geom_type == "Point": 
            return obj
        raise TypeError("start/end_geometry devem ser shapely Point ou WKT de Point.")

    starts = df["start_geometry"].apply(to_point)
    ends   = df["end_geometry"].apply(to_point)

    start_lon = starts.apply(lambda p: float(p.x)).to_numpy()
    start_lat = starts.apply(lambda p: float(p.y)).to_numpy()
    end_lon   = ends.apply(lambda p: float(p.x)).to_numpy()
    end_lat   = ends.apply(lambda p: float(p.y)).to_numpy()
    dist_km   = df["start_end_haversine_km"].to_numpy()

    # Bearing geodésico: 0°=N (norte), sentido horário
    def bearing_deg(lat1, lon1, lat2, lon2):
        phi1 = np.radians(lat1); phi2 = np.radians(lat2)
        dlam = np.radians(lon2 - lon1)
        y = np.sin(dlam) * np.cos(phi2)
        x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlam)
        return (np.degrees(np.arctan2(y, x)) + 360) % 360

    bearings = bearing_deg(start_lat, start_lon, end_lat, end_lon)

    # Extent e grade
    if extent is None:
        pad = 0.5
        xmin = np.floor(start_lon.min() / step) * step - pad
        xmax = np.ceil(start_lon.max() / step) * step + pad
        ymin = np.floor(start_lat.min() / step) * step - pad
        ymax = np.ceil(start_lat.max() / step) * step + pad
        extent = [xmin, xmax, ymin, ymax]
    else:
        xmin, xmax, ymin, ymax = extent

    def center_coord(vals, vmin, step_):
        idx = np.floor((vals - vmin) / step_).astype(int)
        return vmin + idx * step_ + step_ / 2.0

    cx = center_coord(start_lon, xmin, step)
    cy = center_coord(start_lat, ymin, step)

    sector_width = 360.0 / dir_bins
    sec_idx = np.floor(bearings / sector_width).astype(int)
    sec_idx = np.clip(sec_idx, 0, dir_bins - 1)

    work = pd.DataFrame({"cx": cx, "cy": cy, "sector": sec_idx, "dist": dist_km})

    # Estatística de distância por setor
    def agg_dist(x, how="median"):
        if how == "median": return float(np.median(x))
        if how == "mean":   return float(np.mean(x))
        if how == "p25":    return float(np.quantile(x, 0.25))
        if how == "p75":    return float(np.quantile(x, 0.75))
        raise ValueError("dist_stat deve ser 'median', 'mean', 'p25' ou 'p75'.")

    grouped = (
        work.groupby(["cx", "cy", "sector"])
            .agg(count=("dist", "size"),
                 dist_stat=("dist", lambda x: agg_dist(x, dist_stat)))
            .reset_index()
    )
    # Totais por célula e frequência relativa
    cell_counts = grouped.groupby(["cx", "cy"])["count"].sum().rename("cell_total").reset_index()
    grouped = grouped.merge(cell_counts, on=["cx", "cy"], how="left")
    grouped["freq_rel"] = grouped["count"] / grouped["cell_total"].replace(0, np.nan)
    grouped["freq_rel"] = grouped["freq_rel"].fillna(0.0)

    if min_events_per_cell > 1:
        grouped = grouped[grouped["cell_total"] >= int(min_events_per_cell)]

    # === MAPEAMENTO DE CORES ===
    if color_bins is not None and len(color_bins) >= 2:
        bins = np.array(sorted(color_bins), dtype=float)
        grouped["dist_for_color"] = np.clip(grouped["dist_stat"], bins[0], bins[-1])
        cmap_obj = get_cmap(cmap, len(bins) - 1)          # discreto
        norm = BoundaryNorm(bins, ncolors=cmap_obj.N, clip=True)
        sm = ScalarMappable(norm=norm, cmap=cmap_obj)
        cbar_kw = dict(boundaries=bins, ticks=bins, spacing="proportional")
    else:
        # contínuo (robusto P5–P95) caso bins não sejam informados
        valid_vals = grouped["dist_stat"].dropna()
        if len(valid_vals) > 0:
            vmin = float(np.nanpercentile(valid_vals, 5))
            vmax = float(np.nanpercentile(valid_vals, 95))
            if not np.isfinite(vmin) or vmin == vmax:
                vmin, vmax = 0.0, valid_vals.max() * 1.1 if valid_vals.max() > 0 else 1.0
        else:
            vmin, vmax = 0.0, 1.0
        cmap_obj = get_cmap(cmap)
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(norm=norm, cmap=cmap_obj)
        grouped["dist_for_color"] = grouped["dist_stat"]
        cbar_kw = {'format': '%.1f'}

    # === FIGURA / CARTOPY ===
    fig = plt.figure(figsize=figsize, facecolor='white', edgecolor='black')
    ax = plt.axes(projection=projection)
    ax.set_extent([xmin, xmax, ymin, ymax], crs=ccrs.PlateCarree())
    ax.set_facecolor('#f8f9fa')  # Fundo suave

    # Base features melhoradas (cores harmonizadas)
    land_color = '#f5f5f5'
    ocean_color = '#e6f2ff'
    coastline_color = '#4a6fa5'
    border_color = '#6a8db8'
    
    ax.add_feature(cfeature.LAND, facecolor=land_color, zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor=ocean_color, zorder=0, alpha=0.6)
    if show_coastlines: 
        ax.coastlines(resolution='50m', color=coastline_color, linewidth=0.7, zorder=2)
    if show_borders:    
        ax.add_feature(cfeature.BORDERS, edgecolor=border_color, linewidth=0.5, zorder=2, alpha=0.8)

    # NOVO: base_map (GeoDataFrame) — método compatível com versões recentes do Cartopy
    if base_map is not None:
        try:
            gdf = base_map.copy()
            # Garante CRS geográfico
            if hasattr(gdf, "crs") and gdf.crs is not None and gdf.crs != "EPSG:4326":
                gdf_wgs84 = gdf.to_crs("EPSG:4326")
            else:
                gdf_wgs84 = gdf

            # Estilo padrão melhorado
            default_style = {
                'edgecolor': '#2c3e50',
                'facecolor': 'none',
                'linewidth': 0.8,
                'alpha': 0.8,
                'linestyle': '-'
            }
            if base_map_style:
                default_style.update(base_map_style)

            # Adiciona geometrias diretamente (compatível com Cartopy >=0.18)
            for geom in gdf_wgs84.geometry:
                if geom.is_empty:
                    continue
                    
                # Trata diferentes tipos de geometria
                if isinstance(geom, (Polygon, MultiPolygon)):
                    ax.add_geometries(
                        [geom],
                        crs=ccrs.PlateCarree(),
                        facecolor=default_style['facecolor'],
                        edgecolor=default_style['edgecolor'],
                        linewidth=default_style['linewidth'],
                        alpha=default_style['alpha'],
                        linestyle=default_style['linestyle'],
                        zorder=3
                    )
                elif isinstance(geom, (LineString, MultiLineString)):
                    ax.add_geometries(
                        [geom],
                        crs=ccrs.PlateCarree(),
                        facecolor='none',
                        edgecolor=default_style['edgecolor'],
                        linewidth=default_style['linewidth'] * 1.2,
                        alpha=default_style['alpha'],
                        linestyle=default_style['linestyle'],
                        zorder=3
                    )
        except Exception as e:
            print(f"[plot_windrose_grid] Aviso: falha ao adicionar base_map ({e}).")

    # Malha (quadrantes) com alpha
    lon_edges = np.arange(xmin, xmax + step, step)
    lat_edges = np.arange(ymin, ymax + step, step)
    
    # Substitui retângulos por linhas discretas
    for lon in lon_edges:
        ax.plot([lon, lon], [ymin, ymax], 
                transform=ccrs.PlateCarree(),
                color=grid_edgecolor, 
                linewidth=grid_linewidth * 0.8,
                alpha=grid_alpha * 1.5,
                linestyle=':',
                zorder=1)
    
    for lat in lat_edges:
        ax.plot([xmin, xmax], [lat, lat], 
                transform=ccrs.PlateCarree(),
                color=grid_edgecolor, 
                linewidth=grid_linewidth * 0.8,
                alpha=grid_alpha * 1.5,
                linestyle=':',
                zorder=1)

    # Raio máximo (circunferência inscrita)
    r_cell_max = 0.5 * step * (1.0 - cell_margin_frac)

    # Setores com raio normalizado localmente
    for (cx_val, cy_val), df_cell in grouped.groupby(["cx", "cy"]):
        cell_total = df_cell["cell_total"].iloc[0]
        fmax = float(df_cell["freq_rel"].max()) if not df_cell.empty else 0.0
        
        # Indicador visual para células com poucos eventos
        if cell_total < min_events_per_cell:
            circ = plt.Circle((cx_val, cy_val), r_cell_max * 0.4,
                             color='#ffcccc', alpha=0.4, zorder=5,
                             transform=ccrs.PlateCarree())
            ax.add_patch(circ)
            continue
            
        if fmax <= 0:
            # Célula vazia: círculo tracejado
            circ = plt.Circle((cx_val, cy_val), r_cell_max,
                             fill=False, edgecolor='#aaaaaa', 
                             linestyle='--', linewidth=0.7,
                             transform=ccrs.PlateCarree(), zorder=3)
            ax.add_patch(circ)
            continue

        # Desenha setores
        for _, row in df_cell.iterrows():
            if row["freq_rel"] <= 0:
                continue
            k = int(row["sector"])
            theta1_brg = k * sector_width
            theta2_brg = (k + 1) * sector_width
            theta1 = 90.0 - theta2_brg
            theta2 = 90.0 - theta1_brg

            f_local = float(row["freq_rel"] / fmax)
            f_local = f_local ** max(0.0, gamma)
            f_local = max(f_local, float(np.clip(min_radius_frac, 0.0, 1.0)))
            r = r_cell_max * f_local

            color = cmap_obj(norm(row["dist_for_color"]))

            wedge = Wedge(
                center=(cx_val, cy_val),
                r=r,
                theta1=theta1,
                theta2=theta2,
                facecolor=color,
                edgecolor='white',  # Borda branca para separação
                linewidth=0.3,
                alpha=0.92,
                transform=ccrs.PlateCarree(),
                zorder=4
            )
            ax.add_patch(wedge)

        # Círculo externo com borda destacada
        circ = plt.Circle(
            (cx_val, cy_val), 
            r_cell_max,
            fill=False,
            edgecolor='#2c3e50',
            linewidth=0.9,
            alpha=0.85,
            transform=ccrs.PlateCarree(),
            zorder=5
        )
        ax.add_patch(circ)
        
        # # Rótulo de contagem (apenas para células relevantes)
        # if cell_total >= 5:
        #     ax.text(
        #         cx_val, cy_val - 0.12*step,
        #         f'n={int(cell_total)}',
        #         ha='center', va='center',
        #         fontsize=7,
        #         color='#2c3e50',
        #         alpha=0.85,
        #         transform=ccrs.PlateCarree(),
        #         zorder=6,
        #         bbox=dict(boxstyle="round,pad=0.2", 
        #                  fc="white", 
        #                  ec="none", 
        #                  alpha=0.7)
        #     )

    # Colorbar estilizado
    cbar = plt.colorbar(
        sm, 
        ax=ax, 
        orientation="vertical",
        extend='max',
        fraction=0.025,
        pad=0.04,
        aspect=25,
        **cbar_kw
    )
    cbar.set_label('Distance [km]', labelpad=10, fontsize=10)
    cbar.ax.tick_params(labelsize=10, width=0.5)
    for spine in cbar.ax.spines.values():
        spine.set_edgecolor('#666666')
        spine.set_linewidth(0.6)

    # NOVO: gridlines com labels (compatível com versões recentes)
    if draw_gridlabels:
        default_gl = {
            'draw_labels': True,
            'dms': True,
            'x_inline': False,
            'y_inline': False,
            'linewidth': 0.5,
            'color': '#888888',
            'alpha': 0.5,
            'linestyle': '--',
            'xlabel_style': {'size': 8, 'color': '#333333'},
            'ylabel_style': {'size': 8, 'color': '#333333'},
        }
        if gridline_kwargs:
            default_gl.update(gridline_kwargs)
            
        gl = ax.gridlines(**default_gl)
        
        # Configuração de labels por lado (compatível com Cartopy >=0.18)
        gl.top_labels = show_top
        gl.bottom_labels = show_bottom
        gl.left_labels = show_left
        gl.right_labels = show_right
        
        # Para versões antigas do Cartopy (fallback)
        if hasattr(gl, 'xlabels_top'):
            gl.xlabels_top = show_top
        if hasattr(gl, 'xlabels_bottom'):
            gl.xlabels_bottom = show_bottom
        if hasattr(gl, 'ylabels_left'):
            gl.ylabels_left = show_left
        if hasattr(gl, 'ylabels_right'):
            gl.ylabels_right = show_right

    if title:
        ax.set_title(
            title,
            loc='left',
            pad=15,
            fontsize=12,
            fontweight='bold',
            color='#2c3e50'
        )

    # Ajuste fino do layout
    plt.tight_layout(pad=2.0)
    fig.subplots_adjust(right=0.92)  # Espaço para colorbar
    
    return fig, ax