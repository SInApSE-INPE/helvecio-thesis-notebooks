import pathlib
import duckdb
import geopandas as gpd
import pandas as pd
import os

def calculate_stats(TRK_FILES, con, threshold=0.1, save_path='/mnt/data/tracks/gsmap_v8_2015-2024/01mmhr_v8/filters/'):
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    stats_file = pathlib.Path(save_path) / f'calculate_stats_{threshold}.parquet'
    if stats_file.exists():
        print(f"Loading existing stats from {stats_file}")
        df_stats = con.execute(f"SELECT * FROM read_parquet('{stats_file}')").df()
        return df_stats
    df_general_gsmap = con.execute(f"""
    WITH base AS (
        SELECT *
        FROM read_parquet({TRK_FILES}, union_by_name=true)
        WHERE threshold = {threshold}
    ),

    uid_duration AS (
        -- Etapa 1: pegar a duração final (maior) de cada uid
        SELECT
            uid,
            MAX(duration) AS duration
        FROM base
        GROUP BY uid
    ),

    uid_sizes AS (
        -- Etapa 2: pegar todos os valores de size por uid
        SELECT
            uid,
            size
        FROM base
    ),

    merged AS (
        -- Etapa 3: associar a duração final aos valores de size
        SELECT
            d.duration,
            s.uid,
            s.size
        FROM uid_duration d
        JOIN uid_sizes s USING (uid)
    ),

    duration_stats AS (
        -- Etapa 4: agregação final por duração final
        SELECT
            duration,
            COUNT(DISTINCT uid) AS uid_count,
            AVG(size) AS mean_size,
            STDDEV_SAMP(size) AS std_size
        FROM merged
        GROUP BY duration
    ),

    total_count AS (
        SELECT SUM(uid_count) AS total_uids
        FROM duration_stats
    )

    -- Etapa 5: resultado final com porcentagem
    SELECT
        d.duration,
        d.uid_count,
        ROUND(100.0 * d.uid_count / t.total_uids, 2) AS percentage,
        ROUND(d.mean_size, 2) AS mean_size,
        ROUND(d.std_size, 2) AS std_size
    FROM duration_stats d, total_count t
    ORDER BY d.duration;
    """).df()
    df_general_gsmap.to_parquet(stats_file, index=False)
    print(f"Statistics saved to {stats_file}")
    return df_general_gsmap


def classificationFilter(con, db):

    ## Classification of events
    classification = con.execute(f"""
            WITH grouped_status AS (
            SELECT
                uid,
                threshold,
                arbitrary(duration) AS duration,
                array_agg(latitude) AS latitudes,
                array_agg(status) AS all_status,
                array_agg(region) AS regions,
                MIN(timestamp) AS start_time,
                MAX(timestamp) AS end_time
            FROM {db}
            GROUP BY uid, threshold
        )

        SELECT
            uid,
            threshold,
            duration,
            start_time,
            end_time,

            -- Classificação baseada em status
            CASE
                WHEN 
                    list_count(list_filter(all_status, x -> x = 'NEW')) = 1 AND
                    list_count(list_filter(all_status, x -> x = 'CON')) = list_count(all_status) - 1
                THEN 'Continuous'
                ELSE 'Non-Continuous'
            END AS classification,

            -- Classificação baseada em latitude
            CASE 
                WHEN list_count(list_filter(latitudes, x -> x = 'Tropical')) = list_count(latitudes)
                THEN 'Tropical'
                ELSE 'Extratropical'
            END AS latitude_region,

            -- Classificação baseada em regions
            CASE
                WHEN list_count(list_filter(regions, x -> 
                    x ILIKE '%Ocean%' OR 
                    x ILIKE '%Sea%' OR 
                    x ILIKE '%Mediterranean%'
                )) = list_count(regions) THEN 'Ocean'
                
                WHEN list_count(list_filter(regions, x -> 
                    x ILIKE '%Ocean%' OR 
                    x ILIKE '%Sea%' OR 
                    x ILIKE '%Mediterranean%'
                )) > 0 THEN 'Transition'
                
                ELSE 'Land'
            END AS surface_type

        FROM grouped_status
        WHERE threshold = 0.1;
        """).df()
    return classification


def classificationFilterFeatures(con, db):

    ## Classification of events
    classification = con.execute(f"""
            WITH grouped_status AS (
            SELECT
                uid,
                threshold,
                arbitrary(duration) AS duration,
                array_agg(latitude) AS latitudes,
                array_agg(status) AS all_status,
                array_agg(region) AS regions,
                MIN(timestamp) AS start_time,
                MAX(timestamp) AS end_time,
                AVG(size) AS mean_size,
                MAX(max) AS max_precipitation,
                MAX(inside_clusters) AS max_n_clusters
            FROM {db}
            GROUP BY uid, threshold
        )

        SELECT
            uid,
            threshold,
            duration,
            start_time,
            end_time,
            mean_size,
            max_precipitation,
            max_n_clusters,

            -- Classificação baseada em status
            CASE
                WHEN 
                    list_count(list_filter(all_status, x -> x = 'NEW')) = 1 AND
                    list_count(list_filter(all_status, x -> x = 'CON')) = list_count(all_status) - 1
                THEN 'Continuous'
                ELSE 'Non-Continuous'
            END AS classification,

            -- Classificação baseada em latitude
            CASE 
                WHEN list_count(list_filter(latitudes, x -> x = 'Tropical')) = list_count(latitudes)
                THEN 'Tropical'
                ELSE 'Extratropical'
            END AS latitude_region,

            -- Classificação baseada em regions
            CASE
                WHEN list_count(list_filter(regions, x -> 
                    x ILIKE '%Ocean%' OR 
                    x ILIKE '%Sea%' OR 
                    x ILIKE '%Mediterranean%'
                )) = list_count(regions) THEN 'Ocean'
                
                WHEN list_count(list_filter(regions, x -> 
                    x ILIKE '%Ocean%' OR 
                    x ILIKE '%Sea%' OR 
                    x ILIKE '%Mediterranean%'
                )) > 0 THEN 'Transition'
                
                ELSE 'Land'
            END AS surface_type
        FROM grouped_status
        WHERE threshold = 0.1;
        """).df()
    return classification


def evolutionByLifetime(con, table, save_path='queries/evolution/'):
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    file_path = pathlib.Path(save_path) / f"{table}_evolution_by_lifetime.parquet"

    delta = {'gsmap_v8': 1, 'imerg_v7': 0.5}.get(table, 0)

    if file_path.exists():
        print(f"File {file_path} already exists. Skipping query execution.")
        return
    query = f"""
        WITH collect_data AS (
        SELECT
            timestamp,
            lifetime,
            size,
            latitude,
            region,
            genesis,
            max as max_precipitation,
            mean as mean_precipitation,
            inside_clusters,
            ST_Centroid(ST_GeomFromText(geometry)) AS centroid
        FROM {table}
        )
        SELECT
            timestamp,
            (lifetime / 60) - {delta} AS lifetime,
            size / (0.1 * 0.1) AS size,
            CASE
                WHEN latitude LIKE '%Tropical%' THEN 'Tropical'
                ELSE 'Extratropical'
            END AS latitude,
            CASE
                WHEN region ILIKE '%Ocean%' OR region ILIKE '%Sea%' THEN 'Ocean'
                ELSE 'Land'
            END AS region_type,
            region,
            genesis,
            CASE
                WHEN ((EXTRACT(hour FROM timestamp) + CAST(ST_X(centroid) / 15 AS INT)) % 24) BETWEEN 6 AND 17
                THEN 'Day'
                ELSE 'Night'
            END AS day_night,
            ST_AsText(centroid) AS centroid_wkt,
            max_precipitation,
            mean_precipitation,
            inside_clusters
        FROM collect_data
        """
    con.execute(f"""
            COPY ({query})
            TO '{file_path}'
            (FORMAT 'parquet', OVERWRITE)
            """)
    print(f"Evolution data saved to {file_path}")

def method_comparison(con, table_name, out_dir="queries/method_comparison/", filename=None, force=False):

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"{table_name}_method_comparison.parquet"
    out_path = out_dir / filename


    # Query principal
    query = f"""
    WITH base AS (
      SELECT
        row_number() OVER () AS rid,
        method AS row_method,
        far_      AS far_noc,
        far_opt   AS far_opt,
        far_inc   AS far_inc,
        far_spl   AS far_spl,
        far_mrg   AS far_mrg,
        far_elp   AS far_elp
      FROM {table_name}
    ),
    long AS (
      SELECT rid, row_method, 'noc' AS comp_method, far_noc AS far FROM base
      UNION ALL SELECT rid, row_method, 'opt', far_opt FROM base
      UNION ALL SELECT rid, row_method, 'inc', far_inc FROM base
      UNION ALL SELECT rid, row_method, 'spl', far_spl FROM base
      UNION ALL SELECT rid, row_method, 'mrg', far_mrg FROM base
      UNION ALL SELECT rid, row_method, 'elp', far_elp FROM base
    ),
    comparisons_valid AS (
      SELECT *
      FROM long
      WHERE far IS NOT NULL AND far <> 1
    ),
    winners AS (
      SELECT *,
             MIN(far) OVER (PARTITION BY rid) AS min_far
      FROM comparisons_valid
    ),
    winner_rows AS (
      SELECT
        rid,
        ANY_VALUE(row_method) AS row_method,
        LIST(comp_method) FILTER (WHERE far = min_far) AS winner_methods,
        min_far AS winner_far
      FROM winners
      GROUP BY rid, min_far
    ),
    pairwise AS (
      SELECT
        w.rid,
        w.row_method,
        wm AS winner_method,
        w.winner_far,
        c.comp_method AS opponent_method,
        c.far AS opponent_far
      FROM winner_rows w
      CROSS JOIN UNNEST(w.winner_methods) AS t(wm)
      JOIN comparisons_valid c
        ON c.rid = w.rid
       AND c.comp_method <> wm
    ),
    effective AS (
      SELECT *
      FROM pairwise
      WHERE winner_method = row_method
    ),
    aggregated AS (
      SELECT
        winner_method AS method,
        opponent_method,
        COUNT(*) AS n_duels,
        AVG(winner_far) AS mean_far_winner,
        AVG(opponent_far) AS mean_far_opponent,
        AVG(opponent_far - winner_far) AS mean_far_gap
      FROM effective
      GROUP BY 1, 2
    )
    SELECT * FROM aggregated
    ORDER BY method, opponent_method
    """

    # Executa e salva como parquet
    con.execute(f"COPY ({query}) TO '{out_path.as_posix()}' (FORMAT PARQUET)")

def stratified_far_dataset(con, table_name, out_dir="queries/stratified_far_dataset/", filename=None, force=False, temp_res=30):
    """
    Gera e salva um dataset estratificado por tamanho (size) e duração (duration),
    incluindo classificação 'Continuous' / 'Non-Continuous' e valores de FAR por método.
    """

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"{table_name}_stratified_far.parquet"
    out_path = out_dir / filename

    query = f"""
        COPY (
            WITH
            -- (1) Filtro inicial otimizado: remove NEW e FAR inválidos logo no scan
            filtered AS (
                SELECT
                    uid,
                    method,
                    far,
                    size,
                    duration,
                    lifetime,
                    status,
                    far_      AS far_noc,
                    far_opt   AS far_opt,
                    far_inc   AS far_inc,
                    far_spl   AS far_spl,
                    far_mrg   AS far_mrg,
                    far_elp   AS far_elp,
                    u_,                -- componentes originais do rastreio
                    v_
                FROM {table_name}
                WHERE size IS NOT NULL
                AND duration IS NOT NULL
                AND far IS NOT NULL
                AND far <> 1
                AND far <> 0
                AND status <> 'NEW'
                -- LIMIT 100000
            ),

            -- (2) Classificação revisada + médias das componentes em m/s
            event_agg AS (
                SELECT
                    uid,
                    AVG(size)     AS mean_size,
                    MAX(duration) AS duration_event,

                    -- fator de conversão: 0.1° em metros, passo de 30 min
                    AVG(u_ * 0.1 * 111000.0 / ({temp_res} * 60.0)) AS mean_u_ms,
                    AVG(v_ * 0.1 * 111000.0 / ({temp_res} * 60.0)) AS mean_v_ms,

                    CASE
                        WHEN COUNT(*) = SUM(CASE WHEN status = 'CON' THEN 1 ELSE 0 END)
                        THEN 'Continuous'
                        ELSE 'Non-Continuous'
                    END AS classification
                FROM filtered
                GROUP BY uid
            ),

            -- (3) Estatísticas globais (leves)
            global_stats AS (
                SELECT
                    AVG(mean_size)                              AS mean_size,
                    STDDEV_SAMP(mean_size)                      AS std_size,
                    AVG(mean_size) + STDDEV_SAMP(mean_size)     AS lim1_size,
                    AVG(mean_size) + 2 * STDDEV_SAMP(mean_size) AS lim2_size,
                    AVG(duration_event)                         AS mean_duration,
                    STDDEV_SAMP(duration_event)                 AS std_duration,
                    AVG(duration_event) + STDDEV_SAMP(duration_event) AS lim1_duration
                FROM event_agg
            )

            -- (4) Seleção final
            SELECT
                s.uid,
                a.classification,
                s.method,

                regexp_replace(
                    (CASE WHEN s.far_noc IS NOT NULL AND s.far_noc <> 1 AND s.method <> 'noc' THEN 'noc, ' ELSE '' END) ||
                    (CASE WHEN s.far_opt IS NOT NULL AND s.far_opt <> 1 AND s.method <> 'opt' THEN 'opt, ' ELSE '' END) ||
                    (CASE WHEN s.far_inc IS NOT NULL AND s.far_inc <> 1 AND s.method <> 'inc' THEN 'inc, ' ELSE '' END) ||
                    (CASE WHEN s.far_spl IS NOT NULL AND s.far_spl <> 1 AND s.method <> 'spl' THEN 'spl, ' ELSE '' END) ||
                    (CASE WHEN s.far_mrg IS NOT NULL AND s.far_mrg <> 1 AND s.method <> 'mrg' THEN 'mrg, ' ELSE '' END) ||
                    (CASE WHEN s.far_elp IS NOT NULL AND s.far_elp <> 1 AND s.method <> 'elp' THEN 'elp, ' ELSE '' END),
                    ', $', ''
                ) AS other_methods,

                s.far,
                s.size      AS size,
                s.lifetime  AS lifetime,
                a.mean_size,
                a.duration_event,

                -- novas colunas: média das componentes de velocidade em m/s
                a.mean_u_ms AS u_ms,
                a.mean_v_ms AS v_ms,

                CASE
                    WHEN a.mean_size < gs.mean_size THEN 'Small'
                    WHEN a.mean_size < gs.lim1_size THEN 'Medium'
                    ELSE 'Large'
                END AS size_class,

                CASE
                    WHEN a.duration_event < gs.mean_duration THEN 'Short'
                    WHEN a.duration_event < gs.lim1_duration THEN 'Medium'
                    ELSE 'Long'
                END AS duration_class

            FROM filtered s
            JOIN event_agg a USING (uid)
            CROSS JOIN global_stats gs
        ) TO '{out_path.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """


    con.execute(query)
    # import pandas as pd
    # Executa e salva direto em Parquet
    # con.execute(f"COPY ({query}) TO '{out_path.as_posix()}' (FORMAT PARQUET);")
    # return pd.read_parquet(out_path)

def get_start_end_trajectory(con, uids, table, file):

    out_path = pathlib.Path(file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # check if file exists
    if out_path.exists():
        print(f"File {out_path} already exists. Skipping query execution.")
        return
    
    uids = ', '.join(f"'{uid}'" for uid in uids)

    query = f"""
    WITH cindex_bounds AS (
        SELECT
            uid,
            MIN(cindex) AS start_cindex,
            MAX(cindex) AS end_cindex
        FROM {table}
        WHERE uid IN ({uids})
        GROUP BY uid
    ),
    start_geom AS (
        SELECT 
            g.uid,
            b.start_cindex,
            g.timestamp AS start_timestamp,
            ST_Centroid(ST_GeomFromText(g.geometry)) AS start_geom
        FROM {table} g
        JOIN cindex_bounds b 
          ON g.uid = b.uid AND g.cindex = b.start_cindex
    ),
    end_geom AS (
        SELECT 
            g.uid,
            b.end_cindex,
            g.timestamp AS end_timestamp,
            ST_Centroid(ST_GeomFromText(g.geometry)) AS end_geom,
            g.trajectory
        FROM {table} g
        JOIN cindex_bounds b 
          ON g.uid = b.uid AND g.cindex = b.end_cindex
    ),
    path AS (
        SELECT 
            s.uid,
            s.start_cindex,
            e.end_cindex,
            s.start_timestamp,
            e.end_timestamp,
            ST_AsText(s.start_geom) AS start_geometry,
            ST_AsText(e.end_geom)   AS end_geometry,
            e.trajectory,
            ST_X(s.start_geom) AS lon1,
            ST_Y(s.start_geom) AS lat1,
            ST_X(e.end_geom)   AS lon2,
            ST_Y(e.end_geom)   AS lat2
        FROM start_geom s
        JOIN end_geom e USING (uid)
    )
    SELECT
        *,
        -- Distância Haversine entre os centroides (km)
        6371.0 * 2 * ASIN(
            SQRT(
                POWER(SIN(RADIANS(lat2 - lat1) / 2), 2) +
                COS(RADIANS(lat1)) * COS(RADIANS(lat2)) *
                POWER(SIN(RADIANS(lon2 - lon1) / 2), 2)
            )
        ) AS start_end_haversine_km,
        -- Comprimento da trajectory
        ST_Length(
          ST_Transform(
            ST_GeomFromText(trajectory),
            'EPSG:4326','EPSG:3857'
          )
        ) / 1000.0 AS trajectory_length_km
    FROM path
    """

    con.execute(f"""
        COPY ({query})
        TO '{out_path.as_posix()}'
        (FORMAT 'parquet', OVERWRITE)
    """)
    print(f"Start, end, and trajectory data saved to {out_path}")



def get_region_tracks(con, roi_gdf, table, file):

    out_path = pathlib.Path(file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ROI (WGS84) → WKB
    if roi_gdf.crs is None:
        raise ValueError("roi_gdf precisa ter CRS definido (ex.: EPSG:4326).")
    roi = roi_gdf.to_crs(4326)
    roi_union = roi.unary_union
    roi_df = pd.DataFrame({"geom_wkb": [roi_union.wkb]})
    con.register("roi_df", roi_df)

    try:
        con.execute("INSTALL spatial;")
    except Exception:
        pass
    con.execute("LOAD spatial;")

    # 1) Coleta de UIDs por interseção do CENTRÓIDE com a ROI
    #    (pré-filtro por envelope opcional para baratear a operação)
    uids_df = con.execute(f"""
        WITH roi AS (SELECT ST_GeomFromWKB(geom_wkb) AS geom FROM roi_df),
             base AS (
                SELECT uid, geometry
                FROM {table}
                WHERE geometry IS NOT NULL AND LENGTH(TRIM(geometry)) > 0
             ),
             g AS (
                SELECT uid,
                       ST_GeomFromText(geometry) AS geom
                FROM base
             ),
             g_ok AS (
                SELECT uid, geom
                FROM g
                WHERE geom IS NOT NULL AND NOT ST_IsEmpty(geom) AND ST_IsValid(geom)
             ),
             centroids AS (
                SELECT uid,
                       ST_Centroid(geom) AS cpt,
                       ST_Envelope(geom) AS env
                FROM g_ok
             )
        SELECT DISTINCT c.uid
        FROM centroids c, roi r
        WHERE ST_Intersects(ST_Envelope(r.geom), c.env)
          AND ST_Intersects(c.cpt, r.geom)
    """).fetchdf()

    uids_list = uids_df["uid"].tolist()
    if not uids_list:
        print("Nenhum uid intersectou a ROI. Nada a exportar.")
        return

    # Monta exatamente como você fazia:
    uids = ", ".join(f"'{str(uid)}'" for uid in uids_list)

    # 2) Query original (com IN (uids)), exportando WKT
    query = f"""
    WITH cindex_bounds AS (
        SELECT
            uid,
            MIN(cindex) AS start_cindex,
            MAX(cindex) AS end_cindex
        FROM {table}
        WHERE uid IN ({uids})
        GROUP BY uid
    ),
    start_geom AS (
        SELECT 
            g.uid,
            b.start_cindex,
            g.timestamp AS start_timestamp,
            ST_Centroid(ST_GeomFromText(g.geometry)) AS start_geom
        FROM {table} g
        JOIN cindex_bounds b 
          ON g.uid = b.uid AND g.cindex = b.start_cindex
    ),
    end_geom AS (
        SELECT 
            g.uid,
            b.end_cindex,
            g.timestamp AS end_timestamp,
            ST_Centroid(ST_GeomFromText(g.geometry)) AS end_geom,
            g.trajectory
        FROM {table} g
        JOIN cindex_bounds b 
          ON g.uid = b.uid AND g.cindex = b.end_cindex
    ),
    path AS (
        SELECT 
            s.uid,
            s.start_cindex,
            e.end_cindex,
            s.start_timestamp,
            e.end_timestamp,
            ST_AsText(s.start_geom) AS start_geometry,
            ST_AsText(e.end_geom)   AS end_geometry,
            e.trajectory,
            ST_X(s.start_geom) AS lon1,
            ST_Y(s.start_geom) AS lat1,
            ST_X(e.end_geom)   AS lon2,
            ST_Y(e.end_geom)   AS lat2
        FROM start_geom s
        JOIN end_geom e USING (uid)
    )
    SELECT
        *,
        6371.0 * 2 * ASIN(
            SQRT(
                POWER(SIN(RADIANS(lat2 - lat1) / 2), 2) +
                COS(RADIANS(lat1)) * COS(RADIANS(lat2)) *
                POWER(SIN(RADIANS(lon2 - lon1) / 2), 2)
            )
        ) AS start_end_haversine_km,
        ST_Length(
          ST_Transform(
            ST_GeomFromText(trajectory),
            'EPSG:4326','EPSG:3857'
          )
        ) / 1000.0 AS trajectory_length_km
    FROM path
    """

    # IMPORTANTE: não coloque ';' dentro do COPY (subquery)
    con.execute(f"""
        COPY ({query})
        TO '{out_path.as_posix()}'
        (FORMAT 'parquet', OVERWRITE)
    """)
    print(f"Start, end, and trajectory data saved to {out_path}")

def calc_distances(con, uids, table, file):
    """
    Calcula unicamente a distância Haversine (km) entre os centróides
    de início e fim para cada uid informado, sem retornar geometrias.
    Salva/recupera de Parquet.
    """
    out_path = pathlib.Path(file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    uids_in = ', '.join(f"'{uid}'" for uid in uids)

    query = f"""
    WITH cindex_bounds AS (
        SELECT
            uid,
            MIN(cindex) AS start_cindex,
            MAX(cindex) AS end_cindex
        FROM {table}
        WHERE uid IN ({uids_in})
        GROUP BY uid
    ),
    start_geom AS (
        SELECT 
            g.uid,
            b.start_cindex,
            ST_Centroid(ST_GeomFromText(g.geometry)) AS start_geom
        FROM {table} g
        JOIN cindex_bounds b 
          ON g.uid = b.uid AND g.cindex = b.start_cindex
    ),
    end_geom AS (
        SELECT 
            g.uid,
            b.end_cindex,
            ST_Centroid(ST_GeomFromText(g.geometry)) AS end_geom
        FROM {table} g
        JOIN cindex_bounds b 
          ON g.uid = b.uid AND g.cindex = b.end_cindex
    ),
    path AS (
        SELECT 
            s.uid,
            -- extrai coord. numéricas (não retorna geometrias)
            ST_X(s.start_geom) AS lon1,
            ST_Y(s.start_geom) AS lat1,
            ST_X(e.end_geom)   AS lon2,
            ST_Y(e.end_geom)   AS lat2
        FROM start_geom s
        JOIN end_geom e USING (uid)
    )
    SELECT
        uid,
        6371.0 * 2 * ASIN(
            SQRT(
                POWER(SIN(RADIANS(lat2 - lat1) / 2), 2) +
                COS(RADIANS(lat1)) * COS(RADIANS(lat2)) *
                POWER(SIN(RADIANS(lon2 - lon1) / 2), 2)
            )
        ) AS start_end_haversine_km
    FROM path
    """

    con.execute(f"""
        COPY ({query})
        TO '{out_path.as_posix()}'
        (FORMAT 'parquet', OVERWRITE);
    """)

 

# If main
if __name__ == '__main__':

    # Create a client for duckdb
    con = duckdb.connect(database=':memory:')

    # Client for duckdb and configure some settings for performance
    con = duckdb.connect(database=':memory:')
    con.execute(f"PRAGMA threads={os.cpu_count()};")
    con.execute("SET memory_limit='90GB';")
    con.execute("SET enable_progress_bar = true")
    con.execute("SET progress_bar_time = 100;")
    # con.execute("PRAGMA enable_object_cache;")
    con.execute(f"SET temp_directory='/storage/tmp/';")
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")

    # Set path for the tracking table files
    gsmap_trackingtable = '/prj/cptec/helvecio.leal/tracks/gsmap/track/trackingtable/*.parquet'
    imerg_trackingtable = '/prj/cptec/helvecio.leal/tracks/imerg/track/trackingtable/*.parquet'

    # Load datasets
    con.execute(f"""
        CREATE OR REPLACE VIEW gsmap_tracking AS
        SELECT *
        FROM read_parquet('{gsmap_trackingtable}', union_by_name=true)
        WHERE threshold = 0.1 AND
        timestamp >= '2015-01-01 00:00:00' AND 
        timestamp < '2025-01-01 00:00:00' AND 
        duration >= 120
    """)
    con.execute(f"""
        CREATE OR REPLACE VIEW imerg_tracking AS
        SELECT *
        FROM read_parquet('{imerg_trackingtable}', union_by_name=true)
        WHERE threshold = 0.1 AND
        timestamp >= '2015-01-01 00:00:00' AND 
        timestamp < '2025-01-01 00:00:00' AND 
        duration >= 120
    """)

    # Execute only evolutionByLifetime
    # evolutionByLifetime(con, 'gsmap_tracking')
    # evolutionByLifetime(con, 'imerg_tracking')
    # gsmap_comp = method_comparison(con, "gsmap_tracking", force=True)
    # imerg_comp = method_comparison(con, "imerg_tracking", force=True)


    stratified_far_dataset(con, 'gsmap_tracking', force=True, temp_res=60)
    stratified_far_dataset(con, 'imerg_tracking', force=True, temp_res=30)

    # gsmap_uid_larger_longer_list = con.execute("""
    # SELECT DISTINCT uid
    # FROM 'queries/stratified_far_dataset/gsmap_tracking_stratified_far.parquet'
    # WHERE duration_class = 'Long';
    # """).df()['uid'].tolist()
    
    # get_start_end_trajectory(con, 
    #     gsmap_uid_larger_longer_list, 
    #     'gsmap_tracking',
    #     'queries/start_end_trajectory/gsmap_tracking_longer.parquet'
    # )

    # imerg_uid_larger_longer_list = con.execute("""
    # SELECT DISTINCT uid
    # FROM 'queries/stratified_far_dataset/imerg_tracking_stratified_far.parquet'
    # WHERE duration_class = 'Long';
    # """).df()['uid'].tolist()

    # get_start_end_trajectory(con, 
    #     imerg_uid_larger_longer_list, 
    #     'imerg_tracking',
    #     'queries/start_end_trajectory/imerg_tracking_longer.parquet'
    # )


    # uids_by_class = con.execute("""
    # SELECT
    #     duration_class,
    #     LIST(DISTINCT uid) AS uids
    # FROM read_parquet('queries/stratified_far_dataset/gsmap_tracking_stratified_far.parquet', union_by_name=true)
    # GROUP BY duration_class
    # ORDER BY duration_class
    # """).df()

    # print(uids_by_class)
    # for _, row in uids_by_class.iterrows():
        
    #     dclass = row["duration_class"]
    #     uids   = row["uids"]
    #     print("GSMAP Processing duration class:", row["duration_class"], len(uids))
    #     out    = f"queries/haversine_by_class/gsmap_{dclass}.parquet"
    #     calc_distances(con, uids, table="gsmap_tracking", file=out)
    #     print(dclass, len(uids), "linhas salvas em", out)

    # uids_by_class_imerg = con.execute("""
    # SELECT
    #     duration_class,
    #     LIST(DISTINCT uid) AS uids
    # FROM read_parquet('queries/stratified_far_dataset/imerg_tracking_stratified_far.parquet', union_by_name=true)
    # GROUP BY duration_class
    # ORDER BY duration_class
    # """).df()
    # for _, row in uids_by_class_imerg.iterrows():
    #     dclass = row["duration_class"]
    #     uids   = row["uids"]    
    #     print("IMERG Processing duration class:", row["duration_class"], len(uids))
    #     out    = f"queries/haversine_by_class/imerg_{dclass}.parquet"
    #     calc_distances(con, uids, table="imerg_tracking", file=out)
    #     print(dclass, len(uids), "linhas salvas em", out)



    # ### Processamento para bacias da HydroBASINS
    # # # Lê diretamente o shapefile da HydroBASINS (nível 0 - bacias principais)
    # url = "zip+https://data.hydrosheds.org/file/HydroBASINS/standard/hybas_sa_lev02_v1c.zip"
    # # # Leitura direta via fiona/pyogrio
    # gdf = gpd.read_file(url)
    # amazon = gdf[gdf["HYBAS_ID"] == 6020006540]
    # laplata = gdf[gdf["HYBAS_ID"] == 6020014330]

    # print("GSMAP Processing Amazon and La Plata basins...")
    # get_region_tracks(
    #     con,
    #     amazon,
    #     'gsmap_tracking',
    #     'queries/start_end_trajectory/gsmap_amazon.parquet'
    # )

    # print("GSMAP Processing La Plata basin...")
    # get_region_tracks(
    #     con,
    #     laplata,
    #     'gsmap_tracking',
    #     'queries/start_end_trajectory/gsmap_laplata.parquet'
    # )

    # print("IMERG Processing Amazon and La Plata basins...")
    # get_region_tracks(
    #     con,
    #     amazon,
    #     'imerg_tracking',
    #     'queries/start_end_trajectory/imerg_amazon.parquet'
    # )
    # print("IMERG Processing La Plata basin...")
    # get_region_tracks(
    #     con,
    #     laplata,
    #     'imerg_tracking',
    #     'queries/start_end_trajectory/imerg_laplata.parquet'
    # )
