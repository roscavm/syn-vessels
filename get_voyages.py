from math import radians, cos, sin, asin, sqrt
import operator

import geopandas as gpd
import numpy as np
import pandas as pd


def process_trips(trips_df: pd.DataFrame, ports_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

    # gpd defaults column types to str so recast...
    trips_df['datetime'] = pd.to_datetime(trips_df['datetime'])
    trips_df['lat'] = trips_df['lat'].astype('float')
    trips_df['long'] = trips_df['long'].astype('float')

    trips_df = gpd.GeoDataFrame(trips_df, geometry=gpd.points_from_xy(trips_df['long'], trips_df['lat']))

    trips_df.sort_values(by=['vessel', 'datetime'], inplace=True)
    trips_df.drop_duplicates(subset=['vessel', 'datetime'], keep='first', inplace=True)

    # set a crs so that we can do spatial analysis
    trips_df = trips_df.set_crs('EPSG:4326')

    trips_df.reset_index(drop=True, inplace=True)

    trips_df = _calculate_distance_travelled(trips_df)

    trips_df = _calculate_speed(trips_df)

    _get_closest_port(trips_df, ports_df)

    return trips_df


def _calculate_distance_travelled(df):
    df['distance_travelled'] = np.nan
    for i in range(1, len(df)):
        df['distance_travelled'][i] = _haversine(df['long'][i],
                                                 df['lat'][i],
                                                 df['long'][i-1],
                                                 df['lat'][i-1])

    return df


def _haversine(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles

    return c * r


def _calculate_speed(df):
    df['calculated_speed'] = np.nan
    for i in range(1, len(df)):
        df['calculated_speed'][i] = _calc_speed_in_knots(df['datetime'][i],
                                                         df['datetime'][i-1],
                                                         df['distance_travelled'][i])

    df = df[df['calculated_speed'] < 40]

    return df


def _calc_speed_in_knots(date_1, date_0, distance):
    t_delta = (date_1 - date_0).seconds / 3600
    speed = (distance/t_delta) / 1.852

    return speed


def _get_closest_port(trips_df, ports_df):
    r = 6371

    coords_list = [(r['lat'], r['long']) for _, r in ports_df.iterrows()]

    coords_arr = np.deg2rad(coords_list)
    trip_lat_long_df = trips_df[['lat', 'long']]
    trip_lat_long_df['lat'] = trip_lat_long_df['lat'].astype('float')
    trip_lat_long_df['long'] = trip_lat_long_df['long'].astype('float')
    a = np.deg2rad(trip_lat_long_df.values)

    lat = coords_arr[:, 0] - a[:, 0, None]
    lng = coords_arr[:, 1] - a[:, 1, None]

    add0 = np.cos(a[:, 0, None]) * np.cos(coords_arr[:, 0]) * np.sin(lng * 0.5) ** 2
    d = np.sin(lat * 0.5) ** 2 + add0

    h = 2 * r * np.arcsin(np.sqrt(d))
    trips_df['min_distance_to_a_port'] = h.min(1)
    closest_port = np.array([(ports_df['port'][x]) for x in h.argmin(1)], dtype='int64')
    trips_df['closest_port'] = closest_port


def process_ports(ports_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Preprocess the ports dataframe to calculate the geometry and apply a buffer.

    Parameters
    ----------
    ports_df : gpd.GeoDataFrame
        Ports geodataframe from the csv.

    Returns
    -------
    ports_df : gpd.GeoDataFrame
        Processesed geodataframe.

    """
    # drop any duplicates if they exist
    ports_df = _move_ports_to_correct_location(ports_df)

    # set a crs so that we can buffer
    ports_df = ports_df.set_crs('EPSG:4326')

    # gpd defaults column types to str so recast...
    ports_df['lat'] = ports_df['lat'].astype('float')
    ports_df['long'] = ports_df['long'].astype('float')

    ports_df = _calc_geometry(ports_df, 'long', 'lat')

    ports_df = ports_df.set_geometry('geometry')

    ports_df = _categorize_ports(ports_df, by='continents')
    ports_df = _categorize_ports(ports_df, by='countries')

    return ports_df


def _move_ports_to_correct_location(ports_df):
    # Some ports are not correctly georeferenced
    # This function moves the ports to a more correct location
    # The 'correct' coordinates are based on a quick cluster analysis of positions where boats
    # had a speed of 0 (0.01 deg clusters)

    # in weird cases ports have multiple docks which are too far apart from each other
    # create a new port near the old port

    length = len(ports_df)

    new_ports = [['16', 51.7000, -5.0000, None],
                 ['154', 22.589232, 114.429928, None],
                 ['152', 35.0000, 136.7000, None],
                 ['149', 38.7500, 117.7200, None],
                 ['104', -23.7600, 151.1800, None],
                 ['100', -12.5200, 130.8500, None],
                 ['37', 31.44455, 30.236817, None],
                 ['13', 30.1062, 122.28077, None],
                 ['59', 22.691807, 59.454105, None],
                 ]

    for idx, i in enumerate(new_ports):
        ports_df.loc[length + idx] = i

    correct_locations = {'177': (-13.783282, 123.3176),
                         '173': (-5.975215, 106.7993),
                         '168': (2.2500, 102.1000),
                         '162': (34.3600, 133.8400),
                         '152': (34.9700, 136.6600),
                         '150': (36.7700, 137.1200),
                         '148': (24.2500, 120.5000),
                         '146': (37.8500, 140.9600),
                         '143': (35.0200, 138.5000),
                         '142': (30.6000, 122.1000),
                         '141': (38.2700, 141.0400),
                         '139': (34.57357807773945, 135.40732121427064),
                         '134': (-32.7700, -71.5100),
                         '130': (33.2800, 131.7100),
                         '126': (24.7719130006179, 67.2976400439874),
                         '123': (34.5000, 133.7400),
                         '121': (12.6500, 101.1500),
                         '117': (55.6600, 21.1400),
                         '115': (1.2329294018336692, 103.67538404437637),  # SINGAPORE
                         '114': (25.0300, 55.0700),
                         '113': (38.7400, 26.9000),
                         '111': (34.7600, 134.6900),
                         '109': (35.4000, 139.6300),  # TOKYO PORTS
                         '105': (17.9700, -66.7600),
                         '101': (25.1600, 52.8900),
                         '98': (38.9600, 121.8900),
                         '96': (45.0900, 12.5800),
                         '88': (21.4500, 109.5400),
                         '82': (38.8200, 26.9100),
                         '76': (21.9000, 113.2200),
                         '74': (19.7800, 109.1500),
                         '71': (33.9200, 130.8600),
                         '69': (-2.4300, 133.1300),
                         '67': (53.937683954680786, 14.281227936981045),
                         '63': (34.548685378047715, 135.40387864669765),
                         '62': (37.1700, 129.3500),
                         '60': (32.0700, 121.7800),
                         '58': (37.0000, 126.7800),
                         '56': (37.9600, 23.4000),
                         '54': (1.331114376450962, 104.17410077125085),  # PENGERANG (SIN STRAIT)
                         '53': (-3.5300, -38.8000),
                         '51': (35.4700, 139.7400),  # TOKYO PORTS
                         '45': (29.0700, 48.1600),
                         '44': (40.99218627304147, 27.986182732598913),
                         '42': (35.460000, 139.720000),  # TOKYO PORTS
                         '38': (37.3400, 126.5800),
                         '37': (31.3700, 30.3000),
                         '33': (51.4300, 0.7000),
                         '23': (21.682679972411158, 72.50972498360328),
                         '18': (32.5500, 121.4300),
                         }

    for k, v in correct_locations.items():
        pindex = ports_df[ports_df['port'] == k].index[0]
        ports_df.at[pindex, 'lat'] = v[0]
        ports_df.at[pindex, 'long'] = v[1]

    return ports_df


def _calc_geometry(df, long, lat):
    df['geometry'] = gpd.points_from_xy(df[long], df[lat])

    return df


def _categorize_ports(ports, by='continents'):
    """
    Categorize the port based on its distance to the closest country/continent.

    Parameters
    ----------
    ports : gpd.GeoDataFrame
        Geodataframe of ports.

    Returns
    -------
    ports : gpd.GeoDataFrame
        The ports dataframe, with an added 'continent' column.

    """
    if by == 'continents':
        locations = _get_continents()
    elif by == 'countries':
        locations = _get_countries()

    ports[by] = ''
    for _, r in ports.iterrows():
        d = {c: 0 for c in list(locations.index)}
        for k in d.keys():
            d[k] = locations.loc[k].geometry.distance(r.geometry)

        ports[by][_] = min(d.items(), key=operator.itemgetter(1))[0]

    return ports


def _get_continents():
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world = world[['continent', 'geometry']]
    continents = world.dissolve(by='continent')
    continents = continents.drop('Seven seas (open ocean)')

    return continents


def _get_countries():
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world = world[['name', 'geometry']]
    countries = world.dissolve(by='name')

    return countries


def find_voyages(trips_df, voyage_df):
    dep_arr = _get_stops(trips_df)
    dep_arr.reset_index(drop=True, inplace=True)
    v_list = []
    for i, r in dep_arr.iterrows():
        if r['departure'] == True:
            v = {'vessel': None,
                 'begin_date': None,
                 'end_date': None,
                 'begin_port_id': None,
                 'end_port_id': None}
            v['vessel'] = r['vessel']
            v['begin_date'] = r['datetime']
            try:
                v['end_date'] = dep_arr['datetime'][i+1]
            except KeyError:
                v['end_date'] = None
            v['begin_port_id'] = r['closest_port']
            try:
                v['end_port_id'] = dep_arr['closest_port'][i+1]
            except KeyError:
                v['end_port_id'] = None

            v_list.append(v)

    voyage_df = voyage_df.append(v_list, ignore_index=True)

    voyage_df = voyage_df.dropna()

    voyage_df['vessel'] = voyage_df['vessel'].astype(int)
    voyage_df['begin_port_id'] = voyage_df['begin_port_id'].astype(int)
    voyage_df['end_port_id'] = voyage_df['end_port_id'].astype(int)

    return voyage_df


def _get_stops(df):
    stops = df[(df['min_distance_to_a_port'] < 2) & (df['calculated_speed'] < 5)]
    ixdiff = stops.index.to_series().diff
    arr = stops[~(ixdiff(1).eq(1))]
    dep = stops[~(ixdiff(-1).eq(-1))]

    for i in arr.index:
        stops.at[i, 'arrival'] = True
    for i in dep.index:
        stops.at[i, 'departure'] = True
    dep_arr = stops[(stops['departure'] == True) | (stops['arrival'] == True)]

    return dep_arr


def process_voyages(voyages, ports):
    voyages = voyages.dropna()
    voyages['begin_port_id'] = voyages['begin_port_id'].astype(int)
    voyages['end_port_id'] = voyages['end_port_id'].astype(int)
    voyages['distance_travelled'] = 0.0
    voyages['voyage_duration_days'] = 0.0

    _calc_voyage_stats(voyages, ports)

    voyages = voyages[voyages['distance_travelled'] > 0]
    voyages = voyages[['vessel', 'begin_date', 'end_date', 'begin_port_id', 'end_port_id']]

    ports = ports.set_index('port')
    ports = ports[~ports.index.duplicated(keep='first')]

    for i, r in voyages.iterrows():
        voyages.at[i, 'start_country'] = ports.at[str(r['begin_port_id']), 'countries']
        voyages.at[i, 'end_country'] = ports.at[str(r['end_port_id']), 'countries']

    for i, r in voyages.iterrows():
        voyages.at[i, 'start_continent'] = ports.at[str(r['begin_port_id']), 'continents']
        voyages.at[i, 'end_continent'] = ports.at[str(r['end_port_id']), 'continents']

    voyages['start_doy'] = voyages['begin_date'].dt.dayofyear
    voyages['end_doy'] = voyages['end_date'].dt.dayofyear

    return voyages

def _get_voyage_draft(voyages, trips):

    vessels = voyages['vessel'].unique()
    vessels_dict = {}

    for v in vessels:
        t = trips[trips['vessel'] == v]
        mean_draft = np.nanmean(t['draft'])
        vessels_dict[v] = mean_draft

    for index in range(len(voyages)):
        start_date = voyages.at[index, 'begin_date']
        end_date = voyages.at[index, 'end_date']
        vessel = voyages.at[index, 'vessel']
        mask = (trips['vessel'] == voyages.at[index, 'vessel']) & (trips['datetime'] >= start_date) & (trips['datetime'] <= end_date)
        masked_trip = trips.loc[mask]
        mean_trip_draft = np.nanmean(masked_trip['draft'])
        draft_difference = mean_trip_draft - vessels_dict[vessel]
        voyages.at[index, 'draft_difference'] = draft_difference

    return voyages


def _calc_voyage_stats(voyages, ports):
    voyages.reset_index(drop=True, inplace=True)

    for i in range(len(voyages)):
        begin_lat = float((ports[ports['port'] == str(voyages['begin_port_id'][i])]['lat']).values[0])
        begin_long = float((ports[ports['port'] == str(voyages['begin_port_id'][i])]['long']).values[0])
        end_lat = float((ports[ports['port'] == str(voyages['end_port_id'][i])]['lat']).values[0])
        end_long = float((ports[ports['port'] == str(voyages['end_port_id'][i])]['long']).values[0])
        voyages['distance_travelled'][i] = _haversine(begin_long, begin_lat, end_long, end_lat)
        voyages['voyage_duration_days'][i] = (voyages['end_date'][i] - voyages['begin_date'][i]).days


def get_voyages_main(tracking_csv, ports_csv, ports_outfile, trips_outfile, voyages_outfile):
    tracking = pd.read_csv(tracking_csv)
    ports = gpd.read_file(ports_csv)

    ports = process_ports(ports)
    voyages = pd.DataFrame(columns=['vessel', 'begin_date', 'end_date', 'begin_port_id', 'end_port_id'])

    list_vessels = tracking['vessel'].unique()

    trips = pd.DataFrame(columns=tracking.columns)

    for v in list_vessels:
        v_df = tracking[tracking['vessel'] == v]
        v_df = process_trips(v_df, ports)
        voyages = find_voyages(v_df, voyages)
        trips = trips.append(v_df)

    voyages = process_voyages(voyages, ports)

    ports.to_csv(ports_outfile)
    trips.to_csv(trips_outfile, index=False)
    voyages.to_csv(voyages_outfile, index=False)


TRACKINGCSV = r'/path/to/tracking.csv'  # /path/to/tracking.csv
PORTSCSV = r'/path/to/ports.csv'  # /path/to/ports.csv

PORTS_OUTFILE = r'/path/to/processed_ports.csv'  # /path/to/processed_ports.csv
TRIPS_OUTFILE = r'/path/to/processed_trips.csv'  # /path/to/processed_trips.csv
VOYAGES_OUTFILE = r'/path/to/processed_voyages.csv'  # /path/to/processed_voyages.csv

get_voyages_main(TRACKINGCSV, PORTSCSV, PORTS_OUTFILE, TRIPS_OUTFILE, VOYAGES_OUTFILE)
