import operator
from collections import Counter

import geopandas as gpd
import pandas as pd
import scipy

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from scipy.sparse import hstack

ORDINAL_ENCODER = OrdinalEncoder()
VESSEL_ENCODER = OneHotEncoder()
COUNTRY_ENCODER = OneHotEncoder()
CONTINENT_ENCODER = OneHotEncoder()


def _get_top_destination(vessel, tgdf, voyages):
    last_points = _get_last_points(vessel, tgdf)
    similar_points = _get_similar_points(last_points, tgdf)
    top_3_likely_destinations = _get_likely_destination(vessel, similar_points, voyages)

    top_port = int(top_3_likely_destinations.iloc[0].name)

    return top_port


def _get_last_points(vessel, trips):
    t = trips[trips['vessel'] == vessel]
    last_5_points = t[-5:]
    last_5_points.reset_index(drop=True, inplace=True)

    lp = gpd.GeoDataFrame(last_5_points, geometry=gpd.points_from_xy(last_5_points['long'], last_5_points['lat']))

    lp['buffer'] = lp.buffer(0.2)

    lp = lp.set_crs('epsg:4326')
    lp = lp.set_geometry('buffer')
    weight = 1.0

    for p in range(len(lp)):
        lp.at[p, 'point_weight'] = weight
        weight = weight * 1.5

    return lp


def _get_similar_points(last_points, trips):
    points_within_buffer = pd.DataFrame(columns=trips.columns)

    for _, point in last_points.iterrows():
        within_point = trips[trips.geometry.within(point.buffer)]
        within_point = within_point.dropna()
        within_point = within_point[within_point['heading'].between(point.heading - 22.5, point.heading + 22.5)]
        within_point['point_weight'] = last_points.at[_, 'point_weight']
        points_within_buffer = points_within_buffer.append(within_point)

    return points_within_buffer


def _get_likely_destination(vessel, similar_points, voyages):

    similar_points.reset_index(drop=True, inplace=True)
    vessel_start_port_id = voyages[voyages['vessel'] == vessel].iloc[-1].vessel

    for _, r in similar_points.iterrows():
        start_port_id = voyages.loc[int(r['voyage_id'])].begin_port_id
        end_port_id = voyages.loc[int(r['voyage_id'])].end_port_id
        similar_point_vessel = voyages.loc[int(r['voyage_id'])].vessel

        if (start_port_id == vessel_start_port_id) & (vessel == similar_point_vessel):
            trip_weight = 5.0
        elif (start_port_id == vessel_start_port_id) & (vessel != similar_point_vessel):
            trip_weight = 1.5
        elif (start_port_id != vessel_start_port_id) & (vessel == similar_point_vessel):
            trip_weight = 1.2
        else:
            trip_weight = 1

        similar_points.at[_, 'end_port_id'] = end_port_id
        if similar_points.at[_, 'end_port_id'] == similar_points.at[_, 'closest_port']:
            port_closeness_weight = 3.0
        else:
            port_closeness_weight = 1.0
        similar_points.at[_, 'weight'] = similar_points.at[_, 'point_weight'] + trip_weight + port_closeness_weight

    destinations = pd.DataFrame(similar_points.groupby(by='end_port_id').sum()['weight'])
    destinations['confidence'] = destinations['weight']/destinations['weight'].sum()

    top_3 = destinations.nlargest(3, 'confidence')

    return top_3


def _process_trips_df(trips, voyages):
    tgdf = _trips_to_voyages_class(trips, voyages)
    tgdf = gpd.GeoDataFrame(tgdf, geometry=gpd.points_from_xy(tgdf['long'], tgdf['lat']))
    tgdf = tgdf.set_crs('epsg:4326')

    return tgdf


def _trips_to_voyages_class(trips, voyages):
    for index in range(len(voyages)):
        start_date = voyages.at[index, 'begin_date']
        end_date = voyages.at[index, 'end_date']
        mask = (trips['vessel'] == voyages.at[index, 'vessel']) & (trips['datetime'] >= start_date) & (trips['datetime'] <= end_date)
        trips.at[mask, 'voyage_id'] = index

    return trips


def _get_first_prediction(vessel, tgdf, voyages):
    d = {}
    last_voyage = voyages[voyages['vessel'] == vessel].iloc[-1]
    prediction_vessel = vessel
    prediction_begin_port_id = last_voyage.end_port_id
    prediction_end_port_id = _get_top_destination(vessel, tgdf, voyages)
    voyage = 1
    d['vessel'] = vessel
    d['begin_port_id'] = prediction_begin_port_id
    d['end_port_id'] = prediction_end_port_id
    d['voyage'] = voyage

    return d


def first_prediction_all_vessels(trips, voyages):
    tgdf = _process_trips_df(trips, voyages)
    vessels = voyages['vessel'].unique()

    d_list = []

    for v in vessels:
        top_port = _get_first_prediction(v, tgdf, voyages)
        d_list.append(top_port)

    prediction_df = pd.DataFrame(d_list)

    return prediction_df


def standard_prediction(voyages, first_prediction):
    vessels = voyages['vessel'].unique()
    predictions = pd.DataFrame(columns=first_prediction.columns)

    for v in vessels:
        destinations = _predict_destinations(v, voyages, first_prediction)
        predictions = predictions.append(destinations)

    predictions = predictions.append(first_prediction)
    predictions.sort_values(by=['vessel', 'voyage'], inplace=True)

    return predictions


def _predict_destinations(vessel, voyages, first_prediction):
    d_list = []
    v = voyages[voyages['vessel'] == vessel]
    begin_port_id = first_prediction[first_prediction['vessel'] == vessel]['end_port_id'].item()

    for i in range(2, 4):
        d = {'vessel': None, 'begin_port_id': None, 'end_port_id': None, 'voyage': None}
        d['vessel'] = vessel
        d['begin_port_id'] = begin_port_id
        d['end_port_id'] = _likely_destination(voyages, vessel, begin_port_id)
        d['voyage'] = i
        d_list.append(d)
        begin_port_id = d['end_port_id']

    return d_list


def _likely_destination(voyages, vessel, port):
    possible_destinations = list(voyages[voyages['begin_port_id'] == port]['end_port_id'])
    observations = []
    for i in possible_destinations:
        n_items = len(voyages[(voyages['vessel'] == vessel) & (voyages['begin_port_id'] == port) & (voyages['end_port_id'] == i)])
        observations.extend([i] * n_items * 5)
    possible_destinations.extend(observations)

    destination_counts = Counter(possible_destinations)

    dest = max(destination_counts.items(), key=operator.itemgetter(1))[0]

    return dest

def rf_prediction(voyages, ports, first_prediction):
    processed_first_prediction = _process_first_prediction(first_prediction, voyages, ports)
    processed_first_prediction_copy = processed_first_prediction.copy()

    model = RandomForestClassifier(n_estimators=100)
    encoded_data, encoded_end_port = _encode_data(voyages)

    X_train, X_test, y_train, y_test = train_test_split(encoded_data,
                                                        encoded_end_port,
                                                        test_size=0.1)

    model.fit(X_train, y_train)

    second_port_data = _port_prediction_data(processed_first_prediction_copy)
    second_port_prediction = model.predict(second_port_data)
    second_prediction = _process_prediction(processed_first_prediction_copy, second_port_prediction, ports, 2)

    processed_second_prediction = _process_second_prediction(processed_first_prediction_copy, second_prediction, ports)

    third_port_data = _port_prediction_data(processed_second_prediction)
    third_port_prediction = model.predict(third_port_data)
    third_prediction = _process_prediction(processed_second_prediction, third_port_prediction, ports, 3)

    predictions = first_prediction.append([second_prediction, third_prediction])
    predictions = predictions[['vessel', 'begin_port_id', 'end_port_id', 'voyage']]
    predictions['begin_port_id'] = predictions['begin_port_id'].astype(int)
    predictions['end_port_id'] = predictions['end_port_id'].astype(int)
    predictions.sort_values(by=['vessel', 'voyage'], inplace=True)

    return predictions


def _process_prediction(first_df, prediction, ports, voyage_id):
    prediction_copy = prediction.copy()
    for i, p in enumerate(prediction):
        predicted_port = ports.at[p, 'port']

        first_df.at[i, 'second_stop'] = predicted_port

    processed_prediction = first_df[['vessel', 'end_port_id', 'second_stop', 'voyage']]
    processed_prediction = processed_prediction.rename(columns={'end_port_id': 'begin_port_id',
                                                              'second_stop': 'end_port_id'})
    processed_prediction['voyage'] = voyage_id

    return processed_prediction


def _port_prediction_data(prediction_df):
    v_id = VESSEL_ENCODER.transform(pd.DataFrame(prediction_df['vessel']))
    s_ports = ORDINAL_ENCODER.transform(pd.DataFrame(prediction_df['end_port_id']))
    p_continent = CONTINENT_ENCODER.transform(pd.DataFrame(prediction_df['continent']))
    p_country = COUNTRY_ENCODER.transform(pd.DataFrame(prediction_df['country']))
    s_doy = scipy.sparse.csr.csr_matrix(pd.DataFrame(prediction_df['doy']))

    pred_stack = hstack([v_id, s_ports, p_country, p_continent, s_doy])

    return pred_stack


def _second_port_prediction_data(first_prediction):
    v_id = VESSEL_ENCODER.transform(pd.DataFrame(first_prediction['vessel']))
    s_ports = ORDINAL_ENCODER.transform(pd.DataFrame(first_prediction['end_port_id']))
    p_continent = CONTINENT_ENCODER.transform(pd.DataFrame(first_prediction['continent']))
    p_country = COUNTRY_ENCODER.transform(pd.DataFrame(first_prediction['country']))
    s_doy = scipy.sparse.csr.csr_matrix(pd.DataFrame(first_prediction['doy']))

    pred_stack = hstack([v_id, s_ports, p_country, p_continent, s_doy])

    return pred_stack

def _process_second_prediction(first_prediction, second_prediction, ports):
    ports_dropped = ports.set_index('port')
    ports_dropped = ports_dropped[~ports_dropped.index.duplicated(keep='first')]

    for i, r in second_prediction.iterrows():
        second_prediction.at[i, 'country'] = ports_dropped.at[int(r['end_port_id']), 'countries']
        second_prediction.at[i, 'continent'] = ports_dropped.at[int(r['end_port_id']), 'continents']
        doy = (first_prediction.at[i, 'doy'] + 20) % 365

        second_prediction.at[i, 'doy'] = doy

    second_prediction['doy'] = second_prediction['doy'].astype(int)

    return second_prediction


def _process_first_prediction(first_prediction, voyages, ports):
    ports_dropped = ports.set_index('port')
    ports_dropped = ports_dropped[~ports_dropped.index.duplicated(keep='first')]
    first_prediction_copy = first_prediction.copy()
    for i, r in first_prediction_copy.iterrows():
        first_prediction_copy.at[i, 'country'] = ports_dropped.at[int(r['end_port_id']), 'countries']
        first_prediction_copy.at[i, 'continent'] = ports_dropped.at[int(r['end_port_id']), 'continents']
        lp = voyages[voyages['vessel'] == r['vessel']].iloc[-1]
        doy = lp['end_date'].dayofyear

        first_prediction_copy.at[i, 'doy'] = (doy + 20) % 365

    first_prediction_copy['doy'] = first_prediction_copy['doy'].astype(int)

    return first_prediction_copy


def _encode_data(voyages):
    b_p_id = ORDINAL_ENCODER.fit_transform(pd.DataFrame(voyages['begin_port_id']))
    e_p_id = ORDINAL_ENCODER.transform(pd.DataFrame(voyages['end_port_id']))

    v = VESSEL_ENCODER.fit_transform(pd.DataFrame(voyages['vessel']))
    s_country = COUNTRY_ENCODER.fit_transform(pd.DataFrame(voyages['start_country']))
    s_continent = CONTINENT_ENCODER.fit_transform(pd.DataFrame(voyages['start_continent']))
    s_doy = scipy.sparse.csr.csr_matrix(pd.DataFrame(voyages['start_doy']))

    encoded = hstack([v, b_p_id, s_country, s_continent, s_doy])

    return encoded, e_p_id


def _write_output(voyages, predictions, voyages_outfile, predictions_outfile):

    voyages = voyages[['vessel', 'begin_date', 'end_date', 'begin_port_id', 'end_port_id']]
    voyages.sort_values(by=['vessel', 'begin_date'])
    voyages['begin_date'] = voyages['begin_date'].dt.strftime('%m/%d/%Y')
    voyages['end_date'] = voyages['end_date'].dt.strftime('%m/%d/%Y')

    predictions.to_csv(predictions_outfile, index=False)
    voyages.to_csv(voyages_outfile, index=False)


def get_prediction_main(trips, voyages, ports, voyages_outfile, predictions_outfile, prediction='standard'):
    import datetime
    print(datetime.datetime.now(), 'getting_first_prediction...')
    first_prediction = first_prediction_all_vessels(trips, voyages)
    print(datetime.datetime.now(), 'all the other stuff...')
    if prediction == 'standard':
        predictions = standard_prediction(voyages, first_prediction)

    elif prediction == 'rf':
        predictions = rf_prediction(voyages, ports, first_prediction)

    _write_output(voyages, predictions, voyages_outfile, predictions_outfile)


TRIPS = pd.read_csv(r'/path/to/processed_trips.csv',
                    parse_dates=['datetime'],
                    infer_datetime_format=True,
                    dayfirst=True)  # /path/to/processed_trips.csv
VOYAGES = pd.read_csv(r'/path/to/processed_voyages.csv',
                      parse_dates=['begin_date', 'end_date'],
                      infer_datetime_format=True,
                      dayfirst=True)  # /path/to/processed_voyages.csv
PORTS = pd.read_csv(r'/path/to/processed_ports.csv')  # /path/to/processed_ports.csv

VOYAGES_OUTFILE = r'/path/to/final_output_voyages.csv'  # /path/to/final_output_voyages.csv
PREDICTION_OUTFILE = r'/path/to/final_output_prediction.csv'  # /path/to/final_output_prediction.csv

get_prediction_main(TRIPS, VOYAGES, PORTS, VOYAGES_OUTFILE, PREDICTION_OUTFILE, prediction='rf')
