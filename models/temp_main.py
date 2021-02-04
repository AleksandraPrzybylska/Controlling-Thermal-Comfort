import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn import metrics, ensemble
from sklearn.preprocessing import MinMaxScaler
import pickle
from pathlib import Path
import glob
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.linear_model import Ridge
import pandasgui

import warnings
warnings.filterwarnings("ignore")


def read_temp_mid_sn()-> int:
    with open('data/additional_info.json') as f:
        additional_data = json.load(f)

    devices = additional_data['offices']['office_1']['devices']
    sn_temp_mid = [d['serialNumber'] for d in devices if d['description'] == 'temperature_middle'][0]
    return sn_temp_mid


def modify_df(df1: pd.DataFrame, name):
    # Df
    df1.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
    df1.rename(columns={'value': name }, inplace=True)
    df1['time'] = pd.to_datetime(df1['time'])
    df1.drop(columns=['unit'], inplace=True)
    df1.set_index('time', inplace=True)

    return df1

def project_check_data():

    sn_temp_mid = read_temp_mid_sn()
    # TODO: First dataset
    df_temp_first = pd.read_csv('data/office_1_temperature_supply_points_data_2020-03-05_2020-03-19.csv')
    df_temp_first = modify_df(df_temp_first, 'temp').dropna()
    df_temp_first = df_temp_first[df_temp_first['serialNumber'] == sn_temp_mid]

    df_target_temp_first = pd.read_csv('data/office_1_targetTemperature_supply_points_data_2020-03-05_2020-03-19.csv')
    df_target_temp_first = modify_df(df_target_temp_first, 'target_temp').dropna()

    df_valve_first = pd.read_csv('data/office_1_valveLevel_supply_points_data_2020-03-05_2020-03-19.csv')
    df_valve_first = modify_df(df_valve_first, 'valve').dropna()


    # TODO: Second Dataset
    df_temp_second = pd.read_csv('data/office_1_temperature_supply_points_data_2020-10-13_2020-11-02.csv')
    df_temp_second = modify_df(df_temp_second, 'temp').dropna()
    df_temp_second = df_temp_second[df_temp_second['serialNumber'] == sn_temp_mid]

    df_target_temp_second = pd.read_csv('data/office_1_targetTemperature_supply_points_data_2020-10-13_2020-11-01.csv')
    df_target_temp_second = modify_df(df_target_temp_second, 'target_temp').dropna()

    df_valve_second = pd.read_csv('data/office_1_valveLevel_supply_points_data_2020-10-13_2020-11-01.csv')
    df_valve_second = modify_df(df_valve_second, 'valve').dropna()


    # TODO: CONCAT FIRST DATASET
    df_combined_first = pd.concat([df_temp_first, df_target_temp_first, df_valve_first])
    df_combined_first = df_combined_first.resample(pd.Timedelta(minutes=3), label='right').mean().fillna(method='ffill')
    df_combined_first['temp_last'] = df_combined_first['temp'].shift(1, fill_value=20)
    df_combined_first['temp_gt'] = df_combined_first['temp'].shift(-1, fill_value=20.34)
    df_combined_first['diff_temp'] = df_combined_first['target_temp'] - df_combined_first['temp']


    # TODO: CONCAT SECOND DATASET
    df_combined_second = pd.concat([df_temp_second, df_target_temp_second, df_valve_second])
    df_combined_second = df_combined_second.resample(pd.Timedelta(minutes=3), label='right').mean().fillna(method='ffill')
    df_combined_second['temp_last'] = df_combined_second['temp'].shift(1, fill_value=20)
    df_combined_second['temp_gt'] = df_combined_second['temp'].shift(-1, fill_value=20.34)
    df_combined_second['diff_temp'] = df_combined_second['target_temp'] - df_combined_second['temp']

    df_combined_first = df_combined_first[1:-1]
    df_combined_second = df_combined_second[1:-1]
    df_combined = pd.concat([df_combined_first, df_combined_second])

    mask = df_combined.index < '2020-10-29'
    df_train = df_combined.loc[mask]

    X_train = df_train[['temp', 'valve', 'target_temp', 'diff_temp']].to_numpy()
    y_train = df_train['temp_gt'].to_numpy()

    mask = (df_combined.index > '2020-10-29')
    df_test = df_combined[mask]

    X_test = df_test[['temp', 'valve', 'target_temp', 'diff_temp']].to_numpy()

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    temp_file = 'temp_model.p'
    pickle.dump(model, open(temp_file, 'wb'))
    y_predicted = model.predict(X_test)

    df_test['temp_predicted'] = y_predicted.tolist()
    y_test = df_test['temp_gt'].to_numpy()
    y_last = df_test['temp_last'].to_numpy()
    print(f'mae base: {metrics.mean_absolute_error(y_test, y_last)}')
    print(f'mae reg: {metrics.mean_absolute_error(y_test, y_predicted)}')

def main():
    random.seed(42)
    pd.options.display.max_columns = None
    project_check_data()


if __name__ == '__main__':
    main()
