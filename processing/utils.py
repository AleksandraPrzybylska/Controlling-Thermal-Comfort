from typing import Tuple
import pandas as pd
from pathlib import Path
import pickle
pd.options.mode.chained_assignment = None

def modify_df(df1: pd.DataFrame, name):
    df1.rename(columns={'value': name}, inplace=True)
    df1.drop(columns=['unit'], inplace=True)

    return df1

def preprocess_data_temp(
        df_temp: pd.DataFrame,
        df_target_temp: pd.DataFrame,
        df_valve: pd.DataFrame
) -> Tuple[float, float]:

    df_combined = pd.concat([df_temp, df_target_temp, df_valve])
    df_combined = df_combined.resample(pd.Timedelta(minutes=1), label='right').mean().fillna(method='ffill') # resample data in order to fulfill nan values
    df_combined['diff_temp'] = df_combined['target_temp'] - df_combined['temp']

    df_test = df_combined
    X_test = df_test[['temp', 'valve', 'target_temp', 'diff_temp']].to_numpy()
    last_resample = df_combined['temp'].to_numpy()[-1]
    return X_test[-1], last_resample

def preprocess_data_valve(
        df_temp: pd.DataFrame,
        df_target_temp: pd.DataFrame,
        df_valve: pd.DataFrame
) -> Tuple[float, float]:

    df_combined = pd.concat([df_temp, df_target_temp, df_valve])
    df_combined = df_combined.resample(pd.Timedelta(minutes=3), label='right').mean().fillna(method='ffill')
    df_combined['valve_last'] = df_combined['valve'].shift(1, fill_value=30)
    df_combined['diff_temp'] = df_combined['target_temp'] - df_combined['temp']
    df_test = df_combined[1:-1]
    X_test = df_test[['valve', 'temp', 'diff_temp', 'valve_last']].to_numpy()

    last_resample = df_combined['valve'].to_numpy()[-1]
    return X_test[-1], last_resample

def perform_processing(
        temperature: pd.DataFrame,
        target_temperature: pd.DataFrame,
        valve_level: pd.DataFrame,
        serial_number_for_prediction: str
) -> Tuple[float, float]:

    df_temp = modify_df(temperature, 'temp').dropna()
    df_temp = df_temp[df_temp['serialNumber'] == serial_number_for_prediction]

    df_target_temp = modify_df(target_temperature, 'target_temp').dropna()
    df_valve = modify_df(valve_level, 'valve').dropna()

    # TODO: TEMPERATURE PREDICTION
    X_test_temp, last_resample_temp = preprocess_data_temp(df_temp.copy(), df_target_temp.copy(), df_valve.copy())
    X_test_temp = X_test_temp.reshape(1, -1)

    with Path('./models/temp_model.p').open('rb') as temp_file:
        temp_model = pickle.load(temp_file)

    y_predicted_temp = temp_model.predict(X_test_temp)

    # TODO: VALVE PREDICTION
    X_test_valve, last_resample_valve = preprocess_data_valve(df_temp.copy(), df_target_temp.copy(), df_valve.copy())
    X_test_valve = X_test_valve.reshape(1, -1)

    with Path('./models/valve_model.p').open('rb') as valve_file:
        valve_model = pickle.load(valve_file)

    y_predicted_valve = valve_model.predict(X_test_valve)
    y_predicted_valve = (0.1*y_predicted_valve + 0.9*last_resample_valve)

    return y_predicted_temp, y_predicted_valve