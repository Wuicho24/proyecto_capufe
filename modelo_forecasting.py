import os
import shutil
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib

DATOS = 'aforos_capufe.csv'
MODELS_DIR = 'models'
TEST_RESULTS_DIR = 'test_results'  # <- agregado

MESES = {
    'ENERO': 1, 'FEBRERO': 2, 'MARZO': 3, 'ABRIL': 4, 'MAYO': 5, 'JUNIO': 6,
    'JULIO': 7, 'AGOSTO': 8, 'SEPTIEMBRE': 9, 'OCTUBRE': 10, 'NOVIEMBRE': 11, 'DICIEMBRE': 12
}

VEHICULOS = [
    'AUTOS', 'MOTOS', 'AUTOBUS DE 2 EJES', 'AUTOBUS DE 3 EJES', 'AUTOBUS DE 4 EJES',
    'CAMIONES DE 2 EJES', 'CAMIONES DE 3 EJES', 'CAMIONES DE 4 EJES', 'CAMIONES DE 5 EJES',
    'CAMIONES DE 6 EJES', 'CAMIONES DE 7 EJES', 'CAMIONES DE 8 EJES', 'CAMIONES DE 9 EJES',
    'TRICICLOS', 'EJE EXTRA AUTOBUS', 'EJE EXTRA CAMION', 'PEATONES'
]

VACACIONES = pd.to_datetime([
    '2022-01-01', '2022-02-05','2022-02-14', '2022-03-21', '2022-04-10', '2022-04-14', '2022-04-15',
    '2022-05-01', '2022-05-10', '2022-06-24', '2022-07-15', '2022-09-15', '2022-10-31',
    '2022-11-01', '2022-11-02', '2022-11-20', '2022-12-12', '2022-12-24', '2022-12-25', '2022-12-31'
])

def cargar_datos_aforos():
    df = pd.read_csv(DATOS, encoding='latin-1')

    for col in VEHICULOS:
        df[col] = df[col].astype(str).str.replace(',', '', regex=False)
        df[col] = df[col].replace('nan', np.nan)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['MES'] = df['MES'].str.strip().str.upper().map(MESES)
    df = df.dropna(subset=['MES', 'AÑO'])

    df['FECHA'] = pd.to_datetime(dict(year=df['AÑO'], month=df['MES'], day=1))
    df = df.groupby(['FECHA'])[VEHICULOS].sum().reset_index()
    df.sort_values('FECHA', inplace=True)

    return df

def crear_directorio_salida(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def entrenar_modelo_forecasting(df, col, test=False):
    print(f'Entrenando modelo para: {col}')
    ts = df[['FECHA', col]].rename(columns={'FECHA': 'ds', col: 'y'}).dropna()

    holidays = pd.DataFrame({
        'holiday': 'vacaciones',
        'ds': VACACIONES,
        'lower_window': 0,
        'upper_window': 1,
    })

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
        holidays=holidays
    )

    if test:
        size = int(len(ts) * 0.8)
        train_df = ts.iloc[:size]
        test_df = ts.iloc[size:]

        model.fit(train_df)
    else:
        model.fit(ts)

    return model

def predecir_valor(tipo_vehiculo: str, meses_adelante: int):
    modelo_path = os.path.join(MODELS_DIR, f'{tipo_vehiculo}.pkl')
    if not os.path.exists(modelo_path):
        raise ValueError(f"Modelo no encontrado para: {tipo_vehiculo}")
    model = joblib.load(modelo_path)

    future = model.make_future_dataframe(periods=meses_adelante, freq='MS')
    forecast = model.predict(future)

    prediccion = forecast.iloc[-1]

    return {
        "fecha": prediccion['ds'].strftime('%Y-%m'),
        "prediccion": round(prediccion['yhat'], 2),
        "inferior": round(prediccion['yhat_lower'], 2),
        "superior": round(prediccion['yhat_upper'], 2)
    }

def iniciar_entrenamiento_series(test=False):
    df = cargar_datos_aforos()
    if test:
        crear_directorio_salida(TEST_RESULTS_DIR)
    crear_directorio_salida(MODELS_DIR)

    for col in VEHICULOS:
        model = entrenar_modelo_forecasting(df, col, test)
        joblib.dump(model, os.path.join(MODELS_DIR, f'{col}.pkl'))

def totales_por_vehiculo_anual(anio: int, tipos: list):
    df = cargar_datos_aforos()
    df['AÑO'] = df['FECHA'].dt.year
    df_anio = df[df['AÑO'] == anio]

    tipos_validos = [t for t in tipos if t in VEHICULOS]
    totales = df_anio[tipos_validos].sum().to_dict()

    return {k: int(v) for k, v in totales.items()}

