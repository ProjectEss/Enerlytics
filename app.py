import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import holidays
import pickle 
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
import json 
import concurrent.futures 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import altair as alt

MODEL_FILE = "model_results.pkl" 
MULTI_TARGET_PKL = "multi_target_forecast.pkl" 
PARAMS_FILE = "lgbm_params.json" 
HISTORY_FILE = "prediction_history.csv" 
MODEL_VERSION = "v1.8.2_Stable"
SHORT_TERM_HORIZON_HOURS = 48 
LONG_TERM_HORIZON_MONTHS = 12
FUTURE_FORECAST_OUTPUT_FILE = "real_consumption.csv" 
MAX_FUTURE_YEAR = 2026 
EXOGENOUS_COLUMNS = [
    'Average temperature [°C]', 'Maximum temperature [°C]', 'Minimum temperature [°C]', 
    'Average relative humidity [%]', 'Wind speed [m/s]', 'Average wind direction [°]', 
    'Precipitation [mm]', 'Average air pressure [hPa]', 'eur_per_mwh'
]
DEFAULT_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'random_state': 42
}

def load_lgbm_params():
    if os.path.exists(PARAMS_FILE):
        try:
            with open(PARAMS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return DEFAULT_PARAMS
    return DEFAULT_PARAMS
    
def save_lgbm_params(params):
    try:
        with open(PARAMS_FILE, 'w') as f:
            json.dump(params, f, indent=4)
    except Exception:
        pass

def update_lgbm_params(key, value):
    st.session_state['lgbm_params'][key] = value
    save_lgbm_params(st.session_state['lgbm_params'])
    
@st.cache_data(show_spinner="Converting data to CSV...")
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data(show_spinner="Loading data...")
def load_data(uploaded_file):
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)
    return df

def reset_predictions():
    if os.path.exists(MULTI_TARGET_PKL):
        os.remove(MULTI_TARGET_PKL)
    if os.path.exists(MODEL_FILE):
        os.remove(MODEL_FILE)
    if 'model_run_summary' in st.session_state:
        del st.session_state['model_run_summary']
    st.success("All previous forecasts reset. Please re-run predictions.")

def log_prediction_run_metadata(summary):
    successful_results = {k: v for k, v in summary['results'].items() if v['status'] == 'Success'}
    if not successful_results:
        st.warning("No successful predictions to log to history.")
        return

    all_mape = [res['mape'] for res in successful_results.values()]
    all_r2 = [res['r2'] for res in successful_results.values()]
    
    new_entry = pd.DataFrame([{
        'Run_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Prediction_Mode': summary['prediction_mode'],
        'IDs_Predicted': len(successful_results),
        'Avg_MAPE': np.mean(all_mape),
        'Avg_R2': np.mean(all_r2),
    }])
    
    if os.path.exists(HISTORY_FILE):
        try:
            history_df = pd.read_csv(HISTORY_FILE)
            history_df = pd.concat([history_df, new_entry], ignore_index=True)
        except Exception:
            history_df = new_entry
    else:
        history_df = new_entry
        
    history_df.to_csv(HISTORY_FILE, index=False)

def auto_detect_time_column(df):
    keywords = ['time', 'date', 'timestamp', 'measured_at', 'datetime', 'ts']
    all_cols = df.columns.tolist()
    
    for col in all_cols:
        if any(keyword in col.lower() for keyword in keywords):
            return col
    
    for col in all_cols:
        try:
            sample_value = df[col].dropna().head(1).iloc[0]
            pd.to_datetime(sample_value, errors='raise')
            return col
        except:
            continue
            
    return all_cols[0] if all_cols else None

@st.cache_data(show_spinner="Filtering IDs by Month...")
def get_active_target_ids(df, time_col, selected_month_year):
    df = df.copy()
    try:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df['month_year'] = df[time_col].dt.strftime('%Y-%m')
    except Exception:
        return []

    df_filtered = df[df['month_year'] == selected_month_year].copy()
    if df_filtered.empty:
        return []

    potential_id_cols = [
        col for col in df.columns 
        if col != time_col and col not in EXOGENOUS_COLUMNS and col not in ['month_year']
    ]
    
    active_ids = []
    for col in potential_id_cols:
        try:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
            if df_filtered[col].sum() > 0:
                active_ids.append(col)
        except:
            continue

    return active_ids

def get_future_month_options(last_historical_timestamp):
    if last_historical_timestamp is None:
        return []

    last_date = pd.to_datetime(last_historical_timestamp).normalize()
    start_date = last_date.to_period('M').start_time + pd.DateOffset(months=1)
    end_date = datetime(MAX_FUTURE_YEAR, 1, 1) - pd.Timedelta(seconds=1)
    
    future_months = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    return [d.strftime('%Y-%m') for d in future_months]

def engineer_features(df, time_col, target_col):
    df = df.copy()
    try:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col).sort_index()
        df = df[~df.index.duplicated(keep='first')] 
    except Exception:
        return None
        
    if time_col in df.columns:
        df = df.drop(columns=[time_col], errors='ignore')

    df['Hour_of_Day'] = df.index.hour
    df['Day_of_Week'] = df.index.dayofweek
    df['Day_of_Year'] = df.index.dayofyear
    df['Month'] = df.index.month
    df['Year'] = df.index.year

    hour_of_day_float = df.index.hour.values
    df['Hour_sin'] = np.sin(2 * np.pi * hour_of_day_float / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * hour_of_day_float / 24)
    
    def categorize_time(hour):
        if 13 <= hour <= 16: return 'Peak_Volatile'
        elif 0 <= hour <= 5: return 'Night_Volatile'
        else: return 'Shoulder'
    df['Time_Category'] = df.index.hour.map(categorize_time).values
    df = pd.get_dummies(df, columns=['Time_Category'], prefix='TC', dtype=int)
    
    country = 'FI'
    try:
        years_to_check = df.index.year.unique().tolist()
        years_to_check.extend(range(df.index.year.max() + 1, MAX_FUTURE_YEAR + 1))
        years_to_check = sorted(list(set(years_to_check)))

        holidays_list = holidays.country_holidays(country, years=years_to_check)
        df['Is_Holiday'] = df.index.to_series().apply(lambda date: date in holidays_list).astype(int)
    except Exception:
        df['Is_Holiday'] = 0 

    df['Is_Weekend'] = (df.index.dayofweek >= 5).astype(int)
    
    if target_col in df.columns:
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(df[target_col].mean())
        
        df['Lag_1'] = df[target_col].shift(1)
        df['Lag_24'] = df[target_col].shift(24)
        df['Lag_48'] = df[target_col].shift(48)
        df['Lag_168'] = df[target_col].shift(168)
        
        df['Rolling_Mean_48H'] = df[target_col].rolling(window=48, min_periods=1).mean().shift(1)

    return df

@st.cache_data(show_spinner="Calculating historical seasonal averages...")
def calculate_seasonal_averages(processed_data):
    exog_data_cols = [c for c in EXOGENOUS_COLUMNS if c in processed_data.columns]
    exog_data = processed_data[exog_data_cols].copy()
    
    if exog_data.empty:
        return pd.DataFrame() 

    for col in exog_data_cols:
        exog_data[col] = pd.to_numeric(exog_data[col], errors='coerce')
        
    exog_data['Hour'] = exog_data.index.hour
    exog_data['Month'] = exog_data.index.month

    seasonal_averages = exog_data.groupby(['Month', 'Hour']).mean()
    
    return seasonal_averages

def get_future_features(timestamp, X_train_cols, last_known_exog, seasonal_averages, mode, country='FI'): 
    future_features = {}
    
    future_features['Hour_of_Day'] = timestamp.hour
    future_features['Day_of_Week'] = timestamp.dayofweek
    future_features['Day_of_Year'] = timestamp.dayofyear
    future_features['Month'] = timestamp.month
    future_features['Year'] = timestamp.year
    hour_of_day_float = timestamp.hour
    future_features['Hour_sin'] = np.sin(2 * np.pi * hour_of_day_float / 24)
    future_features['Hour_cos'] = np.cos(2 * np.pi * hour_of_day_float / 24)
    hour = timestamp.hour
    is_peak_volatile = 13 <= hour <= 16
    is_night_volatile = 0 <= hour <= 5
    future_features['TC_Peak_Volatile'] = int(is_peak_volatile)
    future_features['TC_Night_Volatile'] = int(is_night_volatile)
    future_features['TC_Shoulder'] = int(not is_peak_volatile and not is_night_volatile)
    try:
        is_holiday = timestamp in holidays.country_holidays(country, years=[timestamp.year])
        future_features['Is_Holiday'] = int(is_holiday)
    except Exception:
        future_features['Is_Holiday'] = 0
    future_features['Is_Weekend'] = int(timestamp.dayofweek >= 5)

    for col in EXOGENOUS_COLUMNS:
        if col in X_train_cols:
            if mode == '48h' or mode == 'short' or seasonal_averages.empty:
                future_features[col] = last_known_exog.get(col, np.nan)
            else: 
                try:
                    avg_val = seasonal_averages.loc[(timestamp.month, timestamp.hour), col]
                    future_features[col] = avg_val
                except KeyError:
                    future_features[col] = last_known_exog.get(col, np.nan)
             
    return future_features


def train_lightgbm_model(X, y, base_params):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42 
    )

    params = base_params.copy()
    
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    gbm = lgb.train(
        params, lgb_train, num_boost_round=1000, 
        valid_sets=lgb_eval,
        callbacks=[lgb.early_stopping(50, verbose=False)] 
    )

    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    epsilon = 1e-10 
    mape = np.mean(np.abs(y_test - y_pred) / np.maximum(np.abs(y_test), epsilon)) * 100
    
    return gbm, rmse, r2, mape, X_test.columns.tolist()

def perform_and_save_forecast(gbm, processed_data, X, y, feature_columns, target_column, rmse, r2, mape, raw_data, time_column, prediction_mode, selected_future_month_year, seasonal_averages, last_timestamp):
    try:
        time_diffs = processed_data.index.to_series().diff().dropna().dt.total_seconds().unique()
        valid_diffs = time_diffs[time_diffs > 0]
        freq_seconds = valid_diffs.min() if valid_diffs.size > 0 else 3600
        freq = f'{int(freq_seconds/3600)}H' if freq_seconds >= 3600 and (freq_seconds % 3600) == 0 else f'{int(freq_seconds)}S'
    except Exception:
        freq = 'H'
        freq_seconds = 3600
        
    if prediction_mode == '48h':
        END_DATE = last_timestamp + pd.Timedelta(hours=48)
        forecast_description = "48-Hour Short-Term Forecast (Fixed Horizon)"
    elif prediction_mode == 'short':
        future_month_start = pd.to_datetime(selected_future_month_year)
        END_DATE = future_month_start + pd.DateOffset(months=1) - pd.Timedelta(seconds=freq_seconds) 
        forecast_description = f"Short-Term Forecast (up to {selected_future_month_year})"
    else: 
        END_DATE = last_timestamp + relativedelta(years=1) - pd.Timedelta(seconds=freq_seconds)
        forecast_description = f"Long-Term Forecast (12 months up to {END_DATE.strftime('%Y-%m')})"
    
    full_range = pd.date_range(start=last_timestamp, end=END_DATE, freq=freq)
    forecast_timestamps = full_range[full_range > last_timestamp] 
    FORECAST_HORIZON = len(forecast_timestamps)
    
    last_known_exog = X.iloc[-1].to_dict() 
    
    target_column_name = target_column
    current_data = processed_data[[target_column_name] + [c for c in feature_columns if c in processed_data.columns]].tail(200) 
    forecast_results = []
    static_confidence = np.clip(100 - mape, 0, 100)

    for i in range(FORECAST_HORIZON):
        next_timestamp = forecast_timestamps[i]
        
        future_features_dict = get_future_features(next_timestamp, feature_columns, last_known_exog, seasonal_averages, prediction_mode) 
        next_row = pd.Series(future_features_dict, index=feature_columns)
        
        if prediction_mode in ['48h', 'short'] or i < SHORT_TERM_HORIZON_HOURS:
            if 'Lag_1' in feature_columns:
                next_row['Lag_1'] = current_data[target_column_name].iloc[-1] 
            if 'Lag_24' in feature_columns:
                next_row['Lag_24'] = current_data[target_column_name].iloc[-24] if len(current_data) >= 24 else y.mean()
            if 'Lag_168' in feature_columns:
                 next_row['Lag_168'] = current_data[target_column_name].iloc[-168] if len(current_data) >= 168 else y.mean()
                 
            if 'Rolling_Mean_48H' in feature_columns:
                 if len(current_data) >= 48:
                     next_row['Rolling_Mean_48H'] = current_data[target_column_name].iloc[-48:].mean()
                 else:
                     next_row['Rolling_Mean_48H'] = y.mean() 
        else:
            if 'Lag_1' in feature_columns:
                next_row['Lag_1'] = np.nan
            if 'Lag_24' in feature_columns:
                next_row['Lag_24'] = np.nan
            if 'Lag_168' in feature_columns:
                 next_row['Lag_168'] = current_data[target_column_name].iloc[-168] if len(current_data) >= 168 else y.mean()
            if 'Rolling_Mean_48H' in feature_columns:
                 next_row['Rolling_Mean_48H'] = np.nan
                 

        input_df = pd.DataFrame([next_row.fillna(0)]).set_index(pd.Index([next_timestamp]))
        input_df = input_df[feature_columns] 
        predicted_value = gbm.predict(input_df)[0]
        
        result_row = {
            'Timestamp': next_timestamp,
            'Predicted Consumption': predicted_value,
            'Confidence_Level': static_confidence,
            'Time_Step_Index': i
        }
        forecast_results.append(result_row)
        
        new_row_df = pd.DataFrame({target_column_name: [predicted_value]}, index=[next_timestamp])
        current_data = pd.concat([current_data, new_row_df], axis=0).tail(200)
        current_data = current_data[[target_column_name]]

    if os.path.exists(MULTI_TARGET_PKL):
        try:
            with open(MULTI_TARGET_PKL, 'rb') as f:
                multi_target_forecast_existing = pickle.load(f)
        except:
            multi_target_forecast_existing = None
    else:
        multi_target_forecast_existing = None

    historical_id_cols = [col for col in raw_data.columns if col not in [time_column] + EXOGENOUS_COLUMNS]
    new_forecast_index = forecast_timestamps
    
    if multi_target_forecast_existing is not None:
        all_indices = new_forecast_index.union(multi_target_forecast_existing.index)
        all_ids = list(set(historical_id_cols).union(multi_target_forecast_existing.columns))
        
        multi_target_forecast = pd.DataFrame(index=all_indices, columns=all_ids, dtype=float)
        multi_target_forecast.update(multi_target_forecast_existing)
    else:
        multi_target_forecast = pd.DataFrame(index=new_forecast_index, columns=historical_id_cols, dtype=float)

    multi_target_forecast.index.name = 'measured_at'
    
    future_predictions_df = pd.DataFrame(forecast_results).set_index('Timestamp')
    new_prediction_series = future_predictions_df['Predicted Consumption']
    new_prediction_series.index.name = 'measured_at'
    
    multi_target_forecast.loc[new_prediction_series.index, target_column] = new_prediction_series.values


    with open(MULTI_TARGET_PKL, 'wb') as f:
        pickle.dump(multi_target_forecast, f)
        
    return {
        'id': target_column, 
        'rmse': rmse, 
        'r2': r2, 
        'mape': mape, 
        'confidence': static_confidence,
        'feature_importance': pd.Series(gbm.feature_importance(), index=X.columns).sort_values(ascending=False).head(5),
        'description': forecast_description
    }

def run_single_id_forecast(target_id, raw_data, time_column, prediction_mode, selected_future_month_year, lgbm_params, seasonal_averages, feature_columns_selected):
    processed_data = engineer_features(raw_data, time_column, target_id)
    if processed_data is None:
        return {'id': target_id, 'status': 'Error: Engineering failed. Check data types.'}
        
    available_features = processed_data.select_dtypes(include=np.number).columns.tolist()
    feature_columns = [c for c in feature_columns_selected if c in available_features and c != target_id]
    
    if prediction_mode == 'long':
        feature_columns = [f for f in feature_columns if f not in ['Lag_1', 'Lag_24', 'Rolling_Mean_48H']]
    
    X = processed_data[feature_columns]
    y = processed_data[target_id]
    X = X.fillna(X.mean()) 
    
    if len(X) < 100:
         return {'id': target_id, 'status': 'Error: Insufficient data (need >100 rows)'}

    last_timestamp = processed_data.index[-1]
    last_timestamp_fixed = last_timestamp.tz_localize(None) if last_timestamp.tz is not None else last_timestamp

    try:
        gbm, rmse, r2, mape, final_features = train_lightgbm_model(X, y, lgbm_params)
    except Exception as e:
        return {'id': target_id, 'status': f'Error: Training failed ({e})'}

    try:
        results = perform_and_save_forecast(
            gbm, processed_data, X, y, final_features, target_id, 
            rmse, r2, mape, raw_data, time_column, prediction_mode, 
            selected_future_month_year, seasonal_averages, last_timestamp_fixed
        )
        results['status'] = 'Success'
        return results
    except Exception as e:
        return {'id': target_id, 'status': f'Error: Forecasting failed ({e})'}


st.set_page_config(
    page_title="Advanced Energy Load Predictor", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;} 
    </style>
    """,
    unsafe_allow_html=True,
)

if 'lgbm_params' not in st.session_state:
    st.session_state['lgbm_params'] = load_lgbm_params() 

if 'time_column_selected' not in st.session_state:
    st.session_state['time_column_selected'] = None
if 'target_columns_selected' not in st.session_state:
    st.session_state['target_columns_selected'] = []
if 'month_year_selected' not in st.session_state:
    st.session_state['month_year_selected'] = None 
if 'future_month_selected' not in st.session_state:
    st.session_state['future_month_selected'] = None
if 'prediction_mode' not in st.session_state: 
    st.session_state['prediction_mode'] = 'short' 
if 'feature_columns_selected' not in st.session_state:
    st.session_state['feature_columns_selected'] = []
if 'uploaded_file_name' not in st.session_state: 
    st.session_state['uploaded_file_name'] = None
if 'model_run_summary' not in st.session_state:
    st.session_state['model_run_summary'] = {} 
if 'uploaded_file_object' not in st.session_state:
    st.session_state['uploaded_file_object'] = None
if 'historical_df' not in st.session_state:
    st.session_state['historical_df'] = None
if 'selected_future_month_year' not in st.session_state:
    st.session_state['selected_future_month_year'] = None

lgbm_params = st.session_state['lgbm_params']

st.sidebar.header("⚙️ Model Configuration")
lgbm_params = {
    'objective': st.sidebar.selectbox('Objective', ['regression'], index=0, key='obj', on_change=lambda: update_lgbm_params('objective', st.session_state.obj)),
    'metric': st.sidebar.selectbox('Metric', ['rmse', 'mae'], 
                                   index=['rmse', 'mae'].index(st.session_state['lgbm_params']['metric']), 
                                   key='met', on_change=lambda: update_lgbm_params('metric', st.session_state.met)),
    'boosting_type': st.sidebar.selectbox('Boosting Type', ['gbdt', 'dart', 'goss'], 
                                          index=['gbdt', 'dart', 'goss'].index(st.session_state['lgbm_params']['boosting_type']), 
                                          key='bt', on_change=lambda: update_lgbm_params('boosting_type', st.session_state.bt)),
    'learning_rate': st.sidebar.slider('Learning Rate', 0.01, 0.3, st.session_state['lgbm_params']['learning_rate'], 0.01, 
                                       key='lr', on_change=lambda: update_lgbm_params('learning_rate', st.session_state.lr)),
    'num_leaves': st.sidebar.slider('Number of Leaves', 20, 50, st.session_state['lgbm_params']['num_leaves'], 
                                    key='nl', on_change=lambda: update_lgbm_params('num_leaves', st.session_state.nl)),
    'max_depth': st.sidebar.slider('Max Depth', -1, 15, st.session_state['lgbm_params']['max_depth'], 
                                   key='md', on_change=lambda: update_lgbm_params('max_depth', st.session_state.md)),
    'min_child_samples': st.sidebar.slider('Min Child Samples', 1, 30, st.session_state['lgbm_params']['min_child_samples'], 
                                           key='mcs', on_change=lambda: update_lgbm_params('min_child_samples', st.session_state.mcs)),
    'random_state': 42
}

st.title("⚡ Advanced Energy Load Predictor")
st.markdown(f"**Current Model Version: {MODEL_VERSION}**")
st.markdown("---")

tab_upload, tab_config = st.tabs(["1. Data Upload 📥", "2. Prediction Setup ⚙️"])

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload your energy data file (CSV format):", 
        type=['csv']
    )
    st.session_state['uploaded_file_object'] = uploaded_file 

    if uploaded_file is not None:
        
        is_new_file = st.session_state['uploaded_file_name'] != uploaded_file.name
        
        if is_new_file and st.session_state['uploaded_file_name'] is not None:
            st.warning(f"New file **{uploaded_file.name}** detected. Resetting previous multi-target forecasts.")
            reset_predictions()
            st.session_state['time_column_selected'] = None
            st.session_state['target_columns_selected'] = []
            
        st.session_state['uploaded_file_name'] = uploaded_file.name
        
        data = load_data(uploaded_file)
        if data is not None:
            st.success(f"File **{uploaded_file.name}** uploaded successfully! Showing first 5 rows of raw data:")
            st.dataframe(data.head())
            
            default_time_col = auto_detect_time_column(data) 
            
            all_cols = data.columns.tolist()
            if st.session_state['time_column_selected'] in all_cols:
                time_index = all_cols.index(st.session_state['time_column_selected'])
            elif default_time_col in all_cols:
                time_index = all_cols.index(default_time_col)
            else:
                time_index = 0
            
            time_column = st.selectbox(
                "Select the **Timestamp/Date** Column:", 
                all_cols,
                index=time_index, 
                key='time_column_selected',
                help="This selection is saved."
            )
            
            if time_column:
                try:
                    historical_df = data.drop(columns=EXOGENOUS_COLUMNS, errors='ignore').copy() 
                    historical_df = historical_df.rename(columns={time_column: 'measured_at'})
                    historical_df['measured_at'] = pd.to_datetime(historical_df['measured_at'], errors='coerce')
                    historical_df = historical_df.set_index('measured_at').sort_index()
                    historical_df = historical_df[~historical_df.index.duplicated(keep='first')] 
                    st.session_state['historical_df'] = historical_df
                except Exception as e:
                    st.error(f"Error preparing historical data: {e}")
                    st.stop()


with tab_config:
    if st.session_state['uploaded_file_object'] is None or 'time_column_selected' not in st.session_state or st.session_state['time_column_selected'] is None:
        st.warning("Please complete steps 1 to proceed.")
        if st.session_state['uploaded_file_object'] is None:
             st.stop()

    data = load_data(st.session_state['uploaded_file_object'])
    time_column = st.session_state['time_column_selected']
    
    if data is None or time_column is None:
        st.stop()
    
    st.header("Prediction Mode")
    prediction_mode = st.radio(
        "Choose Prediction Horizon:", 
        ['48h', 'short', 'long'], 
        format_func=lambda x: {'48h': "Fixed 48 Hours", 'short': "Short-Term (Up to 1 Month)", 'long': "Long-Term (1 Year)"}[x],
        key='prediction_mode',
        horizontal=True,
        help="Fixed 48 Hours: Quickest result for immediate load planning. Short-Term: Highest accuracy for immediate future (up to 1 month). Long-Term: Forecasts 1 year using seasonal averages."
    )

    st.header("Select Target IDs (Batch Enabled)")
    try:
        data[time_column] = pd.to_datetime(data[time_column], errors='coerce')
        historical_month_year_options = sorted(data[time_column].dt.strftime('%Y-%m').dropna().unique().tolist(), reverse=True)
        last_historical_timestamp = data[time_column].max()
        
        selected_future_month_year = None
        selected_historical_month_year = None
        
        if historical_month_year_options:
            latest_month = historical_month_year_options[0]

            if prediction_mode == '48h':
                selected_historical_month_year = latest_month
                st.info(f"Fixed 48 Hours mode automatically filters active IDs based on the **latest month of data: {latest_month}**.")
            
            else: 
                if st.session_state['month_year_selected'] not in historical_month_year_options:
                    st.session_state['month_year_selected'] = latest_month

                selected_historical_month_year = st.selectbox(
                    "Filter Active IDs by (Historical Month YYYY-MM):", 
                    historical_month_year_options,
                    index=historical_month_year_options.index(st.session_state['month_year_selected']),
                    key='month_year_selected_filter', 
                    help="Only IDs with consumption > 0 in this month are shown below."
                )
                
                if prediction_mode == 'short':
                    future_month_options = get_future_month_options(last_historical_timestamp)
                    if not future_month_options:
                        st.error("No future months available for short-term prediction.")
                        st.stop()
                    
                    if st.session_state['future_month_selected'] not in future_month_options:
                        st.session_state['future_month_selected'] = future_month_options[0]
                    
                    future_month_index = future_month_options.index(st.session_state['future_month_selected'])
                    
                    selected_future_month_year = st.selectbox(
                        "Short-Term Prediction up to the end of (YYYY-MM):", 
                        future_month_options,
                        index=future_month_index,
                        key='future_month_selected',
                        help="The forecast runs up to the last hour of this month."
                    )
                else: 
                    st.info(f"Long-Term mode predicts for **{LONG_TERM_HORIZON_MONTHS} full months** (1 year) starting from the end of historical data.")

            
            active_id_cols = get_active_target_ids(data, time_column, selected_historical_month_year)
            target_cols_options = active_id_cols
        else:
            target_cols_options = [col for col in data.columns if col not in [time_column] + EXOGENOUS_COLUMNS]
            st.warning("Could not parse dates for month filtering. Showing all consumption IDs.")

    except Exception as e:
        st.error(f"Error during month filtering: {e}")
        target_cols_options = [col for col in data.columns if col not in [time_column] + EXOGENOUS_COLUMNS]

    all_target_options = ["All IDs"] + target_cols_options
    
    initial_selection_raw = st.session_state['target_columns_selected'] if st.session_state['target_columns_selected'] else []
    
    if all(item in target_cols_options for item in initial_selection_raw) and len(initial_selection_raw) == len(target_cols_options):
         default_selection = ["All IDs"] + target_cols_options
    else:
         default_selection = initial_selection_raw
         
    target_columns_raw = st.multiselect(
        "Select **Target IDs** (Consumption Columns) to Predict:", 
        all_target_options,
        default=default_selection, 
        key='target_columns_selected_raw',
        help="Select 'All IDs' to include every active meter in the prediction batch."
    )
    
    if "All IDs" in target_columns_raw:
        target_columns = target_cols_options
    else:
        target_columns = [col for col in target_columns_raw if col != "All IDs"]
        
    st.session_state['target_columns_selected'] = target_columns 
    st.session_state['selected_future_month_year'] = selected_future_month_year
    
    if not target_columns:
        st.warning("Please select at least one Target ID.")
        st.stop()


    st.header("Feature Selection")
    
    test_id = target_columns[0]
    processed_data_test = engineer_features(data, time_column, test_id)
    
    if processed_data_test is None:
        st.error("Feature engineering failed on the test ID. Check data quality.")
        st.stop()

    available_features = processed_data_test.select_dtypes(include=np.number).columns.tolist()
    available_features = [col for col in available_features if col not in target_columns]
    
    default_features_list = [
        'Lag_1', 'Lag_24', 'Lag_48', 'Lag_168', 'Rolling_Mean_48H', 'Hour_sin', 'Hour_cos', 'TC_Peak_Volatile', 'TC_Night_Volatile', 
        'Day_of_Week', 'Month', 'Is_Holiday', 'Is_Weekend'
    ]
    present_defaults = [f for f in default_features_list if f in available_features]
    present_defaults.extend([c for c in EXOGENOUS_COLUMNS if c in available_features and c not in present_defaults])
    redundant_features = ['Hour_of_Day', 'Day_of_Year', 'Year', 'TC_Shoulder']
    present_defaults = [f for f in present_defaults if f not in redundant_features]
    
    if not st.session_state['feature_columns_selected']:
        st.session_state['feature_columns_selected'] = [f for f in available_features if f in present_defaults]

    feature_columns = st.multiselect(
        "Select **Input Features** (Includes Engineered & Exogenous Data):", 
        available_features,
        default=st.session_state['feature_columns_selected'],
        key='feature_columns_selected'
    )
    
    st.info(f"LT Mode will automatically **drop Lag_1, Lag_24, and Rolling_Mean_48H** from this set to prevent error accumulation and use seasonal averages for **{EXOGENOUS_COLUMNS}**.")