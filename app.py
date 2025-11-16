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

# Define filenames and constants
MODEL_FILE = "model_results.pkl" 
MULTI_TARGET_PKL = "multi_target_forecast.pkl" 
PARAMS_FILE = "lgbm_params.json" 
MODEL_VERSION = "v1.7.7_UI_Enhanced" 
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

# --- Utility Functions (Persistence) ---

def load_lgbm_params():
    """Loads parameters from JSON file or returns defaults."""
    if os.path.exists(PARAMS_FILE):
        try:
            with open(PARAMS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return DEFAULT_PARAMS
    return DEFAULT_PARAMS
    
def save_lgbm_params(params):
    """Saves parameters to JSON file."""
    try:
        with open(PARAMS_FILE, 'w') as f:
            json.dump(params, f, indent=4)
    except Exception:
        pass

def update_lgbm_params(key, value):
    """Updates session state and persists to file."""
    st.session_state['lgbm_params'][key] = value
    save_lgbm_params(st.session_state['lgbm_params'])
    
def convert_df_to_csv(df):
    """Converts a DataFrame to CSV format for download."""
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data(show_spinner="Loading data...")
def load_data(uploaded_file):
    """
    Loads data from the uploaded file.
    CRITICAL FIX: Resets the file pointer using .seek(0) before reading.
    """
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)
    return df

def reset_predictions():
    """Deletes the multi-target persistence file and the results summary."""
    if os.path.exists(MULTI_TARGET_PKL):
        os.remove(MULTI_TARGET_PKL)
    if os.path.exists(MODEL_FILE):
        os.remove(MODEL_FILE)
    if 'model_run_summary' in st.session_state:
        del st.session_state['model_run_summary']
    st.success("All previous forecasts reset. Please re-run predictions.")

# --- Utility Functions (Data Prep) ---

def auto_detect_time_column(df):
    """Automatically detects the time column based on keywords or data type."""
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
    """
    Identifies ID columns that have a non-zero sum of consumption for the 
    specified month (YYYY-MM).
    """
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
    """Generates a list of YYYY-MM strings starting from the month after the last historical date."""
    if last_historical_timestamp is None:
        return []

    last_date = pd.to_datetime(last_historical_timestamp).normalize()
    start_date = last_date.to_period('M').start_time + pd.DateOffset(months=1)
    end_date = datetime(MAX_FUTURE_YEAR, 1, 1) - pd.Timedelta(seconds=1)
    
    future_months = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    return [d.strftime('%Y-%m') for d in future_months]


# --- Configuration ---
# START OF UI ENHANCEMENTS
st.set_page_config(
    page_title="Advanced Energy Load Predictor", # Changed the name/title
    page_icon="⚡", # Changed the favicon
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for aesthetics and to hide default Streamlit elements
st.markdown(
    """
    <style>
    /* Hide the Streamlit footer menu button */
    #MainMenu {visibility: hidden;}
    /* Hide the Streamlit "Made with Streamlit" footer */
    footer {visibility: hidden;}
    /* Hide the default header (which often duplicates the title) */
    .stApp > header {visibility: hidden;} 
    </style>
    """,
    unsafe_allow_html=True,
)
# END OF UI ENHANCEMENTS

# --- Session State Initialization (CRITICAL STEP) ---
# This block ensures all necessary session state keys exist before any widgets access them.

if 'lgbm_params' not in st.session_state:
    st.session_state['lgbm_params'] = load_lgbm_params() 
lgbm_params = st.session_state['lgbm_params']

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

# --- Core Feature Engineering and Data Preparation ---

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
        # Ensure the target column is numeric
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(df[target_col].mean())
        
        # Lag features
        df['Lag_1'] = df[target_col].shift(1)
        df['Lag_24'] = df[target_col].shift(24)
        df['Lag_48'] = df[target_col].shift(48)
        df['Lag_168'] = df[target_col].shift(168)
        
        # NEW: Rolling Mean 48 Hours
        df['Rolling_Mean_48H'] = df[target_col].rolling(window=48, min_periods=1).mean().shift(1)

    return df

@st.cache_data(show_spinner="Calculating historical seasonal averages...")
def calculate_seasonal_averages(processed_data):
    """
    Calculates the average value for each hour of the day and month of the year
    for all exogenous columns. 
    """
    
    # 1. Select exogenous data columns
    exog_data_cols = [c for c in EXOGENOUS_COLUMNS if c in processed_data.columns]
    exog_data = processed_data[exog_data_cols].copy()
    
    if exog_data.empty:
        return pd.DataFrame() 

    # 2. CRITICAL FIX: Explicitly convert all exogenous columns to numeric
    for col in exog_data_cols:
        # Coerce to numeric, turning non-parseable strings (like 'N/A') into NaN
        exog_data[col] = pd.to_numeric(exog_data[col], errors='coerce')
        
    # 3. Create grouping keys
    exog_data['Hour'] = exog_data.index.hour
    exog_data['Month'] = exog_data.index.month

    # 4. Calculate mean. NaNs introduced in step 2 are correctly ignored.
    seasonal_averages = exog_data.groupby(['Month', 'Hour']).mean()
    
    return seasonal_averages

def get_future_features(timestamp, X_train_cols, last_known_exog, seasonal_averages, mode, country='FI'): 
    """Generates features for a single future timestamp."""
    
    future_features = {}
    
    # Base Time and Cyclic Features (Always Calculated)
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

    # Exogenous Feature Management based on Mode
    for col in EXOGENOUS_COLUMNS:
        if col in X_train_cols:
            # For 48h and short mode, use the last known exogenous value (assume it persists)
            if mode == '48h' or mode == 'short' or seasonal_averages.empty:
                future_features[col] = last_known_exog.get(col, np.nan)
            else: 
                try:
                    # Access seasonal averages by Month and Hour index for long-term
                    avg_val = seasonal_averages.loc[(timestamp.month, timestamp.hour), col]
                    future_features[col] = avg_val
                except KeyError:
                    # Fallback if specific Month/Hour is missing in seasonal data
                    future_features[col] = last_known_exog.get(col, np.nan)
             
    return future_features


# --- Core Training and Prediction Logic ---

def train_lightgbm_model(X, y, base_params):
    """Trains a single LightGBM model and returns metrics."""
    
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
    """Performs the iterative forecast and saves results to the persistence file."""
    
    # 1. Determine Frequency
    try:
        time_diffs = processed_data.index.to_series().diff().dropna().dt.total_seconds().unique()
        valid_diffs = time_diffs[time_diffs > 0]
        freq_seconds = valid_diffs.min() if valid_diffs.size > 0 else 3600
        freq = f'{int(freq_seconds/3600)}H' if freq_seconds >= 3600 and (freq_seconds % 3600) == 0 else f'{int(freq_seconds)}S'
    except Exception:
        freq = 'H'
        freq_seconds = 3600

    # 2. Determine END_DATE based on Prediction Mode
    if prediction_mode == '48h':
        END_DATE = last_timestamp + pd.Timedelta(hours=48)
        forecast_description = "48-Hour Short-Term Forecast (Fixed Horizon)"
    elif prediction_mode == 'short':
        future_month_start = pd.to_datetime(selected_future_month_year)
        # Prediction goes up to the end of the selected month
        END_DATE = future_month_start + pd.DateOffset(months=1) - pd.Timedelta(seconds=freq_seconds) 
        forecast_description = f"Short-Term Forecast (up to {selected_future_month_year})"
    else: # 'long'
        END_DATE = last_timestamp + relativedelta(years=1) - pd.Timedelta(seconds=freq_seconds)
        forecast_description = f"Long-Term Forecast (12 months up to {END_DATE.strftime('%Y-%m')})"
    
    full_range = pd.date_range(start=last_timestamp, end=END_DATE, freq=freq)
    forecast_timestamps = full_range[full_range > last_timestamp] 
    FORECAST_HORIZON = len(forecast_timestamps)
    
    last_known_exog = X.iloc[-1].to_dict() 
    
    target_column_name = target_column
    # Maintain the last 200 hours of historical data + forecast (to support lags/rolling window)
    current_data = processed_data[[target_column_name] + [c for c in feature_columns if c in processed_data.columns]].tail(200) 
    forecast_results = []
    static_confidence = np.clip(100 - mape, 0, 100)

    for i in range(FORECAST_HORIZON):
        next_timestamp = forecast_timestamps[i]
        
        future_features_dict = get_future_features(next_timestamp, feature_columns, last_known_exog, seasonal_averages, prediction_mode) 
        next_row = pd.Series(future_features_dict, index=feature_columns)
        
        # --- Lag Feature Management based on Mode ---
        # 48h and early short-term rely on the immediately preceding forecast/actual
        if prediction_mode in ['48h', 'short'] or i < SHORT_TERM_HORIZON_HOURS:
            if 'Lag_1' in feature_columns:
                next_row['Lag_1'] = current_data[target_column_name].iloc[-1] 
            if 'Lag_24' in feature_columns:
                next_row['Lag_24'] = current_data[target_column_name].iloc[-24] if len(current_data) >= 24 else y.mean()
            if 'Lag_168' in feature_columns:
                 next_row['Lag_168'] = current_data[target_column_name].iloc[-168] if len(current_data) >= 168 else y.mean()
                 
            # Rolling Mean Management
            if 'Rolling_Mean_48H' in feature_columns:
                 if len(current_data) >= 48:
                     next_row['Rolling_Mean_48H'] = current_data[target_column_name].iloc[-48:].mean()
                 else:
                     next_row['Rolling_Mean_48H'] = y.mean() # Fallback to historical mean
        else:
            # For later long-term predictions, reliance on recent lags/rolling mean is removed
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
        
        # Add the new prediction to current_data to update the rolling window for the next iteration
        new_row_df = pd.DataFrame({target_column_name: [predicted_value]}, index=[next_timestamp])
        current_data = pd.concat([current_data, new_row_df], axis=0).tail(200)
        current_data = current_data[[target_column_name]]


    # --- 3. Persistence Update (Multi-Target) ---

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
        # Preserve existing forecasts, but new IDs/timestamps will be NaN
        multi_target_forecast.update(multi_target_forecast_existing)
    else:
        # Initialize with only historical IDs and the new forecast index
        multi_target_forecast = pd.DataFrame(index=new_forecast_index, columns=historical_id_cols, dtype=float)

    multi_target_forecast.index.name = 'measured_at'
    
    future_predictions_df = pd.DataFrame(forecast_results).set_index('Timestamp')
    new_prediction_series = future_predictions_df['Predicted Consumption']
    new_prediction_series.index.name = 'measured_at'
    
    # Update the forecast column with the new predictions
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
    """Worker function to train and predict a single ID, designed for concurrent execution."""
    
    processed_data = engineer_features(raw_data, time_column, target_id)
    if processed_data is None:
        return {'id': target_id, 'status': 'Error: Engineering failed. Check data types.'}

    available_features = processed_data.select_dtypes(include=np.number).columns.tolist()
    
    # Use the passed argument instead of st.session_state
    feature_columns = [c for c in feature_columns_selected if c in available_features and c != target_id]
    
    if prediction_mode == 'long':
        # Remove volatile short-term lags for long-term prediction
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


def display_model_results(summary):
    """Displays metrics, download buttons, and the download section for individual IDs."""
    st.header("4. Trained Model Results (Batch Summary)")
    
    if not summary or not summary.get('results'):
        st.info("No prediction results available. Run the forecast first.")
        return

    # Check if 'prediction_mode' is available, otherwise assume a fresh run is needed (prevents display error)
    prediction_mode = summary.get('prediction_mode', 'unknown')
    st.subheader(f"Forecast Run in **{prediction_mode.upper()}** Mode")
    
    # --- Metrics Table ---
    st.markdown("##### Prediction Metrics (Test Set)")
    
    metrics_data = []
    for _id, res in summary['results'].items():
        if res['status'] == 'Success':
            metrics_data.append({
                'ID': _id,
                'Status': res['status'],
                'RMSE': f"{res['rmse']:,.4f}",
                'R²': f"{res['r2']:.4f}",
                'MAPE': f"{res['mape']:,.2f}%",
                'Confidence': f"{res['confidence']:,.2f}%"
            })
        else:
            metrics_data.append({'ID': _id, 'Status': res['status'], 'RMSE': '-', 'R²': '-', 'MAPE': '-', 'Confidence': '-'})
            
    metrics_df = pd.DataFrame(metrics_data).set_index('ID')
    st.dataframe(metrics_df, use_container_width=True)

    # --- Feature Importance (Top ID) ---
    successful_results = {k: v for k, v in summary['results'].items() if v['status'] == 'Success'}
    
    if successful_results:
        # Determine the top ID based on Confidence
        top_id = max(successful_results.keys(), key=lambda k: successful_results[k]['confidence'])
        
        if top_id and successful_results[top_id]['status'] == 'Success':
            st.subheader(f"Top 5 Feature Importance for Best ID: {top_id}")
            st.bar_chart(successful_results[top_id]['feature_importance'])

    # --- Download Files (Combined & Individual) ---
    st.markdown("---")
    st.subheader(f"Download Forecast Files")
    
    if os.path.exists(MULTI_TARGET_PKL):
        try:
            with open(MULTI_TARGET_PKL, 'rb') as f:
                multi_target_forecast = pickle.load(f)
        except Exception:
            st.error("Error loading multi-target persistence file. Please run the prediction again.")
            return

        historical_df = summary['historical_df']
        
        # 1. Combination Logic for 'Combined' Download (Historical + Forecast)
        all_indices = historical_df.index.union(multi_target_forecast.index)
        # Filter for only the ID columns that exist in the forecast or historical data
        id_cols_in_forecast = [col for col in multi_target_forecast.columns if col in historical_df.columns]
        all_id_cols = id_cols_in_forecast
        
        base_df = historical_df[all_id_cols].reindex(all_indices) 
        base_df.update(multi_target_forecast)
        final_output_df = base_df.reset_index().rename(columns={'index': 'measured_at'})
        final_output_df['measured_at'] = final_output_df['measured_at'].dt.strftime('%Y-%m-%dT%H:%M:%S')

        # --- Combined Download Button ---
        col_combined, col_sep = st.columns(2)
        
        with col_combined:
            st.markdown("##### 1. All IDs (Historical + Forecast - Wide Format)")
            final_csv = convert_df_to_csv(final_output_df)
            st.download_button(
                label="⬇️ Download **real_consumption.csv** (All IDs)",
                data=final_csv,
                file_name=FUTURE_FORECAST_OUTPUT_FILE, 
                mime='text/csv',
                key='download_all_forecasts_key', 
                help=f"This file contains the full historical data followed by the future forecast for ALL IDs predicted so far (Wide Format)."
            )
        
        with col_sep:
            if st.button("Reset All Forecasts and Models"):
                reset_predictions()
                st.rerun()

        # --- Section 2: Individual ID Download ---
        st.markdown("---")
        st.markdown("##### 2. Single ID (Forecast ONLY - Two Columns)")
        
        all_ids_predicted = [k for k, v in summary['results'].items() if v['status'] == 'Success']
        
        if all_ids_predicted:
            col_id_select, col_id_download = st.columns(2)
            
            with col_id_select:
                selected_id_to_download = st.selectbox(
                    "Select ID for individual download (Forecast ONLY):", 
                    all_ids_predicted,
                    key='individual_download_id'
                )
            
            # 1. Filter the multi-target forecast file by the selected ID column
            forecast_only_df = multi_target_forecast[[selected_id_to_download]].copy()
            
            # 2. Drop rows where the value is NaN. This isolates only the predicted future timestamps.
            forecast_only_df = forecast_only_df.dropna(subset=[selected_id_to_download])
            
            # FIX: Strictly limit to 48 hours if the prediction mode was '48h'
            if prediction_mode == '48h':
                 forecast_only_df = forecast_only_df.head(SHORT_TERM_HORIZON_HOURS)
                 
            # 3. Final formatting
            forecast_only_df = forecast_only_df.reset_index().rename(columns={'measured_at': 'measured_at', selected_id_to_download: 'Predicted Consumption'})
            forecast_only_df['measured_at'] = forecast_only_df['measured_at'].dt.strftime('%Y-%m-%dT%H:%M:%S')

            with col_id_download:
                st.markdown("")
                id_csv = convert_df_to_csv(forecast_only_df)
                st.download_button(
                    label=f"⬇️ Download **{selected_id_to_download}_forecast_ONLY.csv**",
                    data=id_csv,
                    file_name=f"{selected_id_to_download}_forecast_ONLY.csv", 
                    mime='text/csv',
                    key=f'download_id_{selected_id_to_download}_key', 
                    help=f"Contains ONLY the future predicted data for ID: {selected_id_to_download} (2 Columns)."
                )
        else:
             st.info("No successful predictions to download individually.")

        # --- Section 3: All IDs Vertical Download ---
        st.markdown("---")
        st.markdown("##### 3. All Predicted IDs (Forecast ONLY - Vertical Format)")

        if all_ids_predicted:
            
            # 1. Filter multi-target forecast for only successful IDs and drop historical NaNs
            forecast_data_wide = multi_target_forecast[all_ids_predicted].copy().dropna(how='all')

            # 2. Enforce 48-hour limit if the mode was '48h'
            if prediction_mode == '48h':
                 forecast_data_wide = forecast_data_wide.head(SHORT_TERM_HORIZON_HOURS)

            # 3. Convert to long (vertical) format using stack
            forecast_data_long = forecast_data_wide.stack().reset_index()
            
            # 4. Rename columns
            forecast_data_long.columns = ['measured_at', 'ID', 'Predicted Consumption']
            
            # 5. Format timestamp
            forecast_data_long['measured_at'] = forecast_data_long['measured_at'].dt.strftime('%Y-%m-%dT%H:%M:%S')

            # 6. Generate CSV and download button
            vertical_csv = convert_df_to_csv(forecast_data_long)

            st.download_button(
                label="⬇️ Download **all_ids_forecast_vertical.csv**",
                data=vertical_csv,
                file_name="all_ids_forecast_vertical.csv", 
                mime='text/csv',
                key='download_all_ids_vertical_key', 
                help="Contains ONLY the predicted future data for ALL successful IDs in a vertical (long) format (3 Columns)."
            )
        else:
            st.info("No successful predictions available for the vertical download.")
            
        # --- Section 4: All IDs Wide Download (NEW) ---
        st.markdown("---")
        st.markdown("##### 4. All Predicted IDs (Forecast ONLY - Wide Format)")

        if all_ids_predicted:
            
            # 1. Filter multi-target forecast for only successful IDs and drop historical NaNs
            # This isolates the future timestamps where at least one ID has a prediction.
            forecast_data_wide = multi_target_forecast[all_ids_predicted].copy().dropna(how='all')

            # 2. Enforce 48-hour limit if the prediction mode was '48h'
            if prediction_mode == '48h':
                 # Keep only the first 48 timestamps from the start of the prediction
                 forecast_data_wide = forecast_data_wide.head(SHORT_TERM_HORIZON_HOURS)

            # 3. Final formatting: Reset index and format timestamp
            forecast_data_final_wide = forecast_data_wide.reset_index().rename(columns={'measured_at': 'measured_at'})
            forecast_data_final_wide['measured_at'] = forecast_data_final_wide['measured_at'].dt.strftime('%Y-%m-%dT%H:%M:%S')

            # 4. Generate CSV and download button
            wide_forecast_csv = convert_df_to_csv(forecast_data_final_wide)

            st.download_button(
                label="⬇️ Download **all_ids_forecast_wide.csv**",
                data=wide_forecast_csv,
                file_name="all_ids_forecast_wide.csv", 
                mime='text/csv',
                key='download_all_ids_wide_key', 
                help="Contains ONLY the predicted future data for ALL successful IDs in a wide format (Timestamps as rows, IDs as columns)."
            )
        else:
            st.info("No successful predictions available for the wide forecast download.")


    st.success("Results loaded successfully!")


# --- Main Streamlit App Logic (V1.7.7) ---

st.title("⚡ Advanced Energy Load Predictor") # Updated the main title
st.markdown(f"**Current Model Version: {MODEL_VERSION}**")
st.markdown("---")

# --- Sidebar for Hyperparameters ---
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


# --- Data Upload ---
tab_upload, tab_config, tab_predict = st.tabs(["1. Data Upload 📥", "2. Prediction Setup ⚙️", "3. Train & Results 📈"])

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload your energy data file (CSV format):", 
        type=['csv']
    )

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


# --- Prediction Setup ---
with tab_config:
    if uploaded_file is None or 'time_column_selected' not in st.session_state or st.session_state['time_column_selected'] is None:
        st.warning("Please complete steps 1 to proceed.")
        if uploaded_file is None:
             st.stop()

    data = load_data(uploaded_file)
    time_column = st.session_state['time_column_selected']
    
    if data is None or time_column is None:
        st.stop()
    
    # --- Prediction Mode Selector ---
    st.header("Prediction Mode")
    prediction_mode = st.radio(
        "Choose Prediction Horizon:", 
        ['48h', 'short', 'long'], 
        format_func=lambda x: {'48h': "Fixed 48 Hours", 'short': "Short-Term (Up to 1 Month)", 'long': "Long-Term (1 Year)"}[x],
        key='prediction_mode',
        horizontal=True,
        help="Fixed 48 Hours: Quickest result for immediate load planning. Short-Term: Highest accuracy for immediate future (up to 1 month). Long-Term: Forecasts 1 year using seasonal averages."
    )

    # --- Historical Month Filter (Determines Active IDs) ---
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
                # Auto-select latest month and inform user
                selected_historical_month_year = latest_month
                st.info(f"Fixed 48 Hours mode automatically filters active IDs based on the **latest month of data: {latest_month}**.")
            
            else: # 'short' or 'long' mode requires user interaction
                # Show historical month filter for 'short' and 'long'
                if st.session_state['month_year_selected'] not in historical_month_year_options:
                    st.session_state['month_year_selected'] = latest_month

                selected_historical_month_year = st.selectbox(
                    "Filter Active IDs by (Historical Month YYYY-MM):", 
                    historical_month_year_options,
                    index=historical_month_year_options.index(st.session_state['month_year_selected']),
                    key='month_year_selected_filter', 
                    help="Only IDs with consumption > 0 in this month are shown below."
                )
                
                # Show future month selector only for 'short' mode
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
                else: # 'long' mode
                    st.info(f"Long-Term mode predicts for **{LONG_TERM_HORIZON_MONTHS} full months** (1 year) starting from the end of historical data.")

            
            active_id_cols = get_active_target_ids(data, time_column, selected_historical_month_year)
            target_cols_options = active_id_cols
        else:
            target_cols_options = [col for col in data.columns if col not in [time_column] + EXOGENOUS_COLUMNS]
            st.warning("Could not parse dates for month filtering. Showing all consumption IDs.")

    except Exception as e:
        st.error(f"Error during month filtering: {e}")
        target_cols_options = [col for col in data.columns if col not in [time_column] + EXOGENOUS_COLUMNS]

    # --- ID Selection with "All IDs" option ---
    all_target_options = ["All IDs"] + target_cols_options
    
    # Determine which IDs were previously selected, or default to all if 'All IDs' was previously selected.
    # CRITICAL: Preserve the selection based on the raw list
    initial_selection_raw = st.session_state['target_columns_selected'] if st.session_state['target_columns_selected'] else []
    
    # If 'All IDs' was previously selected, display the full list to the user for clarity
    if all(item in target_cols_options for item in initial_selection_raw) and len(initial_selection_raw) == len(target_cols_options):
         default_selection = ["All IDs"] + target_cols_options
    else:
         default_selection = initial_selection_raw
         
    # Handle the multiselect widget:
    target_columns_raw = st.multiselect(
        "Select **Target IDs** (Consumption Columns) to Predict:", 
        all_target_options,
        default=default_selection, 
        key='target_columns_selected_raw',
        help="Select 'All IDs' to include every active meter in the prediction batch."
    )
    
    # Process the selection: If 'All IDs' is present in the raw selection, use the full list.
    if "All IDs" in target_columns_raw:
        target_columns = target_cols_options
    else:
        # Filter out "All IDs" if somehow it got carried over and is not the intended selection
        target_columns = [col for col in target_columns_raw if col != "All IDs"]
        
    st.session_state['target_columns_selected'] = target_columns # Update session state with the final list (without 'All IDs')
    
    if not target_columns:
        st.warning("Please select at least one Target ID.")
        st.stop()


    # --- Feature Selection ---
    st.header("Feature Selection")
    
    test_id = target_columns[0]
    processed_data_test = engineer_features(data, time_column, test_id)
    
    if processed_data_test is None:
        st.error("Feature engineering failed on the test ID. Check data quality.")
        st.stop()

    available_features = processed_data_test.select_dtypes(include=np.number).columns.tolist()
    available_features = [col for col in available_features if col not in target_columns]
    
    # Updated default features list with Rolling_Mean_48H
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


# --- Training and Results ---
with tab_predict:
    
    if uploaded_file is None or not target_columns or not feature_columns:
        st.warning("Please complete steps 1 and 2 (Upload, Select IDs/Mode/Features) to run the prediction.")
        st.stop()

    if st.session_state.get('model_run_summary') and st.session_state['model_run_summary']['prediction_mode'] == prediction_mode:
        st.success("Previous results found for the current mode. Displaying summary.")
        display_model_results(st.session_state['model_run_summary'])
    elif st.session_state.get('model_run_summary'):
        st.info(f"Previous results are for **{st.session_state['model_run_summary']['prediction_mode'].upper()}** mode. Run the prediction for **{prediction_mode.upper()}** mode.")

        
    st.header("3. Run Prediction")
    
    if st.button("🚀 Start Batch Train & Predict"):
        
        # Load the data again before processing to ensure all functions operate on the clean buffer
        data = load_data(uploaded_file)
        
        # Use the first target column for feature engineering test
        test_id = target_columns[0]
        processed_data_for_seasonal_calc = engineer_features(data, time_column, test_id)
        
        if processed_data_for_seasonal_calc is None:
             st.error("Cannot proceed: Seasonal calculation failed during initial processing.")
             st.stop()
             
        seasonal_averages = calculate_seasonal_averages(processed_data_for_seasonal_calc)
        
        st.session_state['model_run_summary'] = {'prediction_mode': prediction_mode, 'results': {}, 'historical_df': st.session_state['historical_df']}
        
        st.info(f"Starting batch prediction for **{len(target_columns)} IDs** in **{prediction_mode.upper()}** mode using concurrent processing.")
        
        MAX_WORKERS = min(5, len(target_columns)) 
        final_results = {}
        
        # Retrieve feature list from session state *outside* the concurrent executor
        current_feature_columns_selected = st.session_state['feature_columns_selected']

        with st.spinner(f"Training and forecasting {len(target_columns)} IDs concurrently (Max workers: {MAX_WORKERS})..."):
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                
                future_to_id = {
                    executor.submit(
                        run_single_id_forecast, 
                        _id, 
                        data, # Pass the original data object
                        time_column, 
                        prediction_mode, 
                        selected_future_month_year, 
                        lgbm_params,
                        seasonal_averages,
                        current_feature_columns_selected # Pass the list explicitly
                    ): _id for _id in target_columns
                }
                
                for future in concurrent.futures.as_completed(future_to_id):
                    _id = future_to_id[future]
                    try:
                        result = future.result()
                        final_results[_id] = result
                        st.text(f"✅ ID {_id} completed: {result['status']}")
                    except Exception as exc:
                        final_results[_id] = {'id': _id, 'status': f'Error: Unhandled exception: {exc}'}
                        st.error(f"❌ ID {_id} generated an exception: {exc}")

        st.session_state['model_run_summary']['results'] = final_results
        display_model_results(st.session_state['model_run_summary'])