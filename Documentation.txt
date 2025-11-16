 Advanced Energy Load PredictorA sophisticated machine learning application for forecasting energy consumption using LightGBM gradient boosting. Built with Streamlit for an intuitive user interface, this tool enables accurate short-term and long-term electricity load predictions for multiple customer segments simultaneously.Show Image


Fixed 48 Hours: Ultra-fast predictions for immediate load planning
Short-Term (Up to 1 Month): High-accuracy forecasts with iterative prediction
Long-Term (1 Year): Strategic planning with seasonal pattern recognition
Advanced Capabilities

✅ Concurrent Processing: Train multiple customer IDs simultaneously for faster results
✅ Feature Engineering: Automated creation of lag features, rolling averages, and cyclical time encodings
✅ Holiday Detection: Automatic Finnish holiday calendar integration
✅ Exogenous Variables: Support for weather data and electricity pricing
✅ Batch Processing: Predict hundreds of customer IDs in a single run
✅ Model Persistence: Incremental forecast building with state management
✅ Multiple Export Formats: Wide, long, and combined historical+forecast outputs
Performance Metrics

Root Mean Squared Error (RMSE)
R² Score
Mean Absolute Percentage Error (MAPE)
Confidence Level Estimation
Feature Importance Analysis
📋 RequirementsDependencies
txtstreamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
lightgbm>=4.0.0
scikit-learn>=1.3.0
holidays>=0.35
python-dateutil>=2.8.0Python Version

Python 3.8 or higher
🚀 Installation1. Clone the Repository
bashgit clone https://github.com/yourusername/energy-load-predictor.git
cd energy-load-predictor2. Create Virtual Environment (Recommended)
bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate3. Install Dependencies
bashpip install -r requirements.txt4. Run the Application
bashstreamlit run app.pyThe application will open in your default browser at http://localhost:8501📊 Usage GuideStep 1: Data Upload

Navigate to the "1. Data Upload 📥" tab
Upload a CSV file containing:

Timestamp column (automatically detected)
Target consumption columns (customer IDs)
Optional: Exogenous variables (weather, pricing)


Expected Data Format
csvmeasured_at,customer_001,customer_002,Average temperature [°C],eur_per_mwh
2024-01-01T00:00:00,125.5,89.3,5.2,45.30
2024-01-01T01:00:00,118.2,82.1,4.8,43.20
...Step 2: Prediction Setup

Go to "2. Prediction Setup ⚙️" tab
Choose Prediction Mode:

48h: Fixed 48-hour forecast (fastest)
Short: Flexible short-term forecast (up to 1 month)
Long: 1-year strategic forecast


Filter Active IDs: Select historical month to identify active meters
Select Target IDs: Choose which customer IDs to predict (supports "All IDs")
Select Features: Customize input features (defaults provided)
Step 3: Train & Predict

Navigate to "3. Train & Results 📈" tab
Click "🚀 Start Batch Train & Predict"
Monitor progress as models train concurrently
Review metrics and download forecasts
Download Options
The application provides four export formats:
all_ids_wide.csv: Historical + Forecast data for all IDs (wide format)
[ID]_forecast_ONLY.csv: Individual ID forecast only (2 columns)
all_ids_forecast_vertical.csv: All forecasts in long format (3 columns)
all_ids_forecast_wide.csv: All forecasts in wide format (timestamps × IDs)
⚙️ ConfigurationModel Hyperparameters
Accessible via sidebar:

Learning Rate: 0.01 - 0.3 (default: 0.1)
Number of Leaves: 20 - 50 (default: 31)
Max Depth: -1 to 15 (default: -1, no limit)
Min Child Samples: 1 - 30 (default: 20)
Boosting Type: GBDT, DART, GOSS
Metric: RMSE, MAE
Settings are automatically persisted to lgbm_params.jsonExogenous Variables (Auto-detected)
The system recognizes these columns:

Average temperature [°C]
Maximum temperature [°C]
Minimum temperature [°C]
Average relative humidity [%]
Wind speed [m/s]
Average wind direction [°]
Precipitation [mm]
Average air pressure [hPa]
eur_per_mwh (electricity price)
🏗️ Project Structureenergy-load-predictor/
│
├── app.py                          # Main application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── model_results.pkl               # Auto-generated: Model metrics
├── multi_target_forecast.pkl       # Auto-generated: Forecast persistence
├── lgbm_params.json               # Auto-generated: Hyperparameters
└── real_consumption.csv           # Auto-generated: Combined output🔧 Technical DetailsFeature Engineering
The system automatically creates:

Lag Features: 1h, 24h, 48h, 168h (1 week)
Rolling Statistics: 48-hour rolling mean
Cyclical Encoding: Hour sin/cos transformations
Time Categories: Peak_Volatile, Night_Volatile, Shoulder
Calendar Features: Hour, Day of Week, Month, Year, Day of Year
Binary Indicators: Weekends, Finnish holidays
Prediction MethodologyShort-Term (48h and Short modes)

Uses recent lag features (Lag_1, Lag_24, Rolling_Mean_48H)
Iterative prediction with rolling window updates
Fixed exogenous values (last known state)
Long-Term (1 Year)

Removes volatile short-term lags (Lag_1, Lag_24, Rolling_Mean_48H)
Uses seasonal averages for exogenous variables
Preserves weekly patterns (Lag_168)
Concurrent Processing

Utilizes ThreadPoolExecutor for parallel model training
Default: 5 concurrent workers (adjustable)
Significant speedup for batch predictions
📈 Performance OptimizationBest Practices

Data Quality: Ensure hourly frequency with minimal gaps
Active IDs: Filter by recent months to exclude inactive meters
Feature Selection: Start with defaults, remove low-importance features
Mode Selection:

Use 48h for operational planning
Use short-term for tactical decisions (days to weeks)
Use long-term for strategic capacity planning


Typical Accuracy
Based on Finnish energy data:

48h Mode: MAPE < 5-8%
Short-Term: MAPE < 8-12%
Long-Term: MAPE < 15-20%
Actual performance varies by data quality and consumption patterns🐛 TroubleshootingCommon IssuesProblem: "Insufficient data" error

Solution: Ensure at least 100 hourly data points per ID
Problem: Models fail to train

Solution: Check for NaN values in target columns, verify timestamp parsing
Problem: File upload fails on repeated use

Solution: The app auto-resets; this is expected behavior for new files
Problem: Forecast stops prematurely

Solution: Check that selected future month is within 2026 limit
Reset Functionality
Click "Reset All Forecasts and Models" to:

Clear all cached predictions
Delete persistence files
Start fresh without restarting the app
🤝 ContributingContributions are welcome! Please:

Fork the repository
Create a feature branch (git checkout -b feature/AmazingFeature)
Commit changes (git commit -m 'Add AmazingFeature')
Push to branch (git push origin feature/AmazingFeature)
Open a Pull Request
Development Guidelines

Follow PEP 8 style guide
Add docstrings to new functions
Update version number in MODEL_VERSION constant
Test with multiple data formats before submitting
📄 LicenseThis project is licensed under the MIT License - see the LICENSE file for details.👤 
Project: Junction 2025 Hackathon - Fortum Energy Challenge
🙏 Acknowledgments
Built for the Junction 2025 Hackathon
Fortum's energy consumption forecasting challenge
LightGBM team for the excellent gradient boosting framework
Streamlit for the intuitive UI framework
📞 SupportFor issues, questions, or suggestions:



Note: This application was developed for the Junction 2025 Hackathon focusing on energy consumption forecasting for 112 Finnish customer groups. The system is designed to handle spot-price and fixed-rate customer segments with sophisticated feature engineering and concurrent processing capabilities.