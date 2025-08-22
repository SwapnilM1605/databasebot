import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Optional
from datetime import datetime
import pmdarima as pm
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForecastingService:
    def __init__(self):
        self.available_models = {
            'arima': self.arima_forecast,
            'random_forest': self.random_forest_forecast
        }
        self.scaler = StandardScaler()
    
    def prepare_time_series_data(self, data: List[Dict], date_col: str, value_col: str, 
                               feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Prepare time series data for forecasting with additional features.
        Handles data coming directly from database queries.
        """
        try:
            # Convert to DataFrame if not already
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
            
            # Validate input
            if df.empty:
                raise ValueError("Empty DataFrame provided")
                
            # Convert date column - handle database timestamp formats
            if date_col in df.columns:
                # Handle various date formats from databases
                if pd.api.types.is_datetime64_any_dtype(df[date_col]):
                    pass  # Already datetime
                else:
                    try:
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    except Exception as e:
                        logger.error(f"Date conversion error: {str(e)}")
                        raise ValueError(f"Could not convert {date_col} to datetime")
                
                if df[date_col].isnull().any():
                    raise ValueError(f"Invalid dates found in {date_col}")
                    
            # Convert value column to numeric
            if value_col in df.columns:
                df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
                if df[value_col].isnull().any():
                    raise ValueError(f"Invalid numeric values found in {value_col}")
                    
            # Sort by date in ascending order (oldest to newest)
            if date_col in df.columns:
                df = df.sort_values(date_col, ascending=True)
            
            # Reset index after sorting
            df = df.reset_index(drop=True)
            
            # Handle feature columns
            if feature_cols:
                for col in feature_cols:
                    if col not in df.columns:
                        continue
                        
                    # Skip date column if it's in feature_cols
                    if col == date_col:
                        continue
                        
                    # Convert to numeric if possible
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Fill any remaining NA values
                    df[col] = df[col].ffill().bfill()
                    
                    # Skip if still empty or constant
                    if df[col].isnull().all() or df[col].nunique() == 1:
                        continue
                        
                    # Scale numeric features
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = self.scaler.fit_transform(df[[col]].values.reshape(-1, 1))
            
            return df.dropna(subset=[date_col, value_col])
            
        except Exception as e:
            logger.error(f"Error preparing time series data: {str(e)}")
            raise ValueError(f"Error preparing data: {str(e)}")
    
    def arima_forecast(self, ts_data: pd.Series, periods: int = 5) -> Dict[str, Any]:
        try:
            # Auto ARIMA with automatic parameter selection
            model = pm.auto_arima(
                ts_data,
                start_p=1, start_q=1,
                d=1,  # Force at least first differencing
                max_p=3, max_q=3,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore'
            )
            
            forecast, conf_int = model.predict(n_periods=periods, return_conf_int=True)
            
            return {
                'model': 'ARIMA',
                'order': model.order,
                'forecast': forecast.tolist(),
                'last_date': ts_data.index[-1],
                'metrics': {
                    'aic': model.aic(),
                    'bic': model.bic(),
                    'conf_int': conf_int.tolist()
                }
            }
        except Exception as e:
            logger.error(f"ARIMA forecasting error: {str(e)}")
            raise ValueError(f"ARIMA forecasting error: {str(e)}")
    
    def random_forest_forecast(self, df: pd.DataFrame, date_col: str, 
                            value_col: str, feature_cols: Optional[List[str]] = None,
                            periods: int = 5) -> Dict[str, Any]:
        """
        Random Forest forecasting with feature engineering
        """
        try:
            df = df.copy()
            
            # Feature engineering from date
            df['date'] = pd.to_datetime(df[date_col])
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['day_of_week'] = df['date'].dt.dayofweek
            df['day_of_year'] = df['date'].dt.dayofyear
            df['week_of_year'] = df['date'].dt.isocalendar().week
            
            # Prepare features
            base_features = ['year', 'month', 'day', 'day_of_week', 'day_of_year', 'week_of_year']
            if feature_cols:
                base_features.extend(feature_cols)
                
            # Remove duplicates
            base_features = list(set(base_features))
            
            X = df[base_features]
            y = df[value_col]
            
            # Time-series aware train-test split
            test_size = min(0.2, 30/len(df))
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=False)
            
            # Model with optimized hyperparameters
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Generate future dates
            last_date = df['date'].max()
            future_dates = pd.date_range(
                start=last_date, 
                periods=periods+1, 
                freq='D'
            )[1:]
            
            # Create future dataframe with date features
            future_df = pd.DataFrame({
                'date': future_dates,
                'year': future_dates.year,
                'month': future_dates.month,
                'day': future_dates.day,
                'day_of_week': future_dates.dayofweek,
                'day_of_year': future_dates.dayofyear,
                'week_of_year': future_dates.isocalendar().week
            })
            
            # Handle additional features for future periods
            if feature_cols:
                for col in feature_cols:
                    if col not in ['date', 'year', 'month', 'day', 'day_of_week', 
                                'day_of_year', 'week_of_year']:
                        future_df[col] = df[col].iloc[-1]
            
            # Make predictions
            predictions = model.predict(future_df[base_features])
            
            # Calculate metrics and handle NaN values
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Convert NaN to None (which becomes null in JSON)
            metrics = {
                'mae': float(mae) if not np.isnan(mae) else None,
                'r2_score': float(r2) if not np.isnan(r2) else None,
                'feature_importance': dict(zip(base_features, model.feature_importances_))
            }
            
            return {
                'model': 'Random Forest',
                'forecast': [{'date': d, 'prediction': p} for d, p in zip(future_dates, predictions)],
                'last_date': last_date,
                'metrics': metrics
            }
        except Exception as e:
            logger.error(f"Random Forest forecasting error: {str(e)}")
            raise ValueError(f"Random Forest forecasting error: {str(e)}")
    
    def detect_best_model(self, df: pd.DataFrame, date_col: str, 
                         value_col: str, feature_cols: Optional[List[str]] = None) -> str:
        """
        Heuristic to suggest the best model based on data characteristics
        """
        try:
            # Check data size and characteristics
            if len(df) < 10 or feature_cols:
                return 'random_forest'
            
            # Check for stationarity
            try:
                result = adfuller(df[value_col].dropna())
                if result[1] > 0.05:  # Not stationary
                    return 'random_forest'
            except:
                return 'random_forest'
                
            return 'arima'
        except Exception:
            return 'random_forest'  # Safest fallback
    
    def forecast(self, data: List[Dict[str, Any]], date_col: str, value_col: str, 
                periods: int = 5, model: str = 'auto', 
                feature_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Main forecasting method with fallback logic
        """
        try:
            # Prepare data - handle both raw data and database results
            prepared_data = self.prepare_time_series_data(data, date_col, value_col, feature_cols)
            
            # Try requested model first
            if model != 'auto':
                try:
                    if model == 'arima':
                        ts_data = prepared_data.set_index(date_col)[value_col]
                        return self.available_models[model](ts_data, periods)
                    else:
                        return self.available_models[model](
                            prepared_data, date_col, value_col, 
                            feature_cols=feature_cols, periods=periods
                        )
                except Exception as e:
                    logger.warning(f"Model {model} failed, trying auto selection: {str(e)}")
                    model = 'auto'
            
            # Auto model selection with fallbacks
            if model == 'auto':
                models_to_try = [
                    self.detect_best_model(prepared_data, date_col, value_col, feature_cols),
                    'random_forest',
                    'arima'
                ]
                
                for model_name in set(models_to_try):
                    try:
                        if model_name == 'arima':
                            ts_data = prepared_data.set_index(date_col)[value_col]
                            return self.available_models[model_name](ts_data, periods)
                        else:
                            return self.available_models[model_name](
                                prepared_data, date_col, value_col,
                                feature_cols=feature_cols, periods=periods
                            )
                    except Exception as e:
                        logger.warning(f"Model {model_name} failed: {str(e)}")
                        continue
                        
            raise ValueError("All attempted forecasting models failed")
            
        except Exception as e:
            logger.error(f"Forecasting failed: {str(e)}")
            raise ValueError(f"Forecasting failed: {str(e)}")
    
    def format_forecast_results(self, forecast_result: Dict[str, Any], 
                            original_query: str) -> str:
        """
        Format forecasting results into natural language
        """
        model_name = forecast_result['model']
        forecast_values = forecast_result['forecast']
        last_date = forecast_result['last_date']
        
        if isinstance(last_date, str):
            last_date = pd.to_datetime(last_date)
        
        response = f"""Based on our analysis using the {model_name} forecasting model, here are the predictions:
        
    **Forecast Results:**
    """
        if model_name == 'ARIMA':
            for i, value in enumerate(forecast_values, 1):
                forecast_date = last_date + pd.Timedelta(days=i)
                response += f"- {forecast_date.strftime('%Y-%m-%d')}: {value:.2f}\n"
        elif model_name == 'Random Forest':
            for item in forecast_values:
                date = pd.to_datetime(item['date']).strftime('%Y-%m-%d')
                response += f"- {date}: {item['prediction']:.2f}\n"
        
        # Add model metrics if available
        if 'metrics' in forecast_result:
            response += f"""
    **Model Metrics:**
    """
            for metric, value in forecast_result['metrics'].items():
                if metric == 'feature_importance':
                    response += f"- Feature Importance:\n"
                    for feat, imp in sorted(value.items(), key=lambda x: x[1], reverse=True)[:5]:
                        response += f"  - {feat}: {imp:.3f}\n"
                else:
                    # Handle None values
                    if value is None:
                        response += f"- {metric}: Not available\n"
                    else:
                        response += f"- {metric}: {value:.3f}\n"
        
        response += f"""
    **Last Historical Data Point**: {last_date.strftime('%Y-%m-%d')}
    """
        return response

    def create_forecast_visualization(self, historical_df: pd.DataFrame, 
                                    forecast_result: Dict[str, Any], 
                                    date_col: str, value_col: str) -> Dict[str, Any]:
        """
        Create visualization data for historical + forecasted values
        """
        try:
            # Prepare historical data
            historical_data = historical_df[[date_col, value_col]].copy()
            historical_data['type'] = 'Historical'
            
            # Prepare forecast data based on model type
            if forecast_result['model'] == 'ARIMA':
                forecast_dates = pd.date_range(
                    start=forecast_result['last_date'],
                    periods=len(forecast_result['forecast'])+1,
                    freq='D'
                )[1:]
                forecast_data = pd.DataFrame({
                    date_col: forecast_dates,
                    value_col: forecast_result['forecast'],
                    'type': 'Forecast'
                })
                # Add confidence intervals if available
                if 'conf_int' in forecast_result['metrics']:
                    forecast_data['yhat_lower'] = [x[0] for x in forecast_result['metrics']['conf_int']]
                    forecast_data['yhat_upper'] = [x[1] for x in forecast_result['metrics']['conf_int']]
            elif forecast_result['model'] == 'Random Forest':
                forecast_data = pd.DataFrame(forecast_result['forecast'])
                forecast_data = forecast_data.rename(columns={
                    'date': date_col,
                    'prediction': value_col
                })
                forecast_data['type'] = 'Forecast'
                forecast_data[date_col] = pd.to_datetime(forecast_data[date_col])
            
            # Combine historical and forecast data
            combined = pd.concat([historical_data, forecast_data])
            
            # Create chart data
            chart_data = {
                "type": "line",
                "data": {
                    "labels": combined[date_col].dt.strftime('%Y-%m-%d').tolist(),
                    "datasets": [
                        {
                            "label": "Historical",
                            "data": combined[combined['type'] == 'Historical'][value_col].tolist(),
                            "borderColor": "rgba(75, 192, 192, 1)",
                            "backgroundColor": "rgba(75, 192, 192, 0.2)",
                            "borderWidth": 2,
                            "fill": False
                        },
                        {
                            "label": "Forecast",
                            "data": combined[combined['type'] == 'Forecast'][value_col].tolist(),
                            "borderColor": "rgba(255, 99, 132, 1)",
                            "backgroundColor": "rgba(255, 99, 132, 0.2)",
                            "borderWidth": 2,
                            "borderDash": [5, 5],
                            "fill": False
                        }
                    ]
                },
                "options": {
                    "responsive": True,
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": f"{value_col} Forecast",
                            "font": {"size": 16}
                        }
                    }
                }
            }
            
            # Add confidence intervals for ARIMA
            if forecast_result['model'] == 'ARIMA' and 'yhat_lower' in forecast_data.columns:
                chart_data["data"]["datasets"].append({
                    "label": "Confidence Interval",
                    "data": combined[combined['type'] == 'Forecast']['yhat_upper'].tolist(),
                    "borderColor": "rgba(255, 99, 132, 0.3)",
                    "backgroundColor": "rgba(255, 99, 132, 0.1)",
                    "borderWidth": 1,
                    "fill": "-1",
                    "pointRadius": 0
                })
                chart_data["data"]["datasets"].append({
                    "label": "Confidence Interval",
                    "data": combined[combined['type'] == 'Forecast']['yhat_lower'].tolist(),
                    "borderColor": "rgba(255, 99, 132, 0.3)",
                    "backgroundColor": "rgba(255, 99, 132, 0.1)",
                    "borderWidth": 1,
                    "pointRadius": 0
                })
            
            return chart_data
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return None