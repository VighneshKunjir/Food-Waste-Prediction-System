# core/feature_engineer.py
"""Feature engineering logic"""

import pandas as pd
import numpy as np


class FeatureEngineer:
    """Create features from raw data"""
    
    def create_features(self, df, is_training=True):
        """Create all features"""
        df = df.copy()
        
        # Date features
        df = self._create_date_features(df)
        
        # Lag features (only for training)
        if is_training and 'quantity_wasted' in df.columns:
            df = self._create_lag_features(df)
        else:
            df = self._add_default_lag_features(df)
        
        # Efficiency metrics
        df = self._create_efficiency_features(df)
        
        # Pattern features
        if is_training:
            df = self._create_pattern_features(df)
        else:
            df = self._add_default_pattern_features(df)
        
        # Weather features
        df = self._create_weather_features(df)
        
        return df
    
    def _create_date_features(self, df):
        """Create date-based features"""
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_month'] = df['date'].dt.day
            df['week_of_year'] = df['date'].dt.isocalendar().week
            df['quarter'] = df['date'].dt.quarter
            df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
            df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        
        return df
    
    def _create_lag_features(self, df):
        """Create lag features for training"""
        if 'food_item' not in df.columns or 'date' not in df.columns:
            return df
        
        df = df.sort_values(['food_item', 'date'])
        
        # Waste lag features
        for lag in [1, 3, 7]:
            df[f'waste_lag_{lag}'] = df.groupby('food_item')['quantity_wasted'].shift(lag)
            if 'quantity_sold' in df.columns:
                df[f'sales_lag_{lag}'] = df.groupby('food_item')['quantity_sold'].shift(lag)
        
        # Rolling statistics
        for window in [3, 7, 14]:
            df[f'waste_rolling_mean_{window}'] = df.groupby('food_item')['quantity_wasted'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'waste_rolling_std_{window}'] = df.groupby('food_item')['quantity_wasted'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
        
        # Category statistics
        if 'food_category' in df.columns:
            df['category_avg_waste'] = df.groupby('food_category')['quantity_wasted'].transform('mean')
            df['item_avg_waste'] = df.groupby('food_item')['quantity_wasted'].transform('mean')
            df['item_std_waste'] = df.groupby('food_item')['quantity_wasted'].transform('std')
        
        return df
    
    def _add_default_lag_features(self, df):
        """Add default lag features for prediction"""
        for lag in [1, 3, 7]:
            df[f'waste_lag_{lag}'] = 0
            df[f'sales_lag_{lag}'] = 0
        
        for window in [3, 7, 14]:
            df[f'waste_rolling_mean_{window}'] = 0
            df[f'waste_rolling_std_{window}'] = 0
        
        df['category_avg_waste'] = 0
        df['item_avg_waste'] = 0
        df['item_std_waste'] = 0
        
        return df
    
    def _create_efficiency_features(self, df):
        """Create efficiency metrics"""
        if 'quantity_sold' in df.columns and 'quantity_prepared' in df.columns:
            df['preparation_efficiency'] = df['quantity_sold'] / (df['quantity_prepared'] + 0.001)
        else:
            df['preparation_efficiency'] = 0
        
        if 'quantity_wasted' in df.columns and 'quantity_prepared' in df.columns:
            df['waste_percentage'] = df['quantity_wasted'] / (df['quantity_prepared'] + 0.001)
        else:
            df['waste_percentage'] = 0
        
        if 'quantity_sold' in df.columns and 'customer_count' in df.columns:
            df['sales_per_customer'] = df['quantity_sold'] / (df['customer_count'] + 1)
        else:
            df['sales_per_customer'] = 0
        
        if 'quantity_wasted' in df.columns and 'customer_count' in df.columns:
            df['waste_per_customer'] = df['quantity_wasted'] / (df['customer_count'] + 1)
        else:
            df['waste_per_customer'] = 0
        
        return df
    
    def _create_pattern_features(self, df):
        """Create pattern-based features"""
        if 'day_of_week' in df.columns and 'customer_count' in df.columns:
            df['dow_avg_customers'] = df.groupby('day_of_week')['customer_count'].transform('mean')
            if 'quantity_wasted' in df.columns:
                df['dow_avg_waste'] = df.groupby('day_of_week')['quantity_wasted'].transform('mean')
            else:
                df['dow_avg_waste'] = 0
        else:
            df['dow_avg_customers'] = df.get('customer_count', 100)
            df['dow_avg_waste'] = 0
        
        return df
    
    def _add_default_pattern_features(self, df):
        """Add default pattern features"""
        df['dow_avg_customers'] = df.get('customer_count', 100)
        df['dow_avg_waste'] = 0
        return df
    
    def _create_weather_features(self, df):
        """Create weather-based features"""
        if 'weather' in df.columns:
            weather_impact = {'sunny': 0, 'cloudy': 1, 'rainy': 2, 'stormy': 3}
            df['weather_severity'] = df['weather'].map(weather_impact).fillna(0)
        else:
            df['weather_severity'] = 0
        
        if 'temperature' in df.columns:
            df['temp_category'] = pd.cut(
                df['temperature'],
                bins=[-np.inf, 10, 20, 30, np.inf],
                labels=['cold', 'mild', 'warm', 'hot']
            )
        else:
            df['temp_category'] = 'mild'
        
        return df