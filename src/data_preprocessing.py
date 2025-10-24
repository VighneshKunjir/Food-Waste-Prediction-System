# src/data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = 'quantity_wasted'
        self.is_training = True  # Flag to distinguish training from prediction
        
    def load_data(self, filepath):
        """Load data from CSV file"""
        print(f"📂 Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(df)} records with {len(df.columns)} columns")
        return df
    
    def create_features(self, df, is_training=True):
        """Feature engineering - create new features from existing ones"""
        if is_training:
            print("🔧 Creating features...")
        
        df = df.copy()
        
        # Convert date to datetime if it's a string
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
            # Time-based features
            df['day_of_month'] = df['date'].dt.day
            df['week_of_year'] = df['date'].dt.isocalendar().week
            df['quarter'] = df['date'].dt.quarter
            df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
            df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        
        # Only create lag features during training when quantity_wasted exists
        if 'quantity_wasted' in df.columns and is_training:
            # Sort for lag features
            if 'food_item' in df.columns and 'date' in df.columns:
                df = df.sort_values(['food_item', 'date'])
                
                # Lag features - previous waste values
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
                
                # Category-level statistics
                if 'food_category' in df.columns:
                    df['category_avg_waste'] = df.groupby('food_category')['quantity_wasted'].transform('mean')
                    df['item_avg_waste'] = df.groupby('food_item')['quantity_wasted'].transform('mean')
                    df['item_std_waste'] = df.groupby('food_item')['quantity_wasted'].transform('std')
        else:
            # For prediction, fill lag features with default values
            for lag in [1, 3, 7]:
                df[f'waste_lag_{lag}'] = 0
                df[f'sales_lag_{lag}'] = 0
                
            for window in [3, 7, 14]:
                df[f'waste_rolling_mean_{window}'] = 0
                df[f'waste_rolling_std_{window}'] = 0
            
            df['category_avg_waste'] = 0
            df['item_avg_waste'] = 0
            df['item_std_waste'] = 0
        
        # Efficiency metrics (safe for both training and prediction)
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
        
        # Day of week patterns (safe for both)
        if 'day_of_week' in df.columns and 'customer_count' in df.columns and is_training:
            df['dow_avg_customers'] = df.groupby('day_of_week')['customer_count'].transform('mean')
            if 'quantity_wasted' in df.columns:
                df['dow_avg_waste'] = df.groupby('day_of_week')['quantity_wasted'].transform('mean')
            else:
                df['dow_avg_waste'] = 0
        else:
            df['dow_avg_customers'] = df.get('customer_count', 100)
            df['dow_avg_waste'] = 0
        
        # Weather impact features
        if 'weather' in df.columns:
            weather_impact = {'sunny': 0, 'cloudy': 1, 'rainy': 2, 'stormy': 3}
            df['weather_severity'] = df['weather'].map(weather_impact).fillna(0)
        else:
            df['weather_severity'] = 0
        
        # Temperature bins
        if 'temperature' in df.columns:
            df['temp_category'] = pd.cut(df['temperature'], 
                                         bins=[-np.inf, 10, 20, 30, np.inf],
                                         labels=['cold', 'mild', 'warm', 'hot'])
        else:
            df['temp_category'] = 'mild'
        
        if is_training:
            print(f"✅ Created {len(df.columns)} total features")
        
        return df
    
    def encode_categorical(self, df, is_training=True):
        """Encode categorical variables"""
        if is_training:
            print("🔄 Encoding categorical variables...")
        
        categorical_columns = ['food_item', 'food_category', 'day_of_week', 
                              'weather', 'special_event', 'temp_category']
        
        for col in categorical_columns:
            if col in df.columns:
                if is_training and col not in self.label_encoders:
                    # During training, fit new encoders
                    self.label_encoders[col] = LabelEncoder()
                    # Handle NaN values
                    df[col] = df[col].fillna('unknown').astype(str)
                    df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col])
                elif col in self.label_encoders:
                    # During prediction, use existing encoders
                    df[col] = df[col].fillna('unknown').astype(str)
                    # Handle unseen categories
                    df[col + '_encoded'] = df[col].apply(
                        lambda x: self.label_encoders[col].transform([x])[0] 
                        if x in self.label_encoders[col].classes_ else -1
                    )
                else:
                    # If encoder doesn't exist, use default encoding
                    df[col + '_encoded'] = 0
        
        return df
    
    def handle_missing_values(self, df, is_training=True):
        """Handle missing values"""
        if is_training:
            print("🧹 Handling missing values...")
        
        # Forward fill for time series features
        time_features = [col for col in df.columns if 'lag' in col or 'rolling' in col]
        if time_features:
            df[time_features] = df[time_features].fillna(0)
        
        # Fill remaining with appropriate methods
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Fill categorical columns
        object_columns = df.select_dtypes(include=['object']).columns
        for col in object_columns:
            df[col] = df[col].fillna('unknown')
        
        if is_training:
            print(f"✅ Missing values handled. Final shape: {df.shape}")
        
        return df
    
    def select_features(self, df):
        """Select relevant features for modeling"""
        print("📊 Selecting features...")
        
        # Features to exclude from training
        exclude_features = ['date', 'quantity_wasted', 'food_item', 'food_category',
                          'day_of_week', 'weather', 'special_event', 'temp_category',
                          'total_cost', 'waste_cost']
        
        # Select all numeric features except excluded ones
        feature_cols = [col for col in df.columns if col not in exclude_features]
        
        # Remove any remaining non-numeric columns
        numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        self.feature_columns = numeric_features
        print(f"✅ Selected {len(self.feature_columns)} features for training")
        
        return numeric_features
    
    def prepare_data(self, filepath, test_size=0.2, random_state=42):
        """Complete data preparation pipeline for training"""
        print("\n" + "="*50)
        print("🚀 STARTING DATA PREPARATION PIPELINE")
        print("="*50 + "\n")
        
        self.is_training = True
        
        # Load data
        df = self.load_data(filepath)
        
        # Create features
        df = self.create_features(df, is_training=True)
        
        # Encode categorical variables
        df = self.encode_categorical(df, is_training=True)
        
        # Handle missing values
        df = self.handle_missing_values(df, is_training=True)
        
        # Select features
        feature_columns = self.select_features(df)
        
        # Prepare X and y
        X = df[feature_columns]
        y = df[self.target_column]
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\n📊 Data split completed:")
        print(f"   Training set: {X_train.shape}")
        print(f"   Test set: {X_test.shape}")
        
        # Save processed data
        df.to_csv('data/processed/processed_data.csv', index=False)
        print(f"\n💾 Processed data saved to data/processed/")
        
        return X_train, X_test, y_train, y_test, df
    
    def prepare_prediction_data(self, df):
        """Prepare data for prediction (not training)"""
        self.is_training = False
        
        # Create features without requiring quantity_wasted
        df = self.create_features(df, is_training=False)
        
        # Encode categorical variables
        df = self.encode_categorical(df, is_training=False)
        
        # Handle missing values
        df = self.handle_missing_values(df, is_training=False)
        
        # Ensure all required features exist
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0
            
            # Select only the features used during training
            df = df[self.feature_columns]
            
            # Scale features
            df_scaled = pd.DataFrame(
                self.scaler.transform(df),
                columns=df.columns,
                index=df.index
            )
            return df_scaled
        
        return df