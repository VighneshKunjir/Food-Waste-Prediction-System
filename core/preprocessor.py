# core/preprocessor.py
"""Data preprocessing pipeline"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class Preprocessor:
    """Handle all preprocessing operations"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = 'quantity_wasted'
    
    def fit_transform(self, df, test_size=0.2, random_state=42):
        """Complete preprocessing pipeline for training"""
        print("\n🔧 PREPROCESSING DATA")
        print("="*50)
        
        # Feature engineering
        from .feature_engineer import FeatureEngineer
        engineer = FeatureEngineer()
        df = engineer.create_features(df, is_training=True)
        
        # Encode categorical
        df = self._encode_categorical(df, fit=True)
        
        # Handle missing values
        df = self._handle_missing(df)
        
        # Select features
        self.feature_columns = self._select_features(df)
        
        # Prepare X and y
        X = df[self.feature_columns]
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
        
        print(f"\n✅ Preprocessing complete")
        print(f"   Training: {X_train.shape}")
        print(f"   Test: {X_test.shape}")
        print(f"   Features: {len(self.feature_columns)}")
        
        return X_train, X_test, y_train, y_test
    
    def transform(self, df):
        """Transform new data using fitted preprocessor"""
        from .feature_engineer import FeatureEngineer
        engineer = FeatureEngineer()
        
        # Create features
        df = engineer.create_features(df, is_training=False)
        
        # Encode categorical
        df = self._encode_categorical(df, fit=False)
        
        # Handle missing
        df = self._handle_missing(df)
        
        # Ensure all features exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Select features
        df = df[self.feature_columns]
        
        # Scale
        df_scaled = pd.DataFrame(
            self.scaler.transform(df),
            columns=df.columns,
            index=df.index
        )
        
        return df_scaled
    
    def _encode_categorical(self, df, fit=True):
        """Encode categorical variables"""
        categorical_columns = [
            'food_item', 'food_category', 'day_of_week',
            'weather', 'special_event'
        ]
        
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('unknown').astype(str)
                
                if fit:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col])
                else:
                    if col in self.label_encoders:
                        df[col + '_encoded'] = df[col].apply(
                            lambda x: self.label_encoders[col].transform([x])[0]
                            if x in self.label_encoders[col].classes_ else -1
                        )
                    else:
                        df[col + '_encoded'] = 0
        
        return df
    
    def _handle_missing(self, df):
        """Handle missing values"""
        # Numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Object columns
        object_columns = df.select_dtypes(include=['object']).columns
        for col in object_columns:
            df[col] = df[col].fillna('unknown')
        
        return df
    
    def _select_features(self, df):
        """Select features for modeling"""
        exclude_features = [
            'date', 'quantity_wasted', 'food_item', 'food_category',
            'day_of_week', 'weather', 'special_event', 'temp_category',
            'total_cost', 'waste_cost'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_features]
        numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"✅ Selected {len(numeric_features)} features")
        
        return numeric_features