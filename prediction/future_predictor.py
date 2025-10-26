# prediction/future_predictor.py
"""Future waste predictions"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

from .predictor import BasePredictor


class FuturePredictor(BasePredictor):
    """Predict future waste with business logic"""
    
    def __init__(self, model_path=None, preprocessor_path=None):
        """Initialize future predictor"""
        # Call parent constructor
        super().__init__(model_path, preprocessor_path)
        
        # Initialize ALL attributes FIRST before any method calls
        self.restaurant_name = None
        self.historical_patterns = None
        self.food_items = None
        self.food_categories = None
        
        # Try to extract restaurant name from model path
        if model_path:
            try:
                path_str = str(model_path)
                if 'restaurants' in path_str:
                    parts = path_str.split('restaurants')[1].split(os.sep)
                    # Filter out empty strings
                    parts = [p for p in parts if p]
                    if parts:
                        self.restaurant_name = parts[0]
            except:
                pass
        
        # Now safe to call this method
        self._load_historical_patterns()
    
    def _load_historical_patterns(self):
        """Load historical patterns for better predictions"""
        try:
            # Build list of possible data file locations
            possible_paths = []
            
            # Restaurant-specific path (if we have restaurant name)
            if self.restaurant_name:
                possible_paths.append(f'data/restaurants/{self.restaurant_name}/sample_data.csv')
            
            # Common paths
            possible_paths.extend([
                'data/raw/food_waste_data.csv',
                'data/processed/processed_data.csv',
                'data/historical_data.csv'
            ])
            
            # Try to load data from any available source
            df = None
            for path in possible_paths:
                if os.path.exists(path):
                    try:
                        df = pd.read_csv(path)
                        # Successfully loaded - no need to print
                        break
                    except:
                        continue
            
            if df is None:
                # No data found - use defaults silently
                self._set_default_patterns()
                return
            
            # Extract patterns from loaded data
            if 'food_item' in df.columns:
                self.food_items = df['food_item'].unique().tolist()
            else:
                self.food_items = ['Grilled Chicken', 'Pasta', 'Pizza', 'Salad', 'Cake']
            
            if 'food_item' in df.columns and 'food_category' in df.columns:
                self.food_categories = df.groupby('food_item')['food_category'].first().to_dict()
            else:
                self.food_categories = {item: 'mains' for item in self.food_items}
            
            # Build historical patterns dictionary
            self.historical_patterns = {}
            
            # Customer count by day of week
            if 'day_of_week' in df.columns and 'customer_count' in df.columns:
                self.historical_patterns['customer_by_day'] = df.groupby('day_of_week')['customer_count'].mean().to_dict()
            else:
                self.historical_patterns['customer_by_day'] = {
                    'Monday': 120, 'Tuesday': 130, 'Wednesday': 140,
                    'Thursday': 150, 'Friday': 180, 'Saturday': 200, 'Sunday': 190
                }
            
            # Waste by item
            if 'food_item' in df.columns and 'quantity_wasted' in df.columns:
                self.historical_patterns['waste_by_item'] = df.groupby('food_item')['quantity_wasted'].mean().to_dict()
            else:
                self.historical_patterns['waste_by_item'] = {item: 5.0 for item in self.food_items}
            
            # Preparation by item
            if 'food_item' in df.columns and 'quantity_prepared' in df.columns:
                self.historical_patterns['prep_by_item'] = df.groupby('food_item')['quantity_prepared'].mean().to_dict()
            else:
                self.historical_patterns['prep_by_item'] = {item: 50.0 for item in self.food_items}
            
            # Sales by item
            if 'food_item' in df.columns and 'quantity_sold' in df.columns:
                self.historical_patterns['sales_by_item'] = df.groupby('food_item')['quantity_sold'].mean().to_dict()
            else:
                self.historical_patterns['sales_by_item'] = {item: 40.0 for item in self.food_items}
            
            # Monthly customer patterns
            if 'month' in df.columns and 'customer_count' in df.columns:
                self.historical_patterns['monthly_customers'] = df.groupby('month')['customer_count'].mean().to_dict()
            else:
                self.historical_patterns['monthly_customers'] = {i: 150 for i in range(1, 13)}
            
        except Exception as e:
            # Silently use defaults on any error
            self._set_default_patterns()
    
    def _set_default_patterns(self):
        """Set default patterns when no historical data available"""
        self.food_items = ['Grilled Chicken', 'Pasta', 'Pizza', 'Salad', 'Cake']
        self.food_categories = {item: 'mains' for item in self.food_items}
        
        self.historical_patterns = {
            'customer_by_day': {
                'Monday': 120, 'Tuesday': 130, 'Wednesday': 140,
                'Thursday': 150, 'Friday': 180, 'Saturday': 200, 'Sunday': 190
            },
            'waste_by_item': {item: 5.0 for item in self.food_items},
            'prep_by_item': {item: 50.0 for item in self.food_items},
            'sales_by_item': {item: 40.0 for item in self.food_items},
            'monthly_customers': {i: 150 for i in range(1, 13)}
        }
    
    def predict_single_day(self, date, food_item, expected_customers=None, 
                          weather='sunny', temperature=20, special_event='none'):
        """Predict waste for a single day and item"""
        
        # Parse date
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        # Auto-estimate customers if not provided
        if expected_customers is None:
            day_name = date.strftime('%A')
            expected_customers = self.historical_patterns['customer_by_day'].get(day_name, 150)
        
        # Get historical averages for food item
        avg_prep = self.historical_patterns['prep_by_item'].get(food_item, 50.0)
        avg_sales = self.historical_patterns['sales_by_item'].get(food_item, 40.0)
        
        # Adjust based on customer count
        customer_factor = expected_customers / 150  # 150 is baseline
        quantity_prepared = avg_prep * customer_factor
        quantity_sold = avg_sales * customer_factor
        
        # Weather adjustments
        if weather == 'rainy':
            quantity_sold *= 0.9
        elif weather == 'stormy':
            quantity_sold *= 0.7
        
        # Special event boost
        if special_event != 'none':
            quantity_prepared *= 1.3
            quantity_sold *= 1.2
        
        # Create input features
        input_data = {
            'date': date.strftime('%Y-%m-%d'),
            'food_item': food_item,
            'food_category': self.food_categories.get(food_item, 'mains'),
            'quantity_prepared': quantity_prepared,
            'quantity_sold': quantity_sold,
            'day_of_week': date.strftime('%A'),
            'is_weekend': int(date.weekday() >= 5),
            'weather': weather,
            'temperature': temperature,
            'special_event': special_event,
            'customer_count': expected_customers,
            'month': date.month,
            'unit_cost': 8.5  # Default cost
        }
        
        # Make prediction
        result = self.predict_with_confidence(input_data)
        
        # Add metadata
        result['date'] = date
        result['food_item'] = food_item
        result['expected_customers'] = expected_customers
        result['estimated_cost'] = result['predictions'][0] * input_data['unit_cost']
        result['preparation'] = quantity_prepared
        result['expected_sales'] = quantity_sold
        
        return result
    
    def predict_multiple_items(self, date, food_items=None, **kwargs):
        """Predict waste for multiple items on a single day"""
        
        if food_items is None:
            food_items = self.food_items
        
        predictions = []
        
        for item in food_items:
            result = self.predict_single_day(date, item, **kwargs)
            predictions.append({
                'food_item': item,
                'predicted_waste': result['predictions'][0],
                'lower_bound': result['lower_bound'][0],
                'upper_bound': result['upper_bound'][0],
                'estimated_cost': result['estimated_cost'],
                'preparation': result['preparation'],
                'expected_sales': result['expected_sales']
            })
        
        return pd.DataFrame(predictions)
    
    def predict_week_ahead(self, start_date=None):
        """Predict waste for the next 7 days"""
        
        if start_date is None:
            start_date = datetime.now() + timedelta(days=1)
        elif isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        
        print(f"\n🔮 PREDICTING WASTE FOR NEXT 7 DAYS")
        print(f"   Starting from: {start_date.strftime('%Y-%m-%d')}")
        print("="*60)
        
        weekly_predictions = []
        
        for day_offset in range(7):
            prediction_date = start_date + timedelta(days=day_offset)
            day_name = prediction_date.strftime('%A')
            
            # Get expected customers for this day
            expected_customers = self.historical_patterns['customer_by_day'].get(day_name, 150)
            
            # Predict for all items
            daily_total = 0
            daily_cost = 0
            
            for food_item in self.food_items[:5]:  # Top 5 items
                result = self.predict_single_day(
                    prediction_date,
                    food_item,
                    expected_customers=expected_customers,
                    weather='sunny',  # Default
                    temperature=20,
                    special_event='none'
                )
                
                daily_total += result['predictions'][0]
                daily_cost += result['estimated_cost']
            
            weekly_predictions.append({
                'date': prediction_date.strftime('%Y-%m-%d'),
                'day': day_name,
                'total_waste': daily_total,
                'total_cost': daily_cost,
                'customer_count': expected_customers
            })
        
        df_week = pd.DataFrame(weekly_predictions)
        
        # Display summary
        self._display_weekly_summary(df_week)
        
        return df_week
    
    def _display_weekly_summary(self, df_week):
        """Display weekly forecast summary"""
        print("\n📊 7-DAY FORECAST SUMMARY:")
        print(df_week.to_string(index=False))
        
        print(f"\n📈 Weekly Statistics:")
        print(f"   Total predicted waste: {df_week['total_waste'].sum():.2f} kg")
        print(f"   Average daily waste: {df_week['total_waste'].mean():.2f} kg")
        print(f"   Total estimated cost: ${df_week['total_cost'].sum():,.2f}")
        print(f"   Peak waste day: {df_week.loc[df_week['total_waste'].idxmax(), 'day']}")
        print(f"   Lowest waste day: {df_week.loc[df_week['total_waste'].idxmin(), 'day']}")
    
    def generate_recommendations(self, predictions_df):
        """Generate actionable recommendations"""
        print("\n💡 RECOMMENDATIONS:")
        print("="*60)
        
        # High waste items
        high_waste = predictions_df[predictions_df['predicted_waste'] > 10]
        if not high_waste.empty:
            print("\n⚠️ HIGH WASTE ALERT:")
            for _, row in high_waste.iterrows():
                reduction = row['predicted_waste'] * 0.2
                print(f"   • {row['food_item']}: Reduce preparation by {reduction:.1f} kg")
        
        # Low waste items
        low_waste = predictions_df[predictions_df['predicted_waste'] < 2]
        if not low_waste.empty:
            print("\n✅ EFFICIENT ITEMS:")
            for _, row in low_waste.iterrows():
                print(f"   • {row['food_item']}: Good waste control ({row['predicted_waste']:.1f} kg)")
        
        # Cost impact
        total_cost = predictions_df['estimated_cost'].sum()
        print(f"\n💰 TOTAL ESTIMATED WASTE COST: ${total_cost:,.2f}")
        
        # Potential savings with 20% reduction
        potential_savings = total_cost * 0.2
        print(f"   Potential savings (20% reduction): ${potential_savings:,.2f}")