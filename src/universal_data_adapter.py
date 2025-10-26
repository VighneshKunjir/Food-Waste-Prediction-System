# src/universal_data_adapter.py
"""
Universal Data Adapter for Food Waste Prediction
Handles multiple CSV formats from different restaurants
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from fuzzywuzzy import fuzz
import warnings
warnings.filterwarnings('ignore')

class UniversalDataAdapter:
    def __init__(self):
        """Initialize the universal adapter with mapping rules"""
        
        # Define column mappings for common variations
        self.column_mappings = {
            'food_item': [
                'Type of Food', 'Food Type', 'Item', 'Menu Item', 'Food Name',
                'Product', 'Dish', 'Food', 'Item Name', 'Product Name',
                'Menu', 'Meal', 'Food Product', 'Recipe', 'Dish Name'
            ],
            'quantity_wasted': [
                'Wastage Food Amount', 'Waste', 'Wasted', 'Food Waste',
                'Waste Amount', 'Wastage', 'Thrown Away', 'Discarded',
                'Loss', 'Food Loss', 'Waste Quantity', 'Spoilage',
                'Waste (kg)', 'Waste (lbs)', 'Waste Weight'
            ],
            'quantity_prepared': [
                'Quantity of Food', 'Prepared', 'Made', 'Cooked',
                'Production', 'Quantity Prepared', 'Total Prepared',
                'Amount Made', 'Produced', 'Initial Quantity',
                'Starting Amount', 'Batch Size', 'Prep Amount'
            ],
            'quantity_sold': [
                'Sold', 'Sales', 'Consumed', 'Used', 'Served',
                'Quantity Sold', 'Amount Sold', 'Portions Served',
                'Sales Quantity', 'Distributed', 'Given Out'
            ],
            'customer_count': [
                'Number of Guests', 'Customers', 'Guests', 'Covers',
                'Customer Count', 'People', 'Diners', 'Attendees',
                'Footfall', 'Visitors', 'Guest Count', 'Pax',
                'Number of Customers', 'Total Guests'
            ],
            'date': [
                'Date', 'Day', 'Time', 'Timestamp', 'Period',
                'Date of Service', 'Service Date', 'Business Date',
                'Transaction Date', 'Order Date', 'When'
            ],
            'special_event': [
                'Event Type', 'Event', 'Special Event', 'Occasion',
                'Function', 'Celebration', 'Event Name', 'Type of Event',
                'Special Occasion', 'Booking Type'
            ],
            'weather': [
                'Weather', 'Weather Condition', 'Climate', 'Weather Type',
                'Conditions', 'Weather Status', 'Forecast'
            ],
            'temperature': [
                'Temperature', 'Temp', 'Degrees', 'Celsius', 'Fahrenheit',
                'Ambient Temperature', 'Temp (C)', 'Temp (F)'
            ],
            'unit_cost': [
                'Pricing', 'Price', 'Cost', 'Unit Cost', 'Cost per Unit',
                'Price per kg', 'Cost per kg', 'Unit Price', 'Rate'
            ],
            'food_category': [
                'Category', 'Food Category', 'Type', 'Food Group',
                'Menu Category', 'Section', 'Course', 'Food Type'
            ]
        }
        
        # Additional feature mappings for extra columns
        self.extra_features = {
            'storage_conditions': ['Storage Conditions', 'Storage', 'Storage Type', 'Storage Method'],
            'preparation_method': ['Preparation Method', 'Prep Method', 'Cooking Method', 'Preparation'],
            'purchase_history': ['Purchase History', 'Purchase', 'Buying History', 'Order History'],
            'seasonality': ['Seasonality', 'Season', 'Seasonal', 'Time of Year'],
            'location': ['Geographical Location', 'Location', 'Region', 'Area', 'City', 'Branch']
        }
        
        # Common date formats to try
        self.date_formats = [
            '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
            '%d-%m-%Y', '%m-%d-%Y', '%d.%m.%Y', '%Y.%m.%d',
            '%B %d, %Y', '%d %B %Y', '%Y-%m-%d %H:%M:%S'
        ]
    
    def detect_column_mapping(self, df_columns):
        """Intelligently detect and map columns from any CSV format"""
        
        print("\n🔍 INTELLIGENT COLUMN DETECTION")
        print("="*50)
        
        mapping_result = {}
        unmapped_columns = []
        
        for csv_col in df_columns:
            best_match = None
            best_score = 0
            best_category = None
            
            # Check standard mappings
            for our_col, variations in self.column_mappings.items():
                for variation in variations:
                    # Use fuzzy matching for intelligent detection
                    score = fuzz.ratio(csv_col.lower(), variation.lower())
                    if score > best_score:
                        best_score = score
                        best_match = our_col
                        best_category = 'standard'
            
            # Check extra features
            for extra_col, variations in self.extra_features.items():
                for variation in variations:
                    score = fuzz.ratio(csv_col.lower(), variation.lower())
                    if score > best_score:
                        best_score = score
                        best_match = extra_col
                        best_category = 'extra'
            
            # Map if confidence is high enough
            if best_score >= 70:  # 70% similarity threshold
                mapping_result[csv_col] = {
                    'maps_to': best_match,
                    'confidence': best_score,
                    'category': best_category
                }
                print(f"  ✅ '{csv_col}' → '{best_match}' (confidence: {best_score}%)")
            else:
                unmapped_columns.append(csv_col)
                print(f"  ⚠️ '{csv_col}' → No clear mapping found")
        
        return mapping_result, unmapped_columns
    
    def adapt_dataframe(self, df, mapping_result):
        """Transform any dataframe to our standard format"""
        
        print("\n🔄 ADAPTING DATA TO STANDARD FORMAT")
        print("="*50)
        
        adapted_df = pd.DataFrame()
        
        # Apply mappings
        for original_col, mapping_info in mapping_result.items():
            if original_col in df.columns:
                target_col = mapping_info['maps_to']
                adapted_df[target_col] = df[original_col]
                print(f"  ✅ Mapped: {original_col} → {target_col}")
        
        return adapted_df
    
    def infer_missing_columns(self, df):
        """Intelligently infer missing required columns"""
        
        print("\n🧠 INFERRING MISSING COLUMNS")
        print("="*50)
        
        # Check for date column
        if 'date' not in df.columns:
            # Try to find any column with dates
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        # Try parsing as date
                        test_date = pd.to_datetime(df[col], errors='coerce')
                        if test_date.notna().sum() > len(df) * 0.5:  # At least 50% valid dates
                            df['date'] = test_date
                            print(f"  ✅ Found dates in column: {col}")
                            break
                    except:
                        pass
            
            # If still no date, generate sequential dates
            if 'date' not in df.columns:
                print("  ⚠️ No date column found, generating sequential dates")
                start_date = datetime.now() - timedelta(days=len(df))
                df['date'] = pd.date_range(start=start_date, periods=len(df), freq='D')
        
        # Infer quantity_sold if missing
        if 'quantity_sold' not in df.columns:
            if 'quantity_prepared' in df.columns and 'quantity_wasted' in df.columns:
                df['quantity_sold'] = df['quantity_prepared'] - df['quantity_wasted']
                df['quantity_sold'] = df['quantity_sold'].clip(lower=0)
                print("  ✅ Calculated quantity_sold = prepared - wasted")
            else:
                # Estimate based on waste percentage
                if 'quantity_prepared' in df.columns:
                    df['quantity_sold'] = df['quantity_prepared'] * 0.8  # Assume 20% waste
                    print("  ✅ Estimated quantity_sold (80% of prepared)")
        
        # Add day_of_week from date
        if 'date' in df.columns and 'day_of_week' not in df.columns:
            df['day_of_week'] = pd.to_datetime(df['date']).dt.day_name()
            print("  ✅ Added day_of_week from date")
        
        # Add is_weekend
        if 'date' in df.columns and 'is_weekend' not in df.columns:
            df['is_weekend'] = (pd.to_datetime(df['date']).dt.dayofweek >= 5).astype(int)
            print("  ✅ Added is_weekend flag")
        
        # Add month
        if 'date' in df.columns and 'month' not in df.columns:
            df['month'] = pd.to_datetime(df['date']).dt.month
            print("  ✅ Added month from date")
        
        # Handle weather if missing
        if 'weather' not in df.columns:
            df['weather'] = 'sunny'  # Default
            print("  ✅ Added default weather (sunny)")
        
        # Handle temperature if missing
        if 'temperature' not in df.columns:
            # Estimate based on month if available
            if 'month' in df.columns:
                temp_by_month = {
                    1: 5, 2: 7, 3: 12, 4: 17, 5: 22, 6: 27,
                    7: 30, 8: 29, 9: 24, 10: 18, 11: 11, 12: 6
                }
                df['temperature'] = df['month'].map(temp_by_month)
            else:
                df['temperature'] = 20  # Default
            print("  ✅ Added temperature estimates")
        
        # Add food_category if missing
        if 'food_category' not in df.columns and 'food_item' in df.columns:
            df['food_category'] = self.categorize_food_items(df['food_item'])
            print("  ✅ Auto-categorized food items")
        
        return df
    
    def categorize_food_items(self, food_items):
        """Auto-categorize food items based on keywords"""
        
        categories = []
        
        category_keywords = {
            'mains': ['chicken', 'beef', 'pork', 'fish', 'steak', 'burger', 'pasta', 'pizza', 'rice', 'curry'],
            'desserts': ['cake', 'ice cream', 'pudding', 'pie', 'sweet', 'chocolate', 'dessert'],
            'salads': ['salad', 'greens', 'lettuce', 'vegetables'],
            'soups': ['soup', 'broth', 'stew', 'chowder'],
            'sides': ['fries', 'chips', 'bread', 'potatoes', 'sides'],
            'beverages': ['drink', 'juice', 'soda', 'coffee', 'tea', 'beverage'],
            'appetizers': ['appetizer', 'starter', 'snack', 'wings', 'nachos']
        }
        
        for item in food_items:
            item_lower = str(item).lower()
            assigned = False
            
            for category, keywords in category_keywords.items():
                for keyword in keywords:
                    if keyword in item_lower:
                        categories.append(category)
                        assigned = True
                        break
                if assigned:
                    break
            
            if not assigned:
                categories.append('others')
        
        return categories
    
    def process_csv(self, filepath):
        """Main function to process any CSV format"""
        
        print("\n" + "="*60)
        print("🔄 UNIVERSAL CSV ADAPTER")
        print("="*60)
        
        # Load CSV
        print(f"\n📂 Loading: {filepath}")
        df = pd.read_csv(filepath)
        print(f"  ✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Display original columns
        print(f"\n📊 Original columns: {df.columns.tolist()}")
        
        # Detect column mappings
        mapping_result, unmapped = self.detect_column_mapping(df.columns)
        
        # Adapt dataframe
        adapted_df = self.adapt_dataframe(df, mapping_result)
        
        # Keep extra features as additional columns
        for original_col, mapping_info in mapping_result.items():
            if mapping_info['category'] == 'extra':
                # Keep extra features for enhanced predictions
                adapted_df[f"extra_{mapping_info['maps_to']}"] = df[original_col]
        
        # Infer missing columns
        adapted_df = self.infer_missing_columns(adapted_df)
        
        # Validate final dataframe
        required_cols = ['date', 'food_item', 'quantity_wasted']
        missing = [col for col in required_cols if col not in adapted_df.columns]
        
        if missing:
            print(f"\n⚠️ Warning: Still missing critical columns: {missing}")
        else:
            print("\n✅ All required columns present!")
        
        # Save adapted data
        output_path = 'data/raw/adapted_data.csv'
        os.makedirs('data/raw', exist_ok=True)
        adapted_df.to_csv(output_path, index=False)
        print(f"\n💾 Adapted data saved to: {output_path}")
        
        return adapted_df, mapping_result