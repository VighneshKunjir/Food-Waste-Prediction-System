# core/adapter.py
"""Universal data adapter for any CSV format"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fuzzywuzzy import fuzz


class UniversalAdapter:
    """Adapt any CSV format to standard format"""
    
    def __init__(self):
        self.column_mappings = {
            'food_item': [
                'Type of Food', 'Food Type', 'Item', 'Menu Item', 'Food Name',
                'Product', 'Dish', 'Food', 'Item Name', 'Product Name'
            ],
            'quantity_wasted': [
                'Wastage Food Amount', 'Waste', 'Wasted', 'Food Waste',
                'Waste Amount', 'Wastage', 'Thrown Away', 'Discarded'
            ],
            'quantity_prepared': [
                'Quantity of Food', 'Prepared', 'Made', 'Cooked',
                'Production', 'Quantity Prepared', 'Total Prepared'
            ],
            'quantity_sold': [
                'Sold', 'Sales', 'Consumed', 'Used', 'Served'
            ],
            'customer_count': [
                'Number of Guests', 'Customers', 'Guests', 'Covers'
            ],
            'date': [
                'Date', 'Day', 'Time', 'Timestamp', 'Period'
            ],
            'special_event': [
                'Event Type', 'Event', 'Special Event', 'Occasion'
            ],
            'weather': [
                'Weather', 'Weather Condition', 'Climate'
            ],
            'temperature': [
                'Temperature', 'Temp', 'Degrees', 'Celsius'
            ],
            'unit_cost': [
                'Pricing', 'Price', 'Cost', 'Unit Cost'
            ],
            'food_category': [
                'Category', 'Food Category', 'Type', 'Food Group'
            ]
        }
        
        self.fuzzy_threshold = 70  # 70% similarity
    
    def detect_and_adapt(self, df):
        """Detect columns and adapt to standard format"""
        print("\n🔍 DETECTING COLUMN MAPPINGS")
        print("="*50)
        
        mapping_result = {}
        unmapped = []
        
        for csv_col in df.columns:
            best_match = None
            best_score = 0
            
            for std_col, variations in self.column_mappings.items():
                for variation in variations:
                    score = fuzz.ratio(csv_col.lower(), variation.lower())
                    if score > best_score:
                        best_score = score
                        best_match = std_col
            
            if best_score >= self.fuzzy_threshold:
                mapping_result[csv_col] = {
                    'maps_to': best_match,
                    'confidence': best_score
                }
                print(f"  ✅ '{csv_col}' → '{best_match}' ({best_score}%)")
            else:
                unmapped.append(csv_col)
                print(f"  ⚠️ '{csv_col}' → No mapping")
        
        # Apply mappings
        adapted_df = self._apply_mappings(df, mapping_result)
        
        # Infer missing columns
        adapted_df = self._infer_missing(adapted_df)
        
        print(f"\n✅ Adaptation complete")
        print(f"   Mapped columns: {len(mapping_result)}")
        print(f"   Unmapped columns: {len(unmapped)}")
        
        return adapted_df, mapping_result
    
    def _apply_mappings(self, df, mapping_result):
        """Apply column mappings"""
        adapted_df = pd.DataFrame()
        
        for original_col, mapping_info in mapping_result.items():
            if original_col in df.columns:
                target_col = mapping_info['maps_to']
                adapted_df[target_col] = df[original_col]
        
        return adapted_df
    
    def _infer_missing(self, df):
        """Infer missing required columns"""
        print("\n🧠 INFERRING MISSING COLUMNS")
        
        # Infer date
        if 'date' not in df.columns:
            start_date = datetime.now() - timedelta(days=len(df))
            df['date'] = pd.date_range(start=start_date, periods=len(df), freq='D')
            print("  ✅ Added date column")
        
        # Infer quantity_sold
        if 'quantity_sold' not in df.columns:
            if 'quantity_prepared' in df.columns and 'quantity_wasted' in df.columns:
                df['quantity_sold'] = df['quantity_prepared'] - df['quantity_wasted']
                print("  ✅ Calculated quantity_sold")
            elif 'quantity_prepared' in df.columns:
                df['quantity_sold'] = df['quantity_prepared'] * 0.8
                print("  ✅ Estimated quantity_sold")
        
        # Add derived columns
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_week'] = df['date'].dt.day_name()
            df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
            df['month'] = df['date'].dt.month
            print("  ✅ Added date-derived columns")
        
        # Default values for optional columns
        if 'weather' not in df.columns:
            df['weather'] = 'sunny'
        
        if 'temperature' not in df.columns:
            df['temperature'] = 20
        
        if 'special_event' not in df.columns:
            df['special_event'] = 'none'
        
        if 'customer_count' not in df.columns:
            df['customer_count'] = 100
        
        # Auto-categorize food items
        if 'food_category' not in df.columns and 'food_item' in df.columns:
            df['food_category'] = self._auto_categorize(df['food_item'])
            print("  ✅ Auto-categorized food items")
        
        return df
    
    def _auto_categorize(self, food_items):
        """Auto-categorize food items"""
        categories = []
        
        keywords = {
            'mains': ['chicken', 'beef', 'fish', 'burger', 'pasta', 'pizza'],
            'desserts': ['cake', 'ice cream', 'pudding', 'sweet'],
            'salads': ['salad', 'greens'],
            'soups': ['soup', 'broth'],
            'sides': ['fries', 'bread', 'potatoes']
        }
        
        for item in food_items:
            item_lower = str(item).lower()
            assigned = False
            
            for category, kws in keywords.items():
                if any(kw in item_lower for kw in kws):
                    categories.append(category)
                    assigned = True
                    break
            
            if not assigned:
                categories.append('others')
        
        return categories