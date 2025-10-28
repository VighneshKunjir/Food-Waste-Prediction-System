# core/data_generator.py
"""Sample data generation for testing"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os


class DataGenerator:
    """Generate realistic sample waste data"""
    
    def __init__(self):
        self.food_items = {
            'Grilled Chicken': 'mains',
            'Paneer': 'mains',
            'Pasta': 'mains',
            'Margherita Pizza': 'mains',
            'Chocolate Cake': 'desserts',
            'Cake': 'desserts',
            'French Fries': 'sides',
            'Garlic Bread': 'sides',
            'Corn Soup': 'soups',
            'Burger': 'mains',
            'Fish': 'mains',
            'Salad': 'salads',
            'Ice Cream': 'desserts',
            'Onion Rings': 'sides',
            'Tomato Soup': 'soups'
        }
        
        self.weather_conditions = ['sunny', 'rainy', 'cloudy', 'stormy']
    
    def generate(self, n_days=365, save_path=None, create_backup=False):
        """
        Generate sample data for n_days
        
        Args:
            n_days: Number of days to generate data for
            save_path: Path where to save the CSV file
            create_backup: If True, creates timestamped backup
        
        Returns:
            DataFrame with generated data
        """
        print("\n" + "="*70)
        print("📊 GENERATING SAMPLE FOOD WASTE DATA")
        print("="*70)
        print(f"\n🔄 Generating {n_days} days of sample data...")
        
        data = []
        start_date = datetime.now() - timedelta(days=n_days)
        
        for day in range(n_days):
            current_date = start_date + timedelta(days=day)
            day_of_week = current_date.strftime('%A')
            
            # Weekend effect
            is_weekend = current_date.weekday() >= 5
            base_customers = 200 if is_weekend else 150
            
            # Seasonal variation
            month = current_date.month
            seasonal_multiplier = self._get_seasonal_multiplier(month)
            
            # Daily customers
            daily_customers = int(base_customers * seasonal_multiplier * np.random.uniform(0.7, 1.3))
            
            # Weather impact
            weather = random.choice(self.weather_conditions)
            if weather == 'rainy':
                daily_customers = int(daily_customers * 0.8)
            elif weather == 'stormy':
                daily_customers = int(daily_customers * 0.6)
            
            temperature = np.random.normal(20, 8)
            
            # Special events
            special_event = self._generate_special_event()
            if special_event != 'none':
                daily_customers = int(daily_customers * 1.5)
            
            # Generate for each food item
            for item, category in self.food_items.items():
                record = self._generate_item_record(
                    current_date, item, category, day_of_week,
                    is_weekend, weather, temperature, special_event,
                    daily_customers, month
                )
                data.append(record)
        
        df = pd.DataFrame(data)
        
        # Display summary statistics
        self._display_summary(df, n_days)
        
        # Save to file
        if save_path:
            self._save_data(df, save_path, create_backup)
        else:
            print("\n⚠️  No save path provided - data not saved to disk")
        
        return df
    
    def _display_summary(self, df, n_days):
        """Display summary statistics of generated data"""
        print("\n" + "─"*70)
        print("📊 DATA SUMMARY")
        print("─"*70)
        
        print(f"\n✅ Successfully generated data:")
        print(f"   Total records: {len(df):,}")
        print(f"   Days covered: {n_days}")
        print(f"   Food items: {df['food_item'].nunique()}")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        
        print(f"\n📈 Waste Statistics:")
        print(f"   Average daily waste: {df.groupby('date')['quantity_wasted'].sum().mean():.2f} kg")
        print(f"   Total waste: {df['quantity_wasted'].sum():.2f} kg")
        print(f"   Average waste cost: ${df['waste_cost'].mean():.2f}")
        print(f"   Total waste cost: ${df['waste_cost'].sum():.2f}")
        
        print(f"\n🍽️  Top 3 Wasted Items:")
        top_wasted = df.groupby('food_item')['quantity_wasted'].sum().sort_values(ascending=False).head(3)
        for i, (item, waste) in enumerate(top_wasted.items(), 1):
            print(f"   {i}. {item}: {waste:.2f} kg")
    
    def _save_data(self, df, save_path, create_backup=False):
        """Save data to CSV with proper directory creation"""
        print("\n" + "─"*70)
        print("💾 SAVING DATA")
        print("─"*70)
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        
        # Save main file
        df.to_csv(save_path, index=False)
        file_size = os.path.getsize(save_path) / 1024  # Size in KB
        
        print(f"\n✅ Data saved successfully!")
        print(f"   📁 Location: {os.path.abspath(save_path)}")
        print(f"   📊 Records: {len(df):,}")
        print(f"   💾 File size: {file_size:.2f} KB")
        
        # Create backup if requested
        if create_backup:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = save_path.replace('.csv', f'_backup_{timestamp}.csv')
            df.to_csv(backup_path, index=False)
            print(f"   🔄 Backup: {backup_path}")
        
        # Validate saved file
        try:
            test_df = pd.read_csv(save_path)
            if len(test_df) == len(df):
                print(f"\n✅ File validation passed - data saved correctly")
            else:
                print(f"\n⚠️  Warning: Saved file has different row count")
        except Exception as e:
            print(f"\n❌ Error validating saved file: {e}")
    
    def _get_seasonal_multiplier(self, month):
        """Get seasonal multiplier"""
        if month in [6, 7, 8]:  # Summer
            return 1.3
        elif month in [11, 12]:  # Holiday
            return 1.4
        elif month in [1, 2]:  # Winter
            return 0.8
        else:
            return 1.0
    
    def _generate_special_event(self):
        """Generate special event"""
        if random.random() < 0.05:  # 5% chance
            return random.choice(['wedding', 'birthday', 'corporate', 'festival'])
        return 'none'
    
    def _generate_item_record(self, date, item, category, day_of_week,
                             is_weekend, weather, temp, event, customers, month):
        """Generate single item record"""
        category_multipliers = {
            'mains': 0.4,
            'desserts': 0.2,
            'sides': 0.25,
            'salads': 0.15,
            'soups': 0.1
        }
        
        base_prep = customers * category_multipliers.get(category, 0.2)
        item_popularity = np.random.uniform(0.7, 1.3)
        quantity_prepared = base_prep * item_popularity
        
        # Sales rate based on conditions
        if weather == 'stormy' and category == 'soups':
            sales_rate = np.random.uniform(0.85, 0.95)
        elif weather == 'sunny' and category == 'salads':
            sales_rate = np.random.uniform(0.80, 0.90)
        else:
            sales_rate = np.random.uniform(0.65, 0.85)
        
        quantity_sold = quantity_prepared * sales_rate
        quantity_wasted = max(0, quantity_prepared - quantity_sold)
        quantity_wasted = quantity_wasted * np.random.uniform(0.9, 1.1)
        
        unit_cost = np.random.uniform(3, 15)
        
        return {
            'date': date.strftime('%Y-%m-%d'),
            'food_item': item,
            'food_category': category,
            'quantity_prepared': round(quantity_prepared, 2),
            'quantity_sold': round(quantity_sold, 2),
            'quantity_wasted': round(quantity_wasted, 2),
            'day_of_week': day_of_week,
            'is_weekend': int(is_weekend),
            'weather': weather,
            'temperature': round(temp, 1),
            'special_event': event,
            'customer_count': customers,
            'month': month,
            'unit_cost': round(unit_cost, 2),
            'total_cost': round(quantity_prepared * unit_cost, 2),
            'waste_cost': round(quantity_wasted * unit_cost, 2)
        }
    
    def generate_custom(self, n_days=365, food_items=None, save_path=None):
        """
        Generate data with custom food items
        
        Args:
            n_days: Number of days
            food_items: Dictionary of {item_name: category}
            save_path: Where to save
        """
        if food_items:
            original_items = self.food_items.copy()
            self.food_items = food_items
            
            df = self.generate(n_days, save_path)
            
            self.food_items = original_items  # Restore
            return df
        else:
            return self.generate(n_days, save_path)