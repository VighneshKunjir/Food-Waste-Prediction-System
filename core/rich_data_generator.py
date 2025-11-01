# core/rich_data_generator.py
"""
Enhanced Data Generator for Research-Quality Food Waste Dataset

Generates realistic, rich food waste data with:
- Complex temporal patterns (trends, seasonality, cycles)
- Multi-modal distributions
- Realistic correlations between features
- External factors (weather, events, holidays)
- Restaurant-specific characteristics
- Controllable noise and anomalies

For research and benchmarking purposes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import os


class RichDataGenerator:
    """
    Generate research-quality food waste dataset with realistic patterns
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize generator
        
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.seed = seed
        
        # Restaurant types with different characteristics
        self.restaurant_types = {
            'fast_food': {
                'base_customers': 200,
                'customer_variance': 0.4,
                'waste_rate': 0.15,
                'price_sensitivity': 0.3,
                'weather_sensitivity': 0.2
            },
            'casual_dining': {
                'base_customers': 120,
                'customer_variance': 0.3,
                'waste_rate': 0.18,
                'price_sensitivity': 0.2,
                'weather_sensitivity': 0.4
            },
            'fine_dining': {
                'base_customers': 60,
                'customer_variance': 0.25,
                'waste_rate': 0.25,
                'price_sensitivity': 0.1,
                'weather_sensitivity': 0.5
            },
            'cafe': {
                'base_customers': 150,
                'customer_variance': 0.35,
                'waste_rate': 0.12,
                'price_sensitivity': 0.25,
                'weather_sensitivity': 0.6
            }
        }
        
        # Comprehensive food catalog
        self.food_catalog = {
            'mains': {
                'Grilled Chicken': {'popularity': 0.9, 'price': 12, 'prep_variance': 0.2},
                'Beef Steak': {'popularity': 0.7, 'price': 18, 'prep_variance': 0.25},
                'Salmon Fillet': {'popularity': 0.6, 'price': 16, 'prep_variance': 0.3},
                'Pasta Carbonara': {'popularity': 0.85, 'price': 10, 'prep_variance': 0.15},
                'Vegetable Curry': {'popularity': 0.5, 'price': 9, 'prep_variance': 0.2},
                'Margherita Pizza': {'popularity': 0.95, 'price': 11, 'prep_variance': 0.15},
                'Burger': {'popularity': 0.9, 'price': 10, 'prep_variance': 0.18},
                'Fish Tacos': {'popularity': 0.6, 'price': 11, 'prep_variance': 0.22},
                'Paneer Tikka': {'popularity': 0.55, 'price': 10, 'prep_variance': 0.2},
                'BBQ Ribs': {'popularity': 0.65, 'price': 15, 'prep_variance': 0.25}
            },
            'sides': {
                'French Fries': {'popularity': 0.95, 'price': 4, 'prep_variance': 0.15},
                'Garlic Bread': {'popularity': 0.8, 'price': 3, 'prep_variance': 0.12},
                'Coleslaw': {'popularity': 0.6, 'price': 3, 'prep_variance': 0.18},
                'Onion Rings': {'popularity': 0.7, 'price': 4, 'prep_variance': 0.2},
                'Mashed Potatoes': {'popularity': 0.65, 'price': 3, 'prep_variance': 0.15}
            },
            'salads': {
                'Caesar Salad': {'popularity': 0.75, 'price': 7, 'prep_variance': 0.2},
                'Greek Salad': {'popularity': 0.65, 'price': 7, 'prep_variance': 0.22},
                'Garden Salad': {'popularity': 0.7, 'price': 6, 'prep_variance': 0.18}
            },
            'soups': {
                'Tomato Soup': {'popularity': 0.7, 'price': 5, 'prep_variance': 0.15},
                'Corn Soup': {'popularity': 0.75, 'price': 5, 'prep_variance': 0.15},
                'Mushroom Soup': {'popularity': 0.6, 'price': 6, 'prep_variance': 0.18}
            },
            'desserts': {
                'Chocolate Cake': {'popularity': 0.85, 'price': 6, 'prep_variance': 0.2},
                'Ice Cream': {'popularity': 0.9, 'price': 5, 'prep_variance': 0.15},
                'Tiramisu': {'popularity': 0.65, 'price': 7, 'prep_variance': 0.25},
                'Cheesecake': {'popularity': 0.75, 'price': 6, 'prep_variance': 0.22},
                'Brownie': {'popularity': 0.8, 'price': 5, 'prep_variance': 0.18}
            }
        }
        
        # Weather patterns by season
        self.weather_patterns = {
            'winter': {'temp_mean': 5, 'temp_std': 8, 'rain_prob': 0.3},
            'spring': {'temp_mean': 15, 'temp_std': 7, 'rain_prob': 0.25},
            'summer': {'temp_mean': 25, 'temp_std': 6, 'rain_prob': 0.15},
            'fall': {'temp_mean': 12, 'temp_std': 7, 'rain_prob': 0.28}
        }
        
        # Special events calendar
        self.special_events = {
            'new_year': {'month': 1, 'day': 1, 'impact': 1.8},
            'valentines': {'month': 2, 'day': 14, 'impact': 1.5},
            'st_patricks': {'month': 3, 'day': 17, 'impact': 1.3},
            'easter': {'month': 4, 'day': 15, 'impact': 1.4},  # Approximate
            'mothers_day': {'month': 5, 'day': 12, 'impact': 1.6},
            'independence_day': {'month': 7, 'day': 4, 'impact': 1.5},
            'halloween': {'month': 10, 'day': 31, 'impact': 1.3},
            'thanksgiving': {'month': 11, 'day': 25, 'impact': 2.0},
            'christmas': {'month': 12, 'day': 25, 'impact': 1.9},
            'new_year_eve': {'month': 12, 'day': 31, 'impact': 1.7}
        }
    
    def generate(
        self,
        n_days: int = 730,  # 2 years default for research
        restaurant_type: str = 'casual_dining',
        start_date: Optional[datetime] = None,
        add_trends: bool = True,
        add_anomalies: bool = True,
        anomaly_rate: float = 0.02,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate rich food waste dataset
        
        Args:
            n_days: Number of days to generate
            restaurant_type: Type of restaurant
            start_date: Start date (default: 2 years before now)
            add_trends: Add long-term trends
            add_anomalies: Add occasional anomalies
            anomaly_rate: Fraction of anomalous days
            save_path: Path to save CSV
            
        Returns:
            DataFrame with rich food waste data
        """
        print("\n" + "="*70)
        print("📊 GENERATING RICH FOOD WASTE DATASET")
        print("="*70)
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=n_days)
        
        # Get restaurant characteristics
        if restaurant_type not in self.restaurant_types:
            print(f"⚠️ Unknown restaurant type '{restaurant_type}', using 'casual_dining'")
            restaurant_type = 'casual_dining'
        
        rest_char = self.restaurant_types[restaurant_type]
        
        print(f"\n🏪 Restaurant Type: {restaurant_type}")
        print(f"📅 Date Range: {start_date.strftime('%Y-%m-%d')} to "
              f"{(start_date + timedelta(days=n_days-1)).strftime('%Y-%m-%d')}")
        print(f"🍽️ Total Items: {sum(len(items) for items in self.food_catalog.values())}")
        print(f"📝 Total Records: {n_days * sum(len(items) for items in self.food_catalog.values()):,}")
        
        # Generate daily data
        data = []
        
        # Global trend (if enabled)
        if add_trends:
            # Simulate gradual waste reduction over time (from awareness/training)
            trend_factor = np.linspace(1.0, 0.85, n_days)  # 15% improvement over period
        else:
            trend_factor = np.ones(n_days)
        
        # Anomaly days
        if add_anomalies:
            n_anomalies = int(n_days * anomaly_rate)
            anomaly_days = set(np.random.choice(n_days, n_anomalies, replace=False))
        else:
            anomaly_days = set()
        
        for day_idx in range(n_days):
            current_date = start_date + timedelta(days=day_idx)
            
            # Temporal features
            day_features = self._get_day_features(current_date, day_idx, n_days)
            
            # Weather
            weather_features = self._generate_weather(current_date)
            
            # Customer count
            customers = self._generate_customer_count(
                day_features, weather_features, rest_char, 
                is_anomaly=(day_idx in anomaly_days)
            )
            
            # Special event
            event_info = self._check_special_event(current_date)
            if event_info['is_special']:
                customers = int(customers * event_info['impact'])
            
            # Generate records for each food item
            for category, items in self.food_catalog.items():
                for item_name, item_info in items.items():
                    record = self._generate_item_record(
                        current_date=current_date,
                        item_name=item_name,
                        category=category,
                        item_info=item_info,
                        customers=customers,
                        day_features=day_features,
                        weather_features=weather_features,
                        event_info=event_info,
                        rest_char=rest_char,
                        trend_factor=trend_factor[day_idx],
                        is_anomaly=(day_idx in anomaly_days)
                    )
                    
                    data.append(record)
            
            # Progress indicator
            if (day_idx + 1) % 100 == 0 or (day_idx + 1) == n_days:
                print(f"   Generated: {day_idx + 1}/{n_days} days", end='\r')
        
        print()  # New line after progress
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add derived features
        df = self._add_derived_features(df)
        
        # Display summary
        self._display_summary(df)
        
        # Save if requested
        if save_path:
            self._save_dataset(df, save_path, restaurant_type, n_days)
        
        return df
    
    def _get_day_features(self, date: datetime, day_idx: int, total_days: int) -> Dict:
        """Extract temporal features from date"""
        return {
            'day_of_week': date.strftime('%A'),
            'is_weekend': date.weekday() >= 5,
            'month': date.month,
            'day_of_month': date.day,
            'week_of_year': date.isocalendar()[1],
            'quarter': (date.month - 1) // 3 + 1,
            'season': self._get_season(date.month),
            'is_month_start': date.day <= 7,
            'is_month_end': date.day >= 23,
            'days_since_start': day_idx,
            'progress': day_idx / total_days
        }
    
    def _get_season(self, month: int) -> str:
        """Get season from month"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def _generate_weather(self, date: datetime) -> Dict:
        """Generate realistic weather for the date"""
        season = self._get_season(date.month)
        pattern = self.weather_patterns[season]
        
        # Temperature with seasonal variation
        temp = np.random.normal(pattern['temp_mean'], pattern['temp_std'])
        
        # Weather condition
        rain_roll = np.random.random()
        if rain_roll < pattern['rain_prob'] * 0.3:
            weather = 'stormy'
            temp -= 3
        elif rain_roll < pattern['rain_prob']:
            weather = 'rainy'
            temp -= 1
        elif rain_roll < pattern['rain_prob'] + 0.3:
            weather = 'cloudy'
        else:
            weather = 'sunny'
            temp += 1
        
        # Humidity (correlated with rain)
        if weather in ['rainy', 'stormy']:
            humidity = np.random.uniform(70, 95)
        elif weather == 'cloudy':
            humidity = np.random.uniform(50, 75)
        else:
            humidity = np.random.uniform(30, 60)
        
        return {
            'weather': weather,
            'temperature': round(temp, 1),
            'humidity': round(humidity, 1)
        }
    
    def _generate_customer_count(
        self, 
        day_features: Dict, 
        weather_features: Dict,
        rest_char: Dict,
        is_anomaly: bool = False
    ) -> int:
        """Generate realistic customer count"""
        base = rest_char['base_customers']
        
        # Weekend effect
        if day_features['is_weekend']:
            base *= 1.3
        
        # Seasonal multiplier
        season_mult = {
            'winter': 0.85,
            'spring': 1.0,
            'summer': 1.25,
            'fall': 0.95
        }
        base *= season_mult[day_features['season']]
        
        # Weather effect
        if weather_features['weather'] == 'stormy':
            base *= 0.6
        elif weather_features['weather'] == 'rainy':
            base *= 0.8
        elif weather_features['weather'] == 'sunny' and weather_features['temperature'] > 20:
            base *= 1.1
        
        # Month-end effect (pay day)
        if day_features['is_month_end']:
            base *= 1.15
        
        # Random variation
        customers = int(base * np.random.uniform(
            1 - rest_char['customer_variance'],
            1 + rest_char['customer_variance']
        ))
        
        # Anomaly (e.g., unexpected closure, food poisoning scare, etc.)
        if is_anomaly:
            customers = int(customers * np.random.uniform(0.2, 0.6))
        
        return max(10, customers)  # Minimum 10 customers
    
    def _check_special_event(self, date: datetime) -> Dict:
        """Check if date is a special event"""
        for event_name, event_info in self.special_events.items():
            if date.month == event_info['month'] and date.day == event_info['day']:
                return {
                    'is_special': True,
                    'event_name': event_name,
                    'impact': event_info['impact']
                }
        
        # Check if it's near a special event (±2 days)
        for event_name, event_info in self.special_events.items():
            event_date = datetime(date.year, event_info['month'], event_info['day'])
            days_diff = abs((date - event_date).days)
            
            if 1 <= days_diff <= 2:
                # Nearby event (smaller impact)
                return {
                    'is_special': True,
                    'event_name': f'near_{event_name}',
                    'impact': 1 + (event_info['impact'] - 1) * 0.3
                }
        
        return {'is_special': False, 'event_name': 'none', 'impact': 1.0}
    
    def _generate_item_record(
        self,
        current_date: datetime,
        item_name: str,
        category: str,
        item_info: Dict,
        customers: int,
        day_features: Dict,
        weather_features: Dict,
        event_info: Dict,
        rest_char: Dict,
        trend_factor: float,
        is_anomaly: bool
    ) -> Dict:
        """Generate a single food item record"""
        
        # Base preparation quantity
        category_multipliers = {
            'mains': 0.4,
            'sides': 0.25,
            'salads': 0.15,
            'soups': 0.12,
            'desserts': 0.2
        }
        
        base_prep = customers * category_multipliers[category]
        
        # Item popularity effect
        prep_mult = item_info['popularity'] * np.random.uniform(
            1 - item_info['prep_variance'],
            1 + item_info['prep_variance']
        )
        
        quantity_prepared = base_prep * prep_mult
        
        # Sales rate (how much gets sold)
        base_sales_rate = 0.75
        
        # Weather effects on specific categories
        if weather_features['weather'] == 'sunny' and category == 'salads':
            base_sales_rate += 0.15
        elif weather_features['weather'] in ['rainy', 'stormy'] and category == 'soups':
            base_sales_rate += 0.12
        elif weather_features['temperature'] > 25 and category == 'desserts' and 'Ice Cream' in item_name:
            base_sales_rate += 0.18
        
        # Weekend effect on desserts
        if day_features['is_weekend'] and category == 'desserts':
            base_sales_rate += 0.1
        
        # Special event effect
        if event_info['is_special']:
            base_sales_rate += 0.08
        
        # Add noise
        sales_rate = np.clip(
            base_sales_rate * np.random.uniform(0.9, 1.1),
            0.5, 0.98
        )
        
        # Calculate quantities
        quantity_sold = quantity_prepared * sales_rate
        quantity_wasted = quantity_prepared - quantity_sold
        
        # Apply trend factor (waste reduction over time)
        quantity_wasted *= trend_factor
        
        # Anomaly effect (e.g., preparation error, storage issue)
        if is_anomaly:
            quantity_wasted *= np.random.uniform(1.5, 3.0)
        
        # Ensure non-negative
        quantity_wasted = max(0, quantity_wasted)
        quantity_sold = max(0, quantity_prepared - quantity_wasted)
        
        # Costs
        unit_cost = item_info['price'] * 0.4  # Cost is ~40% of price
        total_cost = quantity_prepared * unit_cost
        waste_cost = quantity_wasted * unit_cost
        revenue = quantity_sold * item_info['price']
        profit = revenue - total_cost
        
        # Staff and operational features
        day_hour_coverage = 12 if day_features['is_weekend'] else 10
        staff_count = max(3, int(customers / 20))
        
        return {
            # Date features
            'date': current_date.strftime('%Y-%m-%d'),
            'day_of_week': day_features['day_of_week'],
            'is_weekend': int(day_features['is_weekend']),
            'month': day_features['month'],
            'day_of_month': day_features['day_of_month'],
            'week_of_year': day_features['week_of_year'],
            'quarter': day_features['quarter'],
            'season': day_features['season'],
            'is_month_start': int(day_features['is_month_start']),
            'is_month_end': int(day_features['is_month_end']),
            
            # Item features
            'food_item': item_name,
            'food_category': category,
            'item_price': item_info['price'],
            'item_popularity': item_info['popularity'],
            
            # Quantities
            'quantity_prepared': round(quantity_prepared, 2),
            'quantity_sold': round(quantity_sold, 2),
            'quantity_wasted': round(quantity_wasted, 2),
            'waste_percentage': round(100 * quantity_wasted / max(quantity_prepared, 0.01), 2),
            
            # Financial
            'unit_cost': round(unit_cost, 2),
            'total_cost': round(total_cost, 2),
            'waste_cost': round(waste_cost, 2),
            'revenue': round(revenue, 2),
            'profit': round(profit, 2),
            'profit_margin': round(100 * profit / max(revenue, 0.01), 2),
            
            # Operational
            'customer_count': customers,
            'staff_count': staff_count,
            'operating_hours': day_hour_coverage,
            
            # Weather
            'weather': weather_features['weather'],
            'temperature': weather_features['temperature'],
            'humidity': weather_features['humidity'],
            
            # Events
            'special_event': event_info['event_name'],
            'is_special_event': int(event_info['is_special']),
            
            # Temporal
            'days_since_start': day_features['days_since_start']
        }
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for richer analysis"""
        
        # Rolling averages (7-day)
        df = df.sort_values('date')
        
        for item in df['food_item'].unique():
            item_mask = df['food_item'] == item
            df.loc[item_mask, 'waste_7day_avg'] = df.loc[item_mask, 'quantity_wasted'].rolling(7, min_periods=1).mean()
            df.loc[item_mask, 'sales_7day_avg'] = df.loc[item_mask, 'quantity_sold'].rolling(7, min_periods=1).mean()
        
        # Lag features (yesterday's waste)
        for item in df['food_item'].unique():
            item_mask = df['food_item'] == item
            df.loc[item_mask, 'waste_yesterday'] = df.loc[item_mask, 'quantity_wasted'].shift(1)
        
        # Fill NaN in lag features
        df['waste_yesterday'].fillna(df['quantity_wasted'].mean(), inplace=True)
        
        return df
    
    def _display_summary(self, df: pd.DataFrame):
        """Display dataset summary statistics"""
        print("\n" + "="*70)
        print("📊 DATASET SUMMARY")
        print("="*70)
        
        print(f"\n✅ Total Records: {len(df):,}")
        print(f"📅 Date Range: {df['date'].min()} to {df['date'].max()}")
        print(f"🍽️ Unique Items: {df['food_item'].nunique()}")
        print(f"📂 Categories: {', '.join(df['food_category'].unique())}")
        
        print(f"\n📊 Waste Statistics:")
        print(f"   Total Waste: {df['quantity_wasted'].sum():,.2f} kg")
        print(f"   Avg Daily Waste: {df.groupby('date')['quantity_wasted'].sum().mean():.2f} kg")
        print(f"   Avg Item Waste: {df['quantity_wasted'].mean():.2f} kg")
        print(f"   Waste Std Dev: {df['quantity_wasted'].std():.2f} kg")
        print(f"   Min Waste: {df['quantity_wasted'].min():.2f} kg")
        print(f"   Max Waste: {df['quantity_wasted'].max():.2f} kg")
        
        print(f"\n💰 Financial Summary:")
        print(f"   Total Waste Cost: ${df['waste_cost'].sum():,.2f}")
        print(f"   Avg Daily Waste Cost: ${df.groupby('date')['waste_cost'].sum().mean():.2f}")
        print(f"   Total Revenue: ${df['revenue'].sum():,.2f}")
        print(f"   Total Profit: ${df['profit'].sum():,.2f}")
        
        print(f"\n🏆 Top 5 Wasted Items:")
        top_waste = df.groupby('food_item')['quantity_wasted'].sum().sort_values(ascending=False).head(5)
        for i, (item, waste) in enumerate(top_waste.items(), 1):
            cost = df[df['food_item'] == item]['waste_cost'].sum()
            print(f"   {i}. {item}: {waste:.2f} kg (${cost:.2f})")
        
        print(f"\n📈 Data Quality:")
        print(f"   Missing Values: {df.isnull().sum().sum()}")
        print(f"   Duplicate Rows: {df.duplicated().sum()}")
        print(f"   Columns: {len(df.columns)}")
        
        print("="*70)
    
    def _save_dataset(self, df: pd.DataFrame, save_path: str, restaurant_type: str, n_days: int):
        """Save dataset with metadata"""
        # Create directory
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save CSV
        df.to_csv(save_path, index=False)
        
        # Get file size
        file_size = os.path.getsize(save_path) / 1024  # KB
        
        print(f"\n💾 Dataset saved:")
        print(f"   📁 Location: {os.path.abspath(save_path)}")
        print(f"   📊 Records: {len(df):,}")
        print(f"   💾 Size: {file_size:.2f} KB")
        
        # Save metadata
        metadata_path = save_path.replace('.csv', '_metadata.json')
        metadata = {
            'generated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_days': n_days,
            'n_records': len(df),
            'restaurant_type': restaurant_type,
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max()
            },
            'statistics': {
                'total_waste_kg': float(df['quantity_wasted'].sum()),
                'total_waste_cost': float(df['waste_cost'].sum()),
                'avg_daily_waste': float(df.groupby('date')['quantity_wasted'].sum().mean()),
                'unique_items': int(df['food_item'].nunique())
            },
            'features': list(df.columns),
            'seed': self.seed
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   📋 Metadata: {metadata_path}")
    
    def generate_multiple_restaurants(
        self,
        n_restaurants: int = 10,
        n_days: int = 730,
        base_path: str = 'data/research'
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate datasets for multiple restaurants (for research)
        
        Args:
            n_restaurants: Number of restaurants
            n_days: Days per restaurant
            base_path: Base directory for saving
            
        Returns:
            Dictionary mapping restaurant names to DataFrames
        """
        print("\n" + "="*70)
        print(f"🏪 GENERATING {n_restaurants} RESTAURANT DATASETS")
        print("="*70)
        
        datasets = {}
        restaurant_types = list(self.restaurant_types.keys())
        
        for i in range(n_restaurants):
            restaurant_name = f"Restaurant_{i+1:02d}"
            restaurant_type = restaurant_types[i % len(restaurant_types)]
            
            print(f"\n{'='*70}")
            print(f"Restaurant {i+1}/{n_restaurants}: {restaurant_name} ({restaurant_type})")
            print(f"{'='*70}")
            
            # Use different seed for each restaurant
            self.seed = 42 + i
            np.random.seed(self.seed)
            
            save_path = f"{base_path}/{restaurant_name}/data.csv"
            
            df = self.generate(
                n_days=n_days,
                restaurant_type=restaurant_type,
                save_path=save_path
            )
            
            datasets[restaurant_name] = df
        
        print("\n" + "="*70)
        print(f"✅ Generated {n_restaurants} restaurant datasets")
        print(f"📁 Saved to: {base_path}/")
        print("="*70)
        
        return datasets