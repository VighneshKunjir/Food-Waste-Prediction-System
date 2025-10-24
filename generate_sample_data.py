# generate_sample_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_sample_data(n_days=365):
    """Generate realistic sample waste data for demo"""
    
    # Ensure data directory exists
    os.makedirs('data/raw', exist_ok=True)
    
    # Food items and their categories
    food_items = {
        'Grilled Chicken': 'mains',
        'Paneer': 'mains',
        'Pasta ': 'mains',
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
    
    weather_conditions = ['sunny', 'rainy', 'cloudy', 'stormy']
    
    data = []
    start_date = datetime.now() - timedelta(days=n_days)
    
    for day in range(n_days):
        current_date = start_date + timedelta(days=day)
        day_of_week = current_date.strftime('%A')
        
        # Weekend effect on customer count
        is_weekend = current_date.weekday() >= 5
        base_customers = 200 if is_weekend else 150
        
        # Add seasonal variation
        month = current_date.month
        if month in [6, 7, 8]:  # Summer
            seasonal_multiplier = 1.3
        elif month in [11, 12]:  # Holiday season
            seasonal_multiplier = 1.4
        elif month in [1, 2]:  # Winter low season
            seasonal_multiplier = 0.8
        else:
            seasonal_multiplier = 1.0
        
        # Calculate daily customers with randomness
        daily_customers = int(base_customers * seasonal_multiplier * np.random.uniform(0.7, 1.3))
        
        # Weather impact
        weather = random.choice(weather_conditions)
        if weather == 'rainy':
            daily_customers = int(daily_customers * 0.8)
        elif weather == 'stormy':
            daily_customers = int(daily_customers * 0.6)
        
        temperature = np.random.normal(20, 8)  # Average temp with variation
        
        # Special events
        special_event = None
        if random.random() < 0.05:  # 5% chance of special event
            special_event = random.choice(['wedding', 'birthday', 'corporate', 'festival'])
            daily_customers = int(daily_customers * 1.5)
        
        for item, category in food_items.items():
            # Base preparation based on category popularity
            category_multipliers = {
                'mains': 0.4,
                'desserts': 0.2,
                'sides': 0.25,
                'salads': 0.15,
                'soups': 0.1
            }
            
            base_prep = daily_customers * category_multipliers.get(category, 0.2)
            
            # Add item-specific popularity variation
            item_popularity = np.random.uniform(0.7, 1.3)
            quantity_prepared = base_prep * item_popularity
            
            # Sales affected by various factors
            if weather == 'stormy' and category == 'soups':
                sales_rate = np.random.uniform(0.85, 0.95)  # Soups sell better in bad weather
            elif weather == 'sunny' and category == 'salads':
                sales_rate = np.random.uniform(0.80, 0.90)  # Salads sell better in good weather
            else:
                sales_rate = np.random.uniform(0.65, 0.85)
            
            quantity_sold = quantity_prepared * sales_rate
            
            # Calculate waste
            quantity_wasted = max(0, quantity_prepared - quantity_sold)
            
            # Add some noise
            quantity_wasted = quantity_wasted * np.random.uniform(0.9, 1.1)
            
            # Calculate costs (per unit)
            unit_cost = np.random.uniform(3, 15)
            
            data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'food_item': item,
                'food_category': category,
                'quantity_prepared': round(quantity_prepared, 2),
                'quantity_sold': round(quantity_sold, 2),
                'quantity_wasted': round(quantity_wasted, 2),
                'day_of_week': day_of_week,
                'is_weekend': int(is_weekend),
                'weather': weather,
                'temperature': round(temperature, 1),
                'special_event': special_event if special_event else 'none',
                'customer_count': daily_customers,
                'month': month,
                'unit_cost': round(unit_cost, 2),
                'total_cost': round(quantity_prepared * unit_cost, 2),
                'waste_cost': round(quantity_wasted * unit_cost, 2)
            })
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_path = 'data/raw/food_waste_data.csv'
    df.to_csv(output_path, index=False)
    print(f"✅ Generated {len(df)} records")
    print(f"📁 Saved to: {output_path}")
    print(f"📊 Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"🗑️ Total waste: {df['quantity_wasted'].sum():.2f} units")
    print(f"💰 Total waste cost: ${df['waste_cost'].sum():,.2f}")
    
    return df

if __name__ == "__main__":
    df = generate_sample_data(365)
    print("\n📈 Sample Data Overview:")
    print(df.head())
    print("\n📊 Data Statistics:")
    print(df.describe())