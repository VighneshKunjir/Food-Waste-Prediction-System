# src/future_prediction.py
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class FutureWastePredictor:
    def __init__(self, model_path='data/models/best_waste_model.pkl', 
                 preprocessor_path='data/models/preprocessor.pkl'):
        """Initialize future waste predictor"""
        self.model = None
        self.preprocessor = None
        self.food_items = None
        self.food_categories = None
        
        # Load model and preprocessor
        try:
            self.model = joblib.load(model_path)
            self.preprocessor = joblib.load(preprocessor_path)
            print("✅ Model and preprocessor loaded successfully")
        except:
            print("⚠️ Could not load saved model. Please train the model first.")
        
        # Load historical data for reference
        self.load_historical_data()
    
    def load_historical_data(self):
        """Load historical data to get food items and patterns"""
        try:
            df = pd.read_csv('data/raw/food_waste_data.csv')
            
            # Get unique food items and categories
            self.food_items = df['food_item'].unique().tolist()
            self.food_categories = df.groupby('food_item')['food_category'].first().to_dict()
            
            # Calculate average patterns for future predictions
            self.avg_patterns = {
                'customer_count_by_day': df.groupby('day_of_week')['customer_count'].mean().to_dict(),
                'waste_by_item': df.groupby('food_item')['quantity_wasted'].mean().to_dict(),
                'preparation_by_item': df.groupby('food_item')['quantity_prepared'].mean().to_dict(),
                'sales_by_item': df.groupby('food_item')['quantity_sold'].mean().to_dict(),
                'monthly_patterns': df.groupby('month')['customer_count'].mean().to_dict()
            }
            
            print(f"📊 Loaded data for {len(self.food_items)} food items")
            
        except Exception as e:
            print(f"⚠️ Could not load historical data: {e}")
            self.food_items = ['Grilled Chicken', 'Caesar Salad', 'Pasta Carbonara', 
                              'Margherita Pizza', 'Chocolate Cake']
            self.food_categories = {item: 'mains' for item in self.food_items}
    
    def get_user_input(self):
        """Get prediction parameters from user"""
        print("\n" + "="*60)
        print("🔮 FUTURE FOOD WASTE PREDICTION")
        print("="*60)
        
        # Get date
        print("\n📅 Enter Future Date Information:")
        date_str = input("   Date (YYYY-MM-DD) or 'tomorrow'/'+7' for days ahead: ").strip()
        
        # Parse date
        if date_str.lower() == 'tomorrow':
            future_date = datetime.now() + timedelta(days=1)
        elif date_str.startswith('+'):
            days_ahead = int(date_str[1:])
            future_date = datetime.now() + timedelta(days=days_ahead)
        else:
            try:
                future_date = datetime.strptime(date_str, '%Y-%m-%d')
            except:
                print("⚠️ Invalid date format. Using tomorrow.")
                future_date = datetime.now() + timedelta(days=1)
        
        # Get food item (optional)
        print("\n🍔 Food Item Selection:")
        print("   Available items:")
        for i, item in enumerate(self.food_items, 1):
            print(f"      {i}. {item}")
        print("      0. Predict for ALL items")
        
        choice = input("\n   Enter number (or press Enter for ALL): ").strip()
        
        if choice == '' or choice == '0':
            selected_food = None  # Predict for all
            print("   ✅ Will predict for all food items")
        else:
            try:
                idx = int(choice) - 1
                selected_food = self.food_items[idx]
                print(f"   ✅ Selected: {selected_food}")
            except:
                selected_food = None
                print("   ✅ Will predict for all food items")
        
        # Get additional parameters
        print("\n📊 Additional Parameters (press Enter for auto-estimate):")
        
        # Expected customers
        customers_input = input("   Expected customers (or Enter for auto): ").strip()
        if customers_input:
            expected_customers = int(customers_input)
        else:
            # Auto-estimate based on day of week
            day_name = future_date.strftime('%A')
            expected_customers = int(self.avg_patterns['customer_count_by_day'].get(day_name, 150))
            print(f"   Auto-estimated: {expected_customers} customers for {day_name}")
        
        # Weather
        print("\n   Weather forecast:")
        print("      1. Sunny")
        print("      2. Cloudy")
        print("      3. Rainy")
        print("      4. Stormy")
        weather_choice = input("   Enter choice (or Enter for 'sunny'): ").strip()
        
        weather_map = {'1': 'sunny', '2': 'cloudy', '3': 'rainy', '4': 'stormy'}
        weather = weather_map.get(weather_choice, 'sunny')
        print(f"   Weather: {weather}")
        
        # Special event
        special_event = input("   Special event (wedding/birthday/none): ").strip()
        if not special_event:
            special_event = 'none'
        
        # Temperature
        temp_input = input("   Temperature °C (or Enter for seasonal average): ").strip()
        if temp_input:
            temperature = float(temp_input)
        else:
            # Seasonal temperature estimate
            month = future_date.month
            if month in [12, 1, 2]:
                temperature = 10  # Winter
            elif month in [3, 4, 5]:
                temperature = 18  # Spring
            elif month in [6, 7, 8]:
                temperature = 28  # Summer
            else:
                temperature = 20  # Fall
            print(f"   Auto-estimated temperature: {temperature}°C")
        
        return {
            'date': future_date,
            'food_item': selected_food,
            'expected_customers': expected_customers,
            'weather': weather,
            'temperature': temperature,
            'special_event': special_event
        }
    
    def prepare_prediction_data(self, date, food_item, customers, weather, temperature, special_event):
        """Prepare data for prediction"""
        # Extract date features
        day_of_week = date.strftime('%A')
        is_weekend = 1 if date.weekday() >= 5 else 0
        month = date.month
        
        # Get historical averages for the food item
        if food_item in self.avg_patterns['preparation_by_item']:
            avg_prep = self.avg_patterns['preparation_by_item'][food_item]
            avg_sales = self.avg_patterns['sales_by_item'][food_item]
        else:
            avg_prep = 50
            avg_sales = 40
        
        # Adjust based on expected customers
        customer_factor = customers / 150  # Assuming 150 is average
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
        
        # Create feature dictionary
        features = {
            'date': date.strftime('%Y-%m-%d'),
            'food_item': food_item,
            'food_category': self.food_categories.get(food_item, 'mains'),
            'quantity_prepared': quantity_prepared,
            'quantity_sold': quantity_sold,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'weather': weather,
            'temperature': temperature,
            'special_event': special_event,
            'customer_count': customers,
            'month': month,
            'unit_cost': np.random.uniform(5, 15)  # Estimated cost
        }
        
        return features
    
    def predict_single_item(self, params):
        """Predict waste for a single food item"""
        # Prepare features
        features = self.prepare_prediction_data(
            params['date'],
            params['food_item'],
            params['expected_customers'],
            params['weather'],
            params['temperature'],
            params['special_event']
        )
        
        # Create dataframe
        df = pd.DataFrame([features])
        
        # Preprocess
        df_processed = self.preprocessor.prepare_prediction_data(df)
        
        # Predict
        prediction = self.model.predict(df_processed)[0]
        
        # Ensure non-negative
        prediction = max(0, prediction)
        
        # Calculate confidence interval (simplified)
        confidence_margin = prediction * 0.15  # 15% margin
        
        return {
            'food_item': params['food_item'],
            'predicted_waste': prediction,
            'lower_bound': max(0, prediction - confidence_margin),
            'upper_bound': prediction + confidence_margin,
            'estimated_cost': prediction * features['unit_cost'],
            'preparation': features['quantity_prepared'],
            'expected_sales': features['quantity_sold']
        }
    
    def predict_all_items(self, params):
        """Predict waste for all food items"""
        predictions = []
        
        for food_item in self.food_items:
            params_copy = params.copy()
            params_copy['food_item'] = food_item
            
            result = self.predict_single_item(params_copy)
            predictions.append(result)
        
        return predictions
    
    def display_predictions(self, predictions, date):
        """Display predictions in a formatted way"""
        print("\n" + "="*60)
        print(f"📊 WASTE PREDICTIONS FOR {date.strftime('%A, %B %d, %Y')}")
        print("="*60)
        
        if isinstance(predictions, dict):
            # Single item prediction
            predictions = [predictions]
        
        # Create dataframe for display
        df = pd.DataFrame(predictions)
        
        # Display each prediction
        total_waste = 0
        total_cost = 0
        
        for pred in predictions:
            print(f"\n🍽️ {pred['food_item']}:")
            print(f"   Predicted Waste: {pred['predicted_waste']:.2f} kg")
            print(f"   Confidence Range: [{pred['lower_bound']:.2f} - {pred['upper_bound']:.2f}] kg")
            print(f"   Estimated Cost: ${pred['estimated_cost']:.2f}")
            print(f"   Preparation: {pred['preparation']:.2f} kg | Expected Sales: {pred['expected_sales']:.2f} kg")
            
            total_waste += pred['predicted_waste']
            total_cost += pred['estimated_cost']
        
        # Summary
        print("\n" + "-"*60)
        print(f"📈 SUMMARY:")
        print(f"   Total Predicted Waste: {total_waste:.2f} kg")
        print(f"   Total Estimated Cost: ${total_cost:.2f}")
        print(f"   Number of Items: {len(predictions)}")
        print(f"   Average Waste per Item: {total_waste/len(predictions):.2f} kg")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        
        # Find high waste items
        high_waste = [p for p in predictions if p['predicted_waste'] > 10]
        if high_waste:
            print("   ⚠️ High waste predicted for:")
            for item in high_waste:
                reduction = item['predicted_waste'] * 0.2
                print(f"      - {item['food_item']}: Consider reducing preparation by {reduction:.1f} kg")
        
        # Low waste items
        low_waste = [p for p in predictions if p['predicted_waste'] < 2]
        if low_waste:
            print("   ✅ Low waste expected for:")
            for item in low_waste:
                print(f"      - {item['food_item']}")
        
        return df
    
    def visualize_predictions(self, predictions_df, date):
        """Create visualization for predictions"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Food Waste Predictions for {date.strftime("%A, %B %d, %Y")}', 
                     fontsize=16, fontweight='bold')
        
        # Sort by predicted waste
        predictions_df = predictions_df.sort_values('predicted_waste', ascending=False)
        
        # 1. Predicted Waste by Item
        ax1 = axes[0, 0]
        bars1 = ax1.barh(predictions_df['food_item'], predictions_df['predicted_waste'], 
                         color='coral', edgecolor='darkred')
        ax1.set_xlabel('Predicted Waste (kg)')
        ax1.set_title('Predicted Waste by Food Item')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add values
        for bar, val in zip(bars1, predictions_df['predicted_waste']):
            ax1.text(val + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{val:.1f}', va='center')
        
        # 2. Waste Cost Distribution
        ax2 = axes[0, 1]
        colors = plt.cm.RdYlGn_r(predictions_df['estimated_cost'] / predictions_df['estimated_cost'].max())
        bars2 = ax2.bar(range(len(predictions_df)), predictions_df['estimated_cost'], 
                       color=colors, edgecolor='black')
        ax2.set_xticks(range(len(predictions_df)))
        ax2.set_xticklabels(predictions_df['food_item'], rotation=45, ha='right')
        ax2.set_ylabel('Estimated Cost ($)')
        ax2.set_title('Waste Cost by Item')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Preparation vs Expected Sales vs Waste
        ax3 = axes[1, 0]
        x = np.arange(len(predictions_df))
        width = 0.25
        
        bars3_1 = ax3.bar(x - width, predictions_df['preparation'], width, 
                         label='Preparation', color='lightblue')
        bars3_2 = ax3.bar(x, predictions_df['expected_sales'], width, 
                         label='Expected Sales', color='lightgreen')
        bars3_3 = ax3.bar(x + width, predictions_df['predicted_waste'], width, 
                         label='Predicted Waste', color='salmon')
        
        ax3.set_xlabel('Food Items')
        ax3.set_ylabel('Quantity (kg)')
        ax3.set_title('Preparation vs Sales vs Waste')
        ax3.set_xticks(x)
        ax3.set_xticklabels(predictions_df['food_item'], rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Waste Percentage
        ax4 = axes[1, 1]
        waste_percentage = (predictions_df['predicted_waste'] / predictions_df['preparation']) * 100
        
        # Create pie chart of top 5 waste items
        top_5 = predictions_df.head(5)
        sizes = top_5['predicted_waste'].values
        labels = [f"{item}\n{waste:.1f} kg" for item, waste in 
                 zip(top_5['food_item'].values, top_5['predicted_waste'].values)]
        
        colors_pie = plt.cm.Reds(np.linspace(0.4, 0.8, len(top_5)))
        ax4.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Top 5 Waste Contributors')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f'data/plots/future_prediction_{date.strftime("%Y%m%d")}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved: {plot_path}")
        
        plt.show()
        
        return fig
    
    def predict_week_ahead(self):
        """Predict waste for the next 7 days"""
        print("\n" + "="*60)
        print("📅 WEEKLY WASTE FORECAST")
        print("="*60)
        
        weekly_predictions = []
        
        for days_ahead in range(1, 8):
            date = datetime.now() + timedelta(days=days_ahead)
            day_name = date.strftime('%A')
            
            # Estimate parameters based on historical patterns
            expected_customers = int(self.avg_patterns['customer_count_by_day'].get(day_name, 150))
            
            # Predict for all items
            daily_total = 0
            for food_item in self.food_items[:5]:  # Top 5 items for summary
                params = {
                    'date': date,
                    'food_item': food_item,
                    'expected_customers': expected_customers,
                    'weather': 'sunny',  # Default
                    'temperature': 20,    # Default
                    'special_event': 'none'
                }
                
                result = self.predict_single_item(params)
                daily_total += result['predicted_waste']
            
            weekly_predictions.append({
                'date': date.strftime('%Y-%m-%d'),
                'day': day_name,
                'total_waste': daily_total,
                'customer_count': expected_customers
            })
        
        # Display weekly forecast
        df_week = pd.DataFrame(weekly_predictions)
        
        print("\n📊 7-Day Waste Forecast:")
        print(df_week.to_string(index=False))
        
        print(f"\n📈 Weekly Summary:")
        print(f"   Total predicted waste: {df_week['total_waste'].sum():.2f} kg")
        print(f"   Average daily waste: {df_week['total_waste'].mean():.2f} kg")
        print(f"   Peak day: {df_week.loc[df_week['total_waste'].idxmax(), 'day']}")
        print(f"   Lowest day: {df_week.loc[df_week['total_waste'].idxmin(), 'day']}")
        
        return df_week
    
    def run_interactive_prediction(self):
        """Main interactive prediction interface"""
        while True:
            print("\n" + "="*60)
            print("🔮 FUTURE WASTE PREDICTION MODULE")
            print("="*60)
            print("\n1. Single Day Prediction")
            print("2. Weekly Forecast")
            print("3. Exit")
            
            choice = input("\nEnter choice (1-3): ").strip()
            
            if choice == '1':
                # Get user input
                params = self.get_user_input()
                
                # Make predictions
                if params['food_item'] is None:
                    # Predict for all items
                    predictions = self.predict_all_items(params)
                else:
                    # Predict for single item
                    predictions = self.predict_single_item(params)
                
                # Display results
                df_predictions = self.display_predictions(predictions, params['date'])
                
                # Visualize
                if len(df_predictions) > 1:
                    self.visualize_predictions(df_predictions, params['date'])
                
                # Ask for another prediction
                another = input("\n🔄 Make another prediction? (y/n): ").strip().lower()
                if another != 'y':
                    break
                    
            elif choice == '2':
                # Weekly forecast
                self.predict_week_ahead()
                
                another = input("\n🔄 Return to menu? (y/n): ").strip().lower()
                if another != 'y':
                    break
                    
            elif choice == '3':
                print("\n👋 Exiting prediction module...")
                break
            else:
                print("⚠️ Invalid choice. Please try again.")