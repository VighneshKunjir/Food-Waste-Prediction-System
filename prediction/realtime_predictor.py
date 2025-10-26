# prediction/realtime_predictor.py
"""Real-time interactive predictions"""

import pandas as pd
from datetime import datetime
from .predictor import BasePredictor


class RealtimePredictor(BasePredictor):
    """Real-time interactive predictions"""
    
    def __init__(self, model_path=None, preprocessor_path=None):
        super().__init__(model_path, preprocessor_path)
    
    def interactive_prediction(self):
        """Interactive CLI for real-time predictions"""
        print("\n" + "="*60)
        print("⚡ REAL-TIME WASTE PREDICTION")
        print("="*60)
        print("Enter 'quit' to exit\n")
        
        while True:
            try:
                print("\n" + "-"*60)
                print("Enter prediction details:")
                
                # Get food item
                food_item = input("Food item: ").strip()
                if food_item.lower() == 'quit':
                    break
                
                # Get date
                date_input = input("Date (YYYY-MM-DD) or press Enter for today: ").strip()
                if date_input.lower() == 'quit':
                    break
                
                if not date_input:
                    date = datetime.now().strftime('%Y-%m-%d')
                else:
                    date = date_input
                
                # Get other inputs
                input_data = self._collect_input_data(food_item, date)
                
                if input_data is None:  # User quit
                    break
                
                # Make prediction
                result = self.predict_with_confidence(input_data)
                
                # Display results
                self._display_prediction_results(input_data, result)
                
                # Ask for another prediction
                another = input("\n🔄 Make another prediction? (y/n): ").strip().lower()
                if another != 'y':
                    break
                
            except KeyboardInterrupt:
                print("\n\n👋 Exiting...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again with valid inputs.")
        
        print("\n👋 Thank you for using the prediction system!")
    
    def _collect_input_data(self, food_item, date):
        """Collect all input data from user"""
        try:
            # Food category
            food_category = input("Food category (mains/desserts/sides/salads/soups): ").strip()
            if food_category.lower() == 'quit':
                return None
            if not food_category:
                food_category = 'mains'
            
            # Quantity prepared
            qty_prep = input("Quantity prepared (kg): ").strip()
            if qty_prep.lower() == 'quit':
                return None
            quantity_prepared = float(qty_prep) if qty_prep else 50.0
            
            # Expected quantity sold
            qty_sold = input("Expected quantity sold (kg): ").strip()
            if qty_sold.lower() == 'quit':
                return None
            quantity_sold = float(qty_sold) if qty_sold else 40.0
            
            # Day of week (auto-detect from date)
            date_obj = pd.to_datetime(date)
            day_of_week = date_obj.strftime('%A')
            is_weekend = int(date_obj.weekday() >= 5)
            month = date_obj.month
            
            # Weather
            print("\nWeather condition:")
            print("  1. Sunny  2. Cloudy  3. Rainy  4. Stormy")
            weather_choice = input("Choose (1-4) or press Enter for 'sunny': ").strip()
            if weather_choice.lower() == 'quit':
                return None
            
            weather_map = {'1': 'sunny', '2': 'cloudy', '3': 'rainy', '4': 'stormy'}
            weather = weather_map.get(weather_choice, 'sunny')
            
            # Temperature
            temp_input = input("Temperature (°C) or press Enter for 20°C: ").strip()
            if temp_input.lower() == 'quit':
                return None
            temperature = float(temp_input) if temp_input else 20.0
            
            # Special event
            special_event = input("Special event (wedding/birthday/none): ").strip()
            if special_event.lower() == 'quit':
                return None
            if not special_event:
                special_event = 'none'
            
            # Customer count
            customers = input("Expected customers: ").strip()
            if customers.lower() == 'quit':
                return None
            customer_count = int(customers) if customers else 150
            
            # Unit cost
            cost = input("Unit cost ($) or press Enter for $8.50: ").strip()
            if cost.lower() == 'quit':
                return None
            unit_cost = float(cost) if cost else 8.5
            
            # Build input dictionary
            input_data = {
                'date': date,
                'food_item': food_item,
                'food_category': food_category,
                'quantity_prepared': quantity_prepared,
                'quantity_sold': quantity_sold,
                'day_of_week': day_of_week,
                'is_weekend': is_weekend,
                'weather': weather,
                'temperature': temperature,
                'special_event': special_event,
                'customer_count': customer_count,
                'month': month,
                'unit_cost': unit_cost
            }
            
            return input_data
            
        except ValueError as e:
            print(f"❌ Invalid input: {e}")
            return None
    
    def _display_prediction_results(self, input_data, result):
        """Display prediction results in formatted way"""
        print("\n" + "="*60)
        print("📊 PREDICTION RESULTS")
        print("="*60)
        
        predicted_waste = result['predictions'][0]
        lower = result['lower_bound'][0]
        upper = result['upper_bound'][0]
        
        print(f"\n🎯 Predicted Waste: {predicted_waste:.2f} kg")
        print(f"📊 Confidence Interval: [{lower:.2f} - {upper:.2f}] kg")
        print(f"💰 Estimated Waste Cost: ${predicted_waste * input_data['unit_cost']:.2f}")
        
        # Waste percentage
        waste_pct = (predicted_waste / input_data['quantity_prepared']) * 100
        print(f"📉 Waste Percentage: {waste_pct:.1f}%")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        print("-"*60)
        
        if waste_pct > 25:
            print("   ⚠️ HIGH WASTE ALERT!")
            print(f"   • Reduce preparation by {predicted_waste * 0.3:.1f} kg")
            print(f"   • Potential savings: ${predicted_waste * 0.3 * input_data['unit_cost']:.2f}")
        elif waste_pct > 15:
            print("   ⚡ MODERATE WASTE")
            print(f"   • Consider reducing preparation by {predicted_waste * 0.2:.1f} kg")
            print("   • Monitor closely over next few days")
        else:
            print("   ✅ WASTE WITHIN ACCEPTABLE RANGE")
            print("   • Current preparation levels are appropriate")
        
        # Efficiency rating
        efficiency = (input_data['quantity_sold'] / input_data['quantity_prepared']) * 100
        print(f"\n📈 Preparation Efficiency: {efficiency:.1f}%")
        
        if efficiency < 70:
            print("   ⚠️ Low efficiency - consider reducing preparation")
        elif efficiency > 90:
            print("   ⚠️ Very high efficiency - risk of stockouts")
        else:
            print("   ✅ Good efficiency level")
    
    def quick_predict(self, food_item, quantity_prepared, quantity_sold, 
                     customer_count=150, weather='sunny'):
        """Quick prediction with minimal inputs"""
        
        input_data = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'food_item': food_item,
            'food_category': 'mains',
            'quantity_prepared': quantity_prepared,
            'quantity_sold': quantity_sold,
            'day_of_week': datetime.now().strftime('%A'),
            'is_weekend': int(datetime.now().weekday() >= 5),
            'weather': weather,
            'temperature': 20,
            'special_event': 'none',
            'customer_count': customer_count,
            'month': datetime.now().month,
            'unit_cost': 8.5
        }
        
        result = self.predict_with_confidence(input_data)
        
        return {
            'food_item': food_item,
            'predicted_waste': result['predictions'][0],
            'confidence_range': (result['lower_bound'][0], result['upper_bound'][0]),
            'estimated_cost': result['predictions'][0] * input_data['unit_cost']
        }