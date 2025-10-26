# workflows/prediction_workflow.py
"""Complete prediction workflow orchestration"""

from datetime import datetime

from prediction.predictor import BasePredictor
from prediction.future_predictor import FuturePredictor
from prediction.realtime_predictor import RealtimePredictor

from storage.model_storage import ModelStorage
from storage.prediction_storage import PredictionStorage
from storage.restaurant_manager import RestaurantManager

from utils.logger import Logger
from utils.visualization import Visualizer


class PredictionWorkflow:
    """Orchestrate prediction workflows"""
    
    def __init__(self, restaurant_name):
        """
        Initialize prediction workflow
        
        Args:
            restaurant_name: Name of restaurant
        """
        self.restaurant_name = restaurant_name
        
        # Initialize components
        self.logger = Logger(name=f'Prediction_{restaurant_name}')
        self.model_storage = ModelStorage()
        self.prediction_storage = PredictionStorage()
        self.restaurant_manager = RestaurantManager()
        self.visualizer = Visualizer()
        
        # Load model and preprocessor
        self._load_model_and_preprocessor()
        
        # Get training info
        self._load_training_info()

        self.logger.info(f"Initialized prediction workflow for {restaurant_name}")

    def _load_training_info(self):
        """Load information about how the model was trained"""
        try:
            metadata = self.model_storage.load_metadata(self.restaurant_name, 'best')
            if metadata:
                self.training_info = {
                    'model_name': metadata.get('model_name', 'Unknown'),
                    'neural_network_included': metadata.get('neural_network_included', True),
                    'use_gpu': metadata.get('use_gpu', False),
                    'total_models_trained': metadata.get('total_models_trained', 0),
                    'training_date': metadata.get('training_date', 'Unknown')
                }
                
                print(f"📋 Model Info: {self.training_info['model_name']}")
                print(f"   Trained: {self.training_info['training_date']}")
                print(f"   GPU: {self.training_info['use_gpu']}, NN: {self.training_info['neural_network_included']}")
            else:
                self.training_info = None
        except:
            self.training_info = None
    
    def _load_model_and_preprocessor(self):
        """Load trained model and preprocessor"""
        try:
            model_path = f'data/restaurants/{self.restaurant_name}/models/best_model.pkl'
            preprocessor_path = f'data/restaurants/{self.restaurant_name}/models/preprocessor.pkl'
            
            self.model = self.model_storage.load_model(self.restaurant_name, 'best')
            self.preprocessor = self.model_storage.load_preprocessor(self.restaurant_name)
            
            print(f"✅ Loaded model and preprocessor for {self.restaurant_name}")
            
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            self.model = None
            self.preprocessor = None
    
    def run_realtime_prediction(self, interactive=True):
        """
        Run real-time prediction workflow
        
        Args:
            interactive: If True, runs interactive CLI
        
        Returns:
            Prediction results
        """
        print("\n" + "="*70)
        print(f"⚡ REAL-TIME PREDICTION: {self.restaurant_name}")
        print("="*70)
        
        if self.model is None:
            print("❌ No trained model found. Please train a model first.")
            return None
        
        try:
            # Initialize predictor
            predictor = RealtimePredictor()
            predictor.model = self.model
            predictor.preprocessor = self.preprocessor
            
            if interactive:
                # Run interactive prediction
                predictor.interactive_prediction()
            else:
                # Return predictor for programmatic use
                return predictor
            
            self.logger.info("Real-time prediction completed")
            
        except Exception as e:
            self.logger.log_error_with_traceback(e)
            print(f"❌ Prediction failed: {e}")
            return None
    
    def run_future_prediction(self, prediction_type='single', **kwargs):
        """
        Run future prediction workflow
        
        Args:
            prediction_type: 'single', 'multiple', or 'weekly'
            **kwargs: Additional arguments for prediction
        
        Returns:
            Prediction results
        """
        print("\n" + "="*70)
        print(f"🔮 FUTURE PREDICTION: {self.restaurant_name}")
        print("="*70)
        
        if self.model is None:
            print("❌ No trained model found. Please train a model first.")
            return None
        
        try:
            # Initialize predictor
            model_path = f'data/restaurants/{self.restaurant_name}/models/best_model.pkl'
            preprocessor_path = f'data/restaurants/{self.restaurant_name}/models/preprocessor.pkl'
            
            predictor = FuturePredictor(
                model_path=model_path,
                preprocessor_path=preprocessor_path
            )
            
            results = None
            
            if prediction_type == 'single':
                # Single day prediction
                date = kwargs.get('date')
                food_item = kwargs.get('food_item')
                
                if date and food_item:
                    result = predictor.predict_single_day(
                        date=date,
                        food_item=food_item,
                        **{k: v for k, v in kwargs.items() if k not in ['date', 'food_item']}
                    )
                    
                    print(f"\n📊 Prediction for {food_item} on {date}:")
                    print(f"   Predicted waste: {result['predictions'][0]:.2f} kg")
                    print(f"   Range: [{result['lower_bound'][0]:.2f} - {result['upper_bound'][0]:.2f}] kg")
                    
                    results = result
                else:
                    print("⚠️ Please provide 'date' and 'food_item' for single prediction")
            
            elif prediction_type == 'multiple':
                # Multiple items for a single day
                date = kwargs.get('date')
                food_items = kwargs.get('food_items')
                
                if date:
                    predictions_df = predictor.predict_multiple_items(
                        date=date,
                        food_items=food_items,
                        **{k: v for k, v in kwargs.items() if k not in ['date', 'food_items']}
                    )
                    
                    print(f"\n📊 Predictions for {date}:")
                    print(predictions_df.to_string(index=False))
                    
                    # Generate recommendations
                    predictor.generate_recommendations(predictions_df)
                    
                    # Save predictions
                    self.prediction_storage.save_prediction(
                        restaurant_name=self.restaurant_name,
                        prediction_data=predictions_df,
                        prediction_type='future'
                    )
                    
                    results = predictions_df
                else:
                    print("⚠️ Please provide 'date' for multiple predictions")
            
            elif prediction_type == 'weekly':
                # Weekly forecast
                start_date = kwargs.get('start_date')
                
                weekly_df = predictor.predict_week_ahead(start_date=start_date)
                
                # Visualize
                plot_path = f'results/plots/{self.restaurant_name}_weekly_forecast.png'
                self.visualizer.plot_weekly_forecast(weekly_df, save_path=plot_path)
                
                # Save predictions
                self.prediction_storage.save_prediction(
                    restaurant_name=self.restaurant_name,
                    prediction_data=weekly_df,
                    prediction_type='future'
                )
                
                results = weekly_df
            
            # Update metadata
            self._update_prediction_metadata()
            
            self.logger.info(f"Future prediction completed: {prediction_type}")
            
            return results
            
        except Exception as e:
            self.logger.log_error_with_traceback(e)
            print(f"❌ Prediction failed: {e}")
            return None
    
    def run_batch_prediction(self, csv_path, output_path=None):
        """
        Run batch prediction on CSV file
        
        Args:
            csv_path: Path to input CSV
            output_path: Path to save predictions (optional)
        
        Returns:
            DataFrame with predictions
        """
        print("\n" + "="*70)
        print(f"📦 BATCH PREDICTION: {self.restaurant_name}")
        print("="*70)
        
        if self.model is None:
            print("❌ No trained model found. Please train a model first.")
            return None
        
        try:
            # Initialize predictor
            predictor = BasePredictor()
            predictor.model = self.model
            predictor.preprocessor = self.preprocessor
            
            # Generate output path if not provided
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f'data/restaurants/{self.restaurant_name}/predictions/batch_predictions_{timestamp}.csv'
            
            # Run batch prediction
            results_df = predictor.batch_predict(csv_path, output_path)
            
            # Save to prediction storage
            self.prediction_storage.save_prediction(
                restaurant_name=self.restaurant_name,
                prediction_data=results_df,
                prediction_type='realtime'
            )
            
            # Update metadata
            self._update_prediction_metadata()
            
            self.logger.info(f"Batch prediction completed: {len(results_df)} records")
            
            return results_df
            
        except Exception as e:
            self.logger.log_error_with_traceback(e)
            print(f"❌ Batch prediction failed: {e}")
            return None
    
    def view_prediction_history(self, prediction_type='realtime', limit=10):
        """
        View prediction history
        
        Args:
            prediction_type: 'realtime' or 'future'
            limit: Number of recent predictions to show
        """
        print("\n" + "="*70)
        print(f"📜 PREDICTION HISTORY: {self.restaurant_name} ({prediction_type.upper()})")
        print("="*70)
        
        history = self.prediction_storage.get_prediction_history(
            restaurant_name=self.restaurant_name,
            prediction_type=prediction_type,
            limit=limit
        )
        
        if not history:
            print(f"\n⚠️ No {prediction_type} predictions found")
            return
        
        print(f"\nShowing {len(history)} most recent predictions:\n")
        
        for i, pred in enumerate(history, 1):
            print(f"{i}. {pred['datetime']}")
            print(f"   Timestamp: {pred['timestamp']}")
            
            if isinstance(pred['data'], list) and len(pred['data']) > 0:
                first_item = pred['data'][0]
                if 'predicted_waste' in first_item:
                    print(f"   Sample: {first_item.get('food_item', 'N/A')} - {first_item['predicted_waste']:.2f} kg")
            
            print()
    
    def _update_prediction_metadata(self):
        """Update restaurant metadata after predictions"""
        summary = self.prediction_storage.get_prediction_summary(self.restaurant_name)
        
        self.restaurant_manager.update_restaurant_metadata(
            self.restaurant_name,
            {
                'total_predictions': summary['total_predictions'],
                'last_prediction': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        )