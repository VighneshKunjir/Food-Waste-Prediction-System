# train_custom.py
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer

# Load your own data
preprocessor = DataPreprocessor()
X_train, X_test, y_train, y_test, _ = preprocessor.prepare_data('your_data.csv')

# Train models
trainer = ModelTrainer()
results = trainer.train_all_models(X_train, y_train)