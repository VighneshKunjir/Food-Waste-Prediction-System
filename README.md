
# 🍔 Food Waste Prediction System - GPU Accelerated
A comprehensive machine learning system for predicting food waste in restaurants using advanced ML models with GPU acceleration. This system helps restaurants reduce waste, optimize inventory, and save costs through accurate waste predictions.

## ✨Features
### Core Capabilities
- Multiple ML Models: Random Forest, XGBoost, LightGBM, Neural Networks
- GPU Acceleration: Leverages NVIDIA GPUs for 5-15x faster training
- Comprehensive Evaluation: Detailed metrics and visualizations for each model
- Future Predictions: Predict waste for upcoming days with interactive interface
- Visual Analytics: Automatic generation of performance plots and comparisons
- Model Persistence: Save and load trained models for production use

### Prediction Features
- Single-day waste prediction for specific items
- Bulk prediction for entire menu
- 7-day ahead forecasting
- Weather and special event adjustments
- Confidence intervals for predictions

### 💻 Software Requirement
- NVIDIA CUDA Toolkit 11.6+ (for GPU support)
- CUDA: 11.6 or higher
- Python: 3.10 or 3.11 (3.11 recommended)

## 🚀 Installation Guide
### Step 1: Set Up Python Environment
1.  Option A: Using Virtual Environment (Recommended)

	**Create virtual environment**

		python -m venv venv

	**Activate virtual environment**
	- On Windows:

			venv\Scripts\activate

	- On Linux/Mac:*

			source venv/bin/activate

	**Upgrade pip**

			python -m pip install --upgrade pip

2. **Option B: Using Conda**

		# Create conda environment
		conda create -n foodwaste python=3.11
		conda activate foodwaste

### Step 3: Install Dependencies
#### For CPU-Only Version:

		# Install basic requirements
		pip install -r requirements.txt
#### For GPU-Accelerated Version:
	
		# Install PyTorch with CUDA support
		pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

		# Install other requirements
		pip install pandas numpy scikit-learn matplotlib seaborn joblib scipy

		# Install GPU-accelerated libraries
		pip install xgboost lightgbm
### Step 4: Verify Installation
		# Test if everything is installed correctly python test_installation.py


## 📖 How to run / Usage Guide
### Step 1: Initial Setup

	# Navigate to project directory
	cd food-waste-prediction

	# Activate environment
	venv\Scripts\activate  # Windows
	source venv/bin/activate  # Linux/Mac
	
### Step 2. Generate Sample Data & Train Models (First Time)

	# Run the complete pipeline
	python main.py
	
This will:

1.  Generate sample food waste data (365 days)
2.  Preprocess and engineer features
3.  Train multiple ML models
4.  Evaluate and compare models
5.  Save the best model
6.  Generate visualization plots

### Step 3: GPU-Accelerated Training (Optional)

	# If you have NVIDIA GPU
	python main_gpu.py
	
### Step 4: Make Future Predictions (After Training)
```
python predict_future.py
```
### Advanced Usage
#### Custom Data Training
```
# train_custom.py
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer

# Load your own data
preprocessor = DataPreprocessor()
X_train, X_test, y_train, y_test, _ = preprocessor.prepare_data('your_data.csv') # Put your custom .csv Path here

# Train models
trainer = ModelTrainer()
results = trainer.train_all_models(X_train, y_train)
```

## 📁 Project Structure

<img width="832" height="623" alt="structure" src="https://github.com/user-attachments/assets/c4a33d51-8d70-4ac1-b135-ce9c1485b1c0" />


## 📊 Output & Results

### Training Output
```
🚂 TRAINING FOOD WASTE PREDICTION MODELS
==================================================
📊 Waste Data Statistics:
   Total samples: 4562
   Mean waste: 12.45 kg
   Std waste: 5.23 kg

🔄 Training Random Forest...
   Training time: 2.34s
   CV MAE: 2.15 kg (+/- 0.42)

🔄 Training XGBoost (GPU)...
   Training time: 0.45s
   CV MAE: 1.89 kg (+/- 0.38)
   GPU Memory used: 0.82 GB

⚡ GPU Speedup: 5.2x faster for waste prediction
```
### Prediction Output
```
📊 WASTE PREDICTIONS FOR Saturday, January 20, 2024
================================================================

🍽️ Grilled Chicken:
   Predicted Waste: 8.5 kg
   Confidence Range: [7.2 - 9.8] kg
   Estimated Cost: $68.00

📈 SUMMARY:
   Total Predicted Waste: 45.3 kg
   Total Estimated Cost: $362.40

💡 RECOMMENDATIONS:
   ⚠️ High waste predicted for:
      - Grilled Chicken: Consider reducing preparation by 1.7 kg

```
### Generated Visualizations

-   `data/plots/models_comparison.png`  - Model performance comparison
-   `data/plots/Random_Forest_analysis.png`  - Individual model analysis
-   `data/plots/future_prediction_20240120.png`  - Future prediction charts

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. CUDA/GPU Not Detected
```
# Check CUDA installation
nvidia-smi

# If not working, install CUDA Toolkit:
# Download from: https://developer.nvidia.com/cuda-downloads
```

#### 2. Import Errors
```
# Reinstall specific package
pip uninstall package_name
pip install package_name

# Or reinstall all requirements
pip install -r requirements.txt --force-reinstall
```

#### 3. Memory Errors
```
# Reduce batch size in neural network
nn_model = FoodWasteNeuralNetwork(epochs=50, batch_size=64)  # Smaller batch

# Or use CPU for large datasets
python main.py  # Instead of main_gpu.py
```

#### 4. File Not Found Errors
```
# Create required directories
mkdir data data/raw data/processed data/models data/plots

# Generate sample data
python generate_sample_data.py
```

#### 5. Permission Errors (Linux/Mac)
```
# Give execute permissions
chmod +x main.py
chmod -R 755 data/

# Run with sudo if needed
sudo python main.py
```

**Happy Predicting! 🍔📊**

_Reduce waste, save costs, and help the environment with AI-powered predictions!_
