
# 🍔 Food Waste Prediction System - AI-Powered & GPU Accelerated

A comprehensive, production-ready machine learning system for predicting food waste in restaurants. Features  **dynamic model discovery**,  **GPU acceleration**,  **multi-restaurant management**, and an intelligent  **menu-driven interface**  for seamless operation.

----------

## ✨ Features

### 🚀 Core Capabilities

-   **🔄 Dynamic Model Discovery**: Auto-detects available ML models - add/remove models without code changes
-   **🧠 Smart Neural Network Toggle**: Choose whether to train Neural Networks (GPU only) for optimal speed vs accuracy
-   **⚡ GPU Acceleration**: Leverages NVIDIA GPUs for 5-15x faster training with intelligent memory management
-   **🏪 Multi-Restaurant Support**: Manage unlimited restaurants with separate models and predictions
-   **📊 Universal Data Adapter**: Auto-detects and adapts any CSV format to standard schema
-   **💾 Advanced Metadata Tracking**: Stores training configuration (GPU usage, models trained, NN inclusion)
-   **🎯 Intelligent Model Selection**: Automatically trains only available models and evaluates them

### 🤖 Machine Learning Models

**Dynamically Discovered Models Include:**

-   Linear Regression
-   Ridge & Lasso Regression
-   Decision Tree
-   Random Forest
-   Gradient Boosting
-   XGBoost (GPU-accelerated)
-   LightGBM (GPU-accelerated)
-   CatBoost (GPU-accelerated)
-   Support Vector Machine (SVM)
-   Neural Network (PyTorch - optional, GPU-accelerated)
-   Ensemble Model

**✨ NEW: Add your own models!**  Just drop a new model file in  `models/`  directory and it's automatically discovered.

### 📈 Prediction Capabilities

-   **Future Predictions**:
    
    -   Single-day prediction for specific food items
    -   Multiple items prediction for a single day
    -   7-day ahead forecasting with visualizations
-   **Real-time Predictions**:
    
    -   Interactive CLI for instant predictions
    -   Batch prediction from CSV files
    -   Historical prediction tracking
-   **Smart Features**:
    
    -   Confidence intervals for all predictions
    -   Weather and special event adjustments
    -   Automatic waste reduction recommendations
    -   Cost estimation and savings analysis

### 📊 Evaluation & Analytics

-   Comprehensive model comparison reports
-   10+ evaluation metrics (MAE, RMSE, R², MAPE, etc.)
-   Automatic visualization generation:
    -   Actual vs Predicted plots
    -   Residuals analysis
    -   Feature importance charts
    -   Model comparison graphs
-   Exportable JSON/CSV reports

### 🏪 Restaurant Management

-   Create and manage multiple restaurants
-   View detailed restaurant information
-   Track prediction history (real-time & future)
-   Export restaurant data
-   Configuration management per restaurant

### ⚡ GPU Performance

-   Automatic GPU detection and optimization
-   Memory management and cleanup
-   Quick GPU check and full benchmarking
-   CPU fallback for non-GPU systems

----------

## 💻 System Requirements

### Hardware

-   **CPU**: Multi-core processor (4+ cores recommended)
-   **RAM**: 8GB minimum, 16GB recommended
-   **GPU**  (Optional but recommended):
    -   NVIDIA GPU with CUDA support
    -   4GB+ VRAM for Neural Networks
    -   2GB+ VRAM for other GPU models

### Software

-   **Python**: 3.10 or 3.11 (3.11 recommended)
-   **CUDA Toolkit**: 11.6+ (for GPU support)
-   **Operating System**: Windows 10/11, Linux, macOS

## 🚀 Installation Guide

### Step 1: Clone Repository

```
git clone https://github.com/yourusername/food-waste-prediction.git
cd food-waste-prediction
```
### Step 2: Set Up Python Environment

#### Option A: Virtual Environment (Recommended)
```
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

#### Option B: Conda

```
# Create conda environment
conda create -n foodwaste python=3.11
conda activate foodwaste
```

### Step 3: Install Dependencies

#### For CPU-Only Version:
```
pip install -r requirements.txt
```

#### For GPU-Accelerated Version (Recommended):
```
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt

# GPU-accelerated libraries (already in requirements.txt)
# xgboost, lightgbm, catboost
```

### Step 4: Verify Installation
```
# Check GPU availability
python -c "import torch; print('GPU Available:', torch.cuda.is_available())"

# Expected output:
# GPU Available: True  (if GPU properly configured)
# GPU Available: False (if no GPU, system will use CPU)
```

## 📖 Usage Guide / How to Run
#### 1. Launch the System
```
# Navigate to project directory
cd food-waste-prediction

# Activate environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Run main system
python main.py
```

#### 2. First Time Setup - Train Your First Model

```
Select from Menu:
1. TRAIN NEW MODEL
   → 1.1 Generate Sample Data & Train

Follow prompts:
   Enter restaurant name: MyRestaurant
   Number of days: 365 (or press Enter for default)
   
   🔥 GPU available!
   Use GPU acceleration? (y/n): y
   
   🧠 NEURAL NETWORK TRAINING
   Train with Neural Network? (y/n): n  (Faster training)
   
   ⏭️  Neural Network will be skipped
   ℹ️  Will train with 10 other models
   
   System will:
   ✅ Generate sample data
   ✅ Preprocess features
   ✅ Train 10 ML models
   ✅ Evaluate and select best model
   ✅ Save model and metadata
```

#### 3. Make Predictions

```
Select from Menu:
2. MAKE PREDICTIONS
   → 2.1 Future Waste Prediction
   → Select prediction type

Example - Weekly Forecast:
   Start date: 2024-01-20 (or press Enter for tomorrow)
   
   Output:
   📊 7-Day Waste Forecast
   ✅ Predictions saved
   📈 Visualization generated
```

## 🔧 Troubleshooting

### 1. GPU Not Detected


```
# Check NVIDIA driver
nvidia-smi

# If command not found, install NVIDIA drivers:
# Windows: Download from nvidia.com
# Linux: sudo apt install nvidia-driver-XXX

# Check CUDA installation
nvcc --version

# Install CUDA Toolkit if missing:
# https://developer.nvidia.com/cuda-downloads

# Verify PyTorch GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. "No models discovered" Error

```
# Check models directory
ls models/

# Ensure model files exist and are not corrupted
# Restore from backup if needed

# Check for import errors
python -c "from models import AVAILABLE_MODELS; print(AVAILABLE_MODELS)"
```

### 3. Import/Module Errors

```
# Reinstall requirements
pip install -r requirements.txt --force-reinstall

# If specific package fails
pip uninstall package_name
pip install package_name

# Clear pip cache if persistent issues
pip cache purge
pip install -r requirements.txt
```

### 4. CUDA Out of Memory

```
# Reduce batch size (if using Neural Network)
# In models/neural_network.py, change:
# batch_size=64  # Instead of 128

# Or skip Neural Network during training:
# When prompted: Train with Neural Network? → n

# Monitor GPU memory
nvidia-smi -l 1  # Updates every second
```

### 5. File Permission Errors (Linux/Mac)

```
# Give permissions to data directory
chmod -R 755 data/
chmod -R 755 models/

# If still issues, check ownership
sudo chown -R $USER:$USER .
```

### 6. CSV Adapter Issues

```
# If Universal Adapter fails:
# 1. Check CSV format (must be valid CSV)
# 2. Ensure required columns exist (date, food item, waste quantity)
# 3. Try manual column mapping when prompted

# Test CSV loading
python -c "import pandas as pd; df = pd.read_csv('your_file.csv'); print(df.head())"
```

### 7. Model Training Failures

```
# If a specific model fails during training:
# System will skip it and continue with others

# To remove problematic model:
# Delete or rename the model file in models/
# Example: mv models/broken_model.py models/broken_model.py.disabled

# System will auto-discover remaining models
```

### 8. Prediction Not Working

```
# Ensure model is trained first
# Check: data/restaurants/YOUR_RESTAURANT/models/best_model.pkl exists

# Verify metadata
cat data/restaurants/YOUR_RESTAURANT/models/best_model_metadata.json

# If corrupted, retrain model:
# Select: 1 → Train New Model
```
