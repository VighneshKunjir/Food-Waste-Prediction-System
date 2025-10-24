# src/neural_network_gpu.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
import time
import os

class WastePredictionNN(nn.Module):
    def __init__(self, input_dim):
        super(WastePredictionNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.network(x)

class NeuralNetworkWrapper(BaseEstimator, RegressorMixin):
    """Sklearn-compatible wrapper for PyTorch neural network"""
    
    def __init__(self, input_dim=None, epochs=100, batch_size=256, learning_rate=0.001):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X, y):
        """Train the neural network"""
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Ensure correct dtype
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)
        
        # Set input dimension
        if self.input_dim is None:
            self.input_dim = X.shape[1]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Initialize model
        self.model = WastePredictionNN(self.input_dim).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Create dataset and dataloader for batching
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=min(self.batch_size, len(X)),
            shuffle=True
        )
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                avg_loss = epoch_loss / len(dataloader)
                print(f"   Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = np.array(X, dtype=np.float32)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
            predictions = predictions.cpu().numpy().flatten()
        
        return predictions
    
    def score(self, X, y):
        """Return R² score for compatibility with sklearn"""
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        if isinstance(y, pd.Series):
            y = y.values
        return r2_score(y, y_pred)

def train_neural_network_gpu(X_train, y_train, X_test, y_test, epochs=100):
    """Train neural network on GPU with sklearn-compatible wrapper"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🧠 Training Neural Network on {device}")
    
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()  # Clear any existing cache
    
    # Create wrapper model
    nn_wrapper = NeuralNetworkWrapper(epochs=epochs, batch_size=256, learning_rate=0.001)
    
    # Record training time
    start_time = time.time()
    
    # Train the model
    nn_wrapper.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Evaluate
    predictions = nn_wrapper.predict(X_test)
    
    # Convert y_test to numpy if needed
    if isinstance(y_test, pd.Series):
        y_test_np = y_test.values
    else:
        y_test_np = y_test
    
    mae = np.mean(np.abs(y_test_np - predictions))
    
    print(f"\n✅ Neural Network Training Complete")
    print(f"   Training time: {training_time:.2f}s")
    print(f"   Test MAE: {mae:.3f}")
    
    if device.type == 'cuda':
        print(f"   GPU Memory used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        torch.cuda.empty_cache()
    
    # Return the wrapper (which has predict method)
    return nn_wrapper, nn_wrapper.scaler, mae