# models/neural_network.py
"""Neural Network - GPU optimized (PyTorch)"""

from .base_model import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class WasteNN(nn.Module):
    """Neural network architecture"""
    
    def __init__(self, input_dim):
        super(WasteNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.network(x)


class NeuralNetworkModel(BaseModel):
    """Neural Network Model with GPU support"""
    
    def __init__(self, epochs=100, batch_size=128, learning_rate=0.001, use_gpu=True):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.scaler = StandardScaler()
        self.input_dim = None
        super().__init__(use_gpu)
    
    def _check_gpu(self):
        """Check if CUDA is available"""
        return torch.cuda.is_available()
    
    def _create_model(self):
        """Create neural network"""
        # Model is created during fit()
        return None
    
    def fit(self, X, y):
        """Train the neural network"""
        print(f"\n🔄 Training Neural Network...")
        
        # Set device
        device = torch.device('cuda' if (self.use_gpu and self.gpu_available) else 'cpu')
        print(f"   Using: {device}")
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)
        
        self.input_dim = X.shape[1]
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        # Validation split
        val_size = int(0.1 * len(X))
        X_train, X_val = X_scaled[:-val_size], X_scaled[-val_size:]
        y_train, y_val = y[:-val_size], y[-val_size:]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_tensor = torch.FloatTensor(y_val).to(device)
        
        # Initialize model
        self.model = WasteNN(self.input_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        
        # Create dataloader
        dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(X_train)),
            shuffle=True
        )
        
        # Training loop
        import time
        start_time = time.time()
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                val_mae = torch.mean(torch.abs(val_outputs - y_val_tensor))
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"   Epoch {epoch + 1}/{self.epochs} - Loss: {epoch_loss/len(dataloader):.4f}, Val MAE: {val_mae:.2f} kg")
        
        self.training_time = time.time() - start_time
        print(f"   Training time: {self.training_time:.2f}s")
        
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        device = torch.device('cuda' if (self.use_gpu and self.gpu_available) else 'cpu')
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = np.array(X, dtype=np.float32)
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
            predictions = predictions.cpu().numpy().flatten()
        
        # Ensure non-negative
        predictions = np.maximum(predictions, 0)
        
        return predictions