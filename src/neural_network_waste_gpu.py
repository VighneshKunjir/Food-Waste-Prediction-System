# src/neural_network_waste_gpu.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
import time

class FoodWasteNN(nn.Module):
    """Neural Network specifically designed for food waste prediction"""
    def __init__(self, input_dim):
        super(FoodWasteNN, self).__init__()
        
        self.network = nn.Sequential(
            # Input layer
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Hidden layers for complex waste patterns
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
            
            # Output layer - single value for waste quantity
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        return self.network(x)

class FoodWasteNeuralNetwork(BaseEstimator, RegressorMixin):
    """Sklearn-compatible neural network for food waste prediction"""
    
    def __init__(self, epochs=100, batch_size=128, learning_rate=0.001):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = StandardScaler()
        self.training_history = {'loss': [], 'val_loss': []}
        
    def fit(self, X, y):
        """Train the neural network for waste prediction"""
        print(f"   Training on {self.device}")
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split for validation
        val_size = int(0.1 * len(X))
        X_train, X_val = X_scaled[:-val_size], X_scaled[-val_size:]
        y_train, y_val = y[:-val_size], y[-val_size:]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Initialize model
        self.model = FoodWasteNN(X.shape[1]).to(self.device)
        
        # Loss and optimizer for regression
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=min(self.batch_size, len(X_train)),
            shuffle=True
        )
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            epoch_loss = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
            
            avg_train_loss = epoch_loss / len(dataloader)
            self.training_history['loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(val_loss.item())
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            # Progress update
            if (epoch + 1) % 20 == 0:
                # Calculate MAE for waste prediction
                with torch.no_grad():
                    val_pred = self.model(X_val_tensor).cpu().numpy()
                    val_mae = np.mean(np.abs(y_val - val_pred))
                print(f"   Epoch {epoch + 1}/{self.epochs} - Loss: {avg_train_loss:.4f}, Val MAE: {val_mae:.2f} kg")
        
        return self
    
    def predict(self, X):
        """Predict food waste quantities"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = np.array(X, dtype=np.float32)
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
            predictions = predictions.cpu().numpy().flatten()
        
        # Ensure non-negative waste predictions
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def score(self, X, y):
        """Return R² score"""
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        if isinstance(y, pd.Series):
            y = y.values
        return r2_score(y, y_pred)