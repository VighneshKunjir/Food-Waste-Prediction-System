# benchmark_gpu.py
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def benchmark_models():
    """Benchmark GPU vs CPU performance"""
    print("\n" + "="*60)
    print("⚡ GPU vs CPU BENCHMARK")
    print("="*60)
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        print(f"\n🔥 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        # Clear any existing GPU memory
        torch.cuda.empty_cache()
    else:
        print("\n💻 Running on CPU only")
    
    # Test different data sizes
    sizes = [1000, 5000, 10000, 25000]  # Reduced max size to avoid memory issues
    results = []
    
    for size in sizes:
        print(f"\n📊 Testing with {size} samples...")
        
        # Generate synthetic data
        X = np.random.randn(size, 50).astype(np.float32)
        y = np.random.randn(size).astype(np.float32)
        
        result = {'size': size}
        
        # Test Random Forest (CPU)
        print("   Testing Random Forest (CPU)...")
        start = time.time()
        rf = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
        rf.fit(X, y)
        result['rf_cpu_time'] = time.time() - start
        
        # Test XGBoost CPU
        print("   Testing XGBoost (CPU)...")
        start = time.time()
        xgb_cpu = xgb.XGBRegressor(n_estimators=50, tree_method='hist', random_state=42)
        xgb_cpu.fit(X, y)
        result['xgb_cpu_time'] = time.time() - start
        
        # Test XGBoost GPU
        if device.type == 'cuda':
            print("   Testing XGBoost (GPU)...")
            try:
                start = time.time()
                xgb_gpu = xgb.XGBRegressor(
                    n_estimators=50, 
                    tree_method='gpu_hist', 
                    gpu_id=0,
                    random_state=42
                )
                xgb_gpu.fit(X, y)
                result['xgb_gpu_time'] = time.time() - start
                result['xgb_speedup'] = result['xgb_cpu_time'] / result['xgb_gpu_time']
            except Exception as e:
                print(f"      ⚠️ XGBoost GPU failed: {e}")
                result['xgb_gpu_time'] = None
                result['xgb_speedup'] = None
        
        # Test PyTorch Neural Network
        if device.type == 'cuda':
            print("   Testing Neural Network (GPU)...")
            try:
                # Clear GPU cache before neural network
                torch.cuda.empty_cache()
                
                X_tensor = torch.FloatTensor(X).to(device)
                y_tensor = torch.FloatTensor(y).to(device)
                
                model = nn.Sequential(
                    nn.Linear(50, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                ).to(device)
                
                optimizer = optim.Adam(model.parameters())
                criterion = nn.MSELoss()
                
                start = time.time()
                for _ in range(50):  # Reduced epochs for benchmark
                    optimizer.zero_grad()
                    output = model(X_tensor)
                    loss = criterion(output.squeeze(), y_tensor)
                    loss.backward()
                    optimizer.step()
                torch.cuda.synchronize()
                result['nn_gpu_time'] = time.time() - start
                
                # Clear GPU memory after use
                del X_tensor, y_tensor, model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"      ⚠️ Neural Network GPU failed: {e}")
                result['nn_gpu_time'] = None
        
        results.append(result)
        
        # Print results
        print(f"   Results for {size} samples:")
        print(f"      Random Forest (CPU): {result['rf_cpu_time']:.2f}s")
        print(f"      XGBoost (CPU): {result['xgb_cpu_time']:.2f}s")
        if 'xgb_gpu_time' in result and result['xgb_gpu_time'] is not None:
            print(f"      XGBoost (GPU): {result['xgb_gpu_time']:.2f}s (Speedup: {result['xgb_speedup']:.2f}x)")
        if 'nn_gpu_time' in result and result['nn_gpu_time'] is not None:
            print(f"      Neural Network (GPU): {result['nn_gpu_time']:.2f}s")
    
    # Create visualization
    df = pd.DataFrame(results)
    
    # Only create plots if we have valid data
    valid_gpu_data = any(df.get('xgb_gpu_time', [None]).notna()) if 'xgb_gpu_time' in df.columns else False
    
    if valid_gpu_data:
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Training times
        plt.subplot(1, 2, 1)
        plt.plot(df['size'], df['rf_cpu_time'], 'b-o', label='Random Forest (CPU)')
        plt.plot(df['size'], df['xgb_cpu_time'], 'g-o', label='XGBoost (CPU)')
        
        if 'xgb_gpu_time' in df.columns:
            valid_gpu = df['xgb_gpu_time'].notna()
            plt.plot(df.loc[valid_gpu, 'size'], df.loc[valid_gpu, 'xgb_gpu_time'], 'r-o', label='XGBoost (GPU)')
        
        if 'nn_gpu_time' in df.columns:
            valid_nn = df['nn_gpu_time'].notna()
            if valid_nn.any():
                plt.plot(df.loc[valid_nn, 'size'], df.loc[valid_nn, 'nn_gpu_time'], 'm-o', label='Neural Net (GPU)')
        
        plt.xlabel('Number of Samples')
        plt.ylabel('Training Time (seconds)')
        plt.title('Model Training Time Comparison')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Speedup
        if 'xgb_speedup' in df.columns:
            valid_speedup = df['xgb_speedup'].notna()
            if valid_speedup.any():
                plt.subplot(1, 2, 2)
                plt.bar(df.loc[valid_speedup, 'size'].astype(str), df.loc[valid_speedup, 'xgb_speedup'])
                plt.xlabel('Number of Samples')
                plt.ylabel('Speedup Factor')
                plt.title('GPU Speedup (XGBoost GPU vs CPU)')
                plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig('gpu_benchmark_results.png')
        print("\n📊 Benchmark plot saved as 'gpu_benchmark_results.png'")
        plt.show()
    
    return df

if __name__ == "__main__":
    results = benchmark_models()
    print("\n📊 Benchmark Results Summary:")
    print(results.to_string(index=False))
    
    # Final GPU memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\n💾 Final GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")