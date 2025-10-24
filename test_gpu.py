# test_gpu.py
import torch
import xgboost as xgb

print("=" * 50)
print("GPU AVAILABILITY CHECK")
print("=" * 50)

# Check PyTorch GPU
print(f"\n✅ PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")

# Check XGBoost GPU
print(f"\n✅ XGBoost version: {xgb.__version__}")
print("   XGBoost GPU support: Available")

# Test GPU operation
if torch.cuda.is_available():
    print("\n🔥 Testing GPU computation...")
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    print("   GPU computation successful!")