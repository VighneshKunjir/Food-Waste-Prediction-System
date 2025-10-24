# test_installation.py
import sys
print(f"Python Version: {sys.version}")

try:
    import pandas
    print("✅ Pandas installed")
except:
    print("❌ Pandas not installed")

try:
    import sklearn
    print("✅ Scikit-learn installed")
except:
    print("❌ Scikit-learn not installed")

try:
    import torch
    print(f"✅ PyTorch installed")
    if torch.cuda.is_available():
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ No GPU detected, will use CPU")
except:
    print("❌ PyTorch not installed")

try:
    import xgboost
    print("✅ XGBoost installed")
except:
    print("⚠️ XGBoost not installed (optional)")

print("\n✅ Installation check complete!")