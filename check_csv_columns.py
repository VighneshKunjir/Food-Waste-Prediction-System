# check_csv_columns.py
"""Check what columns are in your CSV file"""

import pandas as pd

print("🔍 CSV COLUMN INSPECTOR\n")
print("="*60)

csv_path = input("Enter your CSV file path: ").strip().strip('"')

try:
    df = pd.read_csv(csv_path)
    
    print(f"\n📊 File: {csv_path}")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}\n")
    
    print("📋 COLUMN NAMES:")
    print("="*60)
    
    for i, col in enumerate(df.columns, 1):
        sample_val = df[col].iloc[0] if len(df) > 0 else "N/A"
        print(f"{i:2}. {col:30} (sample: {sample_val})")
    
    print("\n" + "="*60)
    
    # Check for waste-related columns
    waste_columns = [col for col in df.columns if 'waste' in col.lower() or 'wastage' in col.lower()]
    
    if waste_columns:
        print("✅ Found waste-related columns:")
        for col in waste_columns:
            print(f"   • {col}")
    else:
        print("⚠️ No obvious waste column found")
        print("\n💡 Look for columns that might contain waste data:")
        print("   Common names: Wastage Food Amount, Food Waste, Waste Amount, etc.")
    
    # Show first few rows
    print("\n📊 First 3 rows preview:")
    print("="*60)
    print(df.head(3).to_string())
    
except Exception as e:
    print(f"❌ Error: {e}")