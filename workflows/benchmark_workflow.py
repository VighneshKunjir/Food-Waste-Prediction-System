# workflows/benchmark_workflow.py
"""GPU benchmarking workflow"""

import time
import numpy as np
import pandas as pd
import torch

from models.model_factory import ModelFactory
from utils.gpu_utils import GPUManager
from utils.visualization import Visualizer
from utils.file_utils import FileManager
from utils.logger import Logger


class BenchmarkWorkflow:
    """Orchestrate GPU benchmarking"""
    
    def __init__(self):
        """Initialize benchmark workflow"""
        self.gpu_manager = GPUManager()
        self.visualizer = Visualizer()
        self.logger = Logger(name='Benchmark')
        
        self.results = []
    
    def run_complete_benchmark(self, data_sizes=[1000, 5000, 10000, 25000]):
        """
        Run complete GPU vs CPU benchmark
        
        Args:
            data_sizes: List of data sizes to test
        
        Returns:
            Benchmark results DataFrame
        """
        print("\n" + "="*70)
        print("⚡ GPU vs CPU PERFORMANCE BENCHMARK")
        print("="*70)
        
        # Display GPU info
        self.gpu_manager.display_info()
        self.gpu_manager.display_capabilities()
        
        # Run benchmarks for each data size
        for size in data_sizes:
            print(f"\n{'='*70}")
            print(f"📊 Testing with {size:,} samples")
            print('='*70)
            
            result = self._benchmark_single_size(size)
            self.results.append(result)
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Display summary
        self._display_benchmark_summary(results_df)
        
        # Generate visualization
        self._generate_benchmark_plots(results_df)
        
        # Save results
        self._save_benchmark_results(results_df)
        
        self.logger.info("Benchmark completed")
        
        return results_df
    
    def _benchmark_single_size(self, size):
        """Benchmark single data size"""
        # Generate synthetic data
        print(f"\n🔄 Generating {size:,} samples with 50 features...")
        X = np.random.randn(size, 50).astype(np.float32)
        y = np.random.randn(size).astype(np.float32)
        
        result = {
            'size': size,
            'features': 50
        }
        
        # Test CPU models
        print("\n💻 CPU Models:")
        cpu_models = ModelFactory.get_cpu_models()
        
        for model_name, model_wrapper in list(cpu_models.items())[:3]:  # Test first 3
            try:
                start_time = time.time()
                model_wrapper.fit(X, y)
                training_time = time.time() - start_time
                
                result[f'{model_name}_cpu_time'] = training_time
                print(f"   {model_name}: {training_time:.3f}s")
                
            except Exception as e:
                print(f"   {model_name}: Failed ({e})")
        
        # Test GPU models (if available)
        if self.gpu_manager.gpu_available:
            print("\n🔥 GPU Models:")
            
            # Clear GPU cache
            self.gpu_manager.clear_cache()
            
            gpu_models = ModelFactory.get_gpu_models()
            
            for model_name, model_wrapper in gpu_models.items():
                try:
                    # Measure GPU memory before
                    mem_before = self.gpu_manager.get_memory_info()
                    
                    start_time = time.time()
                    model_wrapper.fit(X, y)
                    
                    # Ensure GPU sync
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    training_time = time.time() - start_time
                    
                    # Measure GPU memory after
                    mem_after = self.gpu_manager.get_memory_info()
                    
                    result[f'{model_name}_gpu_time'] = training_time
                    if mem_after and mem_before:
                        result[f'{model_name}_gpu_memory'] = mem_after['allocated']
                    
                    print(f"   {model_name}: {training_time:.3f}s")
                    
                    # Calculate speedup if CPU version exists
                    cpu_key = f'{model_name.replace(" (GPU)", "")}_cpu_time'
                    if cpu_key in result:
                        speedup = result[cpu_key] / training_time
                        result[f'{model_name}_speedup'] = speedup
                        print(f"      Speedup: {speedup:.2f}x")
                    
                    # Clear GPU memory
                    self.gpu_manager.clear_cache()
                    
                except Exception as e:
                    print(f"   {model_name}: Failed ({e})")
        
        return result
    
    def _display_benchmark_summary(self, results_df):
        """Display benchmark summary"""
        print("\n" + "="*70)
        print("📊 BENCHMARK SUMMARY")
        print("="*70)
        
        print("\n" + results_df.to_string(index=False))
        
        # Calculate average speedup
        speedup_cols = [col for col in results_df.columns if 'speedup' in col]
        
        if speedup_cols:
            print("\n⚡ Average Speedup Factors:")
            for col in speedup_cols:
                avg_speedup = results_df[col].mean()
                model_name = col.replace('_speedup', '')
                print(f"   {model_name}: {avg_speedup:.2f}x faster on GPU")
    
    def _generate_benchmark_plots(self, results_df):
        """Generate benchmark visualization"""
        import matplotlib.pyplot as plt
        
        print("\n📊 Generating benchmark plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Training Times Comparison
        ax1 = axes[0]
        
        time_cols = [col for col in results_df.columns if '_time' in col and 'cpu' not in col.lower()]
        
        for col in time_cols:
            model_name = col.replace('_gpu_time', '').replace('_cpu_time', '')
            ax1.plot(results_df['size'], results_df[col], 'o-', label=model_name, linewidth=2)
        
        ax1.set_xlabel('Number of Samples', fontsize=12)
        ax1.set_ylabel('Training Time (seconds)', fontsize=12)
        ax1.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: GPU Speedup
        ax2 = axes[1]
        
        speedup_cols = [col for col in results_df.columns if 'speedup' in col]
        
        if speedup_cols:
            x = np.arange(len(results_df))
            width = 0.2
            
            for i, col in enumerate(speedup_cols):
                model_name = col.replace('_speedup', '')
                ax2.bar(x + i*width, results_df[col], width, label=model_name)
            
            ax2.set_xlabel('Data Size Index', fontsize=12)
            ax2.set_ylabel('Speedup Factor (x)', fontsize=12)
            ax2.set_title('GPU Speedup vs CPU', fontsize=14, fontweight='bold')
            ax2.set_xticks(x + width)
            ax2.set_xticklabels(results_df['size'])
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = 'results/plots/gpu_benchmark.png'
        FileManager.create_directory('results/plots')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Benchmark plot saved: {plot_path}")
        
        plt.show()
    
    def _save_benchmark_results(self, results_df):
        """Save benchmark results"""
        # Save to CSV
        csv_path = 'results/reports/benchmark_results.csv'
        FileManager.save_csv(results_df, csv_path)
        
        # Save to JSON
        json_path = 'results/reports/benchmark_results.json'
        results_dict = results_df.to_dict('records')
        FileManager.save_json(results_dict, json_path)
        
        print(f"\n💾 Benchmark results saved:")
        print(f"   CSV: {csv_path}")
        print(f"   JSON: {json_path}")
    
    def quick_gpu_check(self):
        """Quick GPU availability check"""
        print("\n" + "="*70)
        print("🔍 QUICK GPU CHECK")
        print("="*70)
        
        self.gpu_manager.display_info()
        self.gpu_manager.display_capabilities()
        
        if self.gpu_manager.gpu_available:
            # Quick test
            print("\n⚡ Running quick GPU test...")
            
            X = np.random.randn(1000, 50).astype(np.float32)
            y = np.random.randn(1000).astype(np.float32)
            
            # Test XGBoost GPU
            try:
                from models.xgboost_model import XGBoostModel
                
                model = XGBoostModel(use_gpu=True)
                start = time.time()
                model.fit(X, y)
                elapsed = time.time() - start
                
                print(f"✅ XGBoost GPU test: {elapsed:.3f}s")
                print("🎉 GPU is working correctly!")
                
            except Exception as e:
                print(f"❌ GPU test failed: {e}")
        else:
            print("\n💡 To enable GPU acceleration:")
            print("   1. Install CUDA Toolkit")
            print("   2. Install GPU-enabled libraries:")
            print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")