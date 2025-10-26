# utils/gpu_utils.py
"""GPU detection and management utilities"""

import os
import warnings


class GPUManager:
    """Manage GPU resources and detection"""
    
    def __init__(self):
        self.gpu_available = False
        self.device_name = None
        self.device_count = 0
        self.cuda_version = None
        self.memory_total = 0
        self.memory_available = 0
        
        self._detect_gpu()
    
    def _detect_gpu(self):
        """Detect GPU availability and properties"""
        try:
            import torch
            
            if torch.cuda.is_available():
                self.gpu_available = True
                self.device_count = torch.cuda.device_count()
                self.device_name = torch.cuda.get_device_name(0)
                self.cuda_version = torch.version.cuda
                
                # Get memory info
                props = torch.cuda.get_device_properties(0)
                self.memory_total = props.total_memory / 1e9  # Convert to GB
                
                # Get available memory
                self.memory_available = (torch.cuda.get_device_properties(0).total_memory - 
                                        torch.cuda.memory_allocated(0)) / 1e9
            else:
                self.gpu_available = False
                
        except ImportError:
            self.gpu_available = False
    
    def get_device(self):
        """Get torch device (cuda or cpu)"""
        try:
            import torch
            return torch.device('cuda' if self.gpu_available else 'cpu')
        except ImportError:
            return 'cpu'
    
    def clear_cache(self):
        """Clear GPU cache"""
        if self.gpu_available:
            try:
                import torch
                torch.cuda.empty_cache()
                print("✅ GPU cache cleared")
            except:
                pass
    
    def get_memory_info(self):
        """Get current GPU memory usage"""
        if not self.gpu_available:
            return None
        
        try:
            import torch
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            free = self.memory_total - allocated
            
            return {
                'total': self.memory_total,
                'allocated': allocated,
                'reserved': reserved,
                'free': free
            }
        except:
            return None
    
    def display_info(self):
        """Display GPU information"""
        print("\n" + "="*60)
        print("🔥 GPU INFORMATION")
        print("="*60)
        
        if self.gpu_available:
            print(f"✅ GPU Available: {self.device_name}")
            print(f"   CUDA Version: {self.cuda_version}")
            print(f"   Device Count: {self.device_count}")
            print(f"   Total Memory: {self.memory_total:.2f} GB")
            
            mem_info = self.get_memory_info()
            if mem_info:
                print(f"   Allocated Memory: {mem_info['allocated']:.2f} GB")
                print(f"   Free Memory: {mem_info['free']:.2f} GB")
        else:
            print("💻 No GPU detected - using CPU")
            print("   To enable GPU:")
            print("   1. Install CUDA toolkit")
            print("   2. Install PyTorch with CUDA support:")
            print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
        print("="*60)
    
    def set_device(self, device_id=0):
        """Set active GPU device"""
        if self.gpu_available:
            try:
                import torch
                torch.cuda.set_device(device_id)
                print(f"✅ Active GPU device set to: {device_id}")
            except:
                print(f"⚠️ Could not set device {device_id}")
    
    def check_cuml_available(self):
        """Check if RAPIDS cuML is available"""
        try:
            import cuml
            return True
        except ImportError:
            return False
    
    def check_xgboost_gpu(self):
        """Check if XGBoost GPU is available"""
        try:
            import xgboost as xgb
            return self.gpu_available
        except ImportError:
            return False
    
    def check_lightgbm_gpu(self):
        """Check if LightGBM GPU is available"""
        try:
            import lightgbm as lgb
            return self.gpu_available
        except ImportError:
            return False
    
    def get_gpu_capabilities(self):
        """Get all GPU capabilities"""
        capabilities = {
            'pytorch_cuda': self.gpu_available,
            'cuml': self.check_cuml_available(),
            'xgboost_gpu': self.check_xgboost_gpu(),
            'lightgbm_gpu': self.check_lightgbm_gpu()
        }
        
        return capabilities
    
    def display_capabilities(self):
        """Display GPU capabilities"""
        print("\n🔍 GPU Capabilities:")
        
        caps = self.get_gpu_capabilities()
        
        print(f"   PyTorch CUDA: {'✅' if caps['pytorch_cuda'] else '❌'}")
        print(f"   RAPIDS cuML: {'✅' if caps['cuml'] else '❌'}")
        print(f"   XGBoost GPU: {'✅' if caps['xgboost_gpu'] else '❌'}")
        print(f"   LightGBM GPU: {'✅' if caps['lightgbm_gpu'] else '❌'}")
    
    @staticmethod
    def optimize_gpu_settings():
        """Optimize GPU settings for training"""
        try:
            import torch
            
            # Enable TF32 for better performance on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable cudnn benchmarking
            torch.backends.cudnn.benchmark = True
            
            print("✅ GPU settings optimized")
            
        except:
            pass
    
    @staticmethod
    def set_memory_limit(fraction=0.9):
        """Set GPU memory limit"""
        try:
            import torch
            
            if torch.cuda.is_available():
                # Set memory fraction
                torch.cuda.set_per_process_memory_fraction(fraction, 0)
                print(f"✅ GPU memory limit set to {fraction*100}%")
        except:
            pass