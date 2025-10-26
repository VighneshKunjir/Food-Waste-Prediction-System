# models/model_selector.py
"""Best model selection logic"""

import numpy as np


class ModelSelector:
    """Select best model based on metrics"""
    
    @staticmethod
    def select_best(results, metric='mae'):
        """Select best model from results"""
        if not results:
            return None, None, None
        
        best_model = None
        best_model_name = None
        best_score = float('inf')
        
        for model_name, model_data in results.items():
            score = model_data.get(metric, float('inf'))
            
            if score < best_score:
                best_score = score
                best_model = model_data['model']
                best_model_name = model_name
        
        print(f"\n🏆 BEST MODEL SELECTED")
        print(f"   Model: {best_model_name}")
        print(f"   {metric.upper()}: {best_score:.3f} kg")
        
        return best_model, best_model_name, best_score
    
    @staticmethod
    def rank_models(results, metric='mae'):
        """Rank all models by metric"""
        rankings = []
        
        for model_name, model_data in results.items():
            score = model_data.get(metric, float('inf'))
            rankings.append({
                'model': model_name,
                'score': score,
                'gpu_enabled': model_data.get('gpu_enabled', False)
            })
        
        rankings.sort(key=lambda x: x['score'])
        
        print(f"\n📊 MODEL RANKINGS ({metric.upper()}):")
        for i, rank in enumerate(rankings, 1):
            gpu_tag = "⚡ GPU" if rank['gpu_enabled'] else "💻 CPU"
            print(f"   {i}. {rank['model']}: {rank['score']:.3f} [{gpu_tag}]")
        
        return rankings