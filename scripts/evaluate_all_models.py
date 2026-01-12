#!/usr/bin/env python3
"""
Evaluate all trained VulnAI models
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import RESULTS_DIR


def load_results(model_name: str) -> dict:
    """Load results for a model"""
    results_file = RESULTS_DIR / 'metrics' / f'{model_name}_results.json'
    
    if not results_file.exists():
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)


def print_comparison():
    """Print comparison table of all models"""
    
    models = ['codebert', 'graphcodebert', 'codet5']
    
    print(f"\n{'='*70}")
    print("📊 MODEL COMPARISON")
    print(f"{'='*70}\n")
    
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 70)
    
    all_results = []
    
    for model in models:
        results = load_results(model)
        
        if results:
            acc = results.get('test_accuracy', 0) * 100
            prec = results.get('test_precision', 0) * 100
            rec = results.get('test_recall', 0) * 100
            f1 = results.get('test_f1', 0) * 100
            
            print(f"{model:<20} {acc:<12.2f} {prec:<12.2f} {rec:<12.2f} {f1:<12.2f}")
            all_results.append((model, results))
        else:
            print(f"{model:<20} {'NOT TRAINED'}")
    
    print(f"\n{'='*70}")
    
    # Find best model
    if all_results:
        best_model = max(all_results, key=lambda x: x[1].get('test_f1', 0))
        best_f1 = best_model[1]['test_f1'] * 100
        
        print(f"\n🏆 Best model: {best_model[0].upper()} (F1: {best_f1:.2f}%)")
        
        # Training time summary
        total_time = sum(r[1].get('training_time_minutes', 0) for r in all_results)
        print(f"⏱️  Total training time: {total_time:.1f} minutes ({total_time/60:.1f} hours)")
    
    print(f"\n{'='*70}\n")


def main():
    """Main evaluation function"""
    
    print(f"""
{'='*70}
📊 VulnAI - Evaluate All Models
{'='*70}
Evaluated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}
    """)
    
    print_comparison()
    
    print("💡 Next steps:")
    print("   1. Review detailed metrics in results/metrics/")
    print("   2. Check visualizations in results/visualizations/")
    print("   3. Try inference: python -m src.inference --help")
    print("\n")


if __name__ == '__main__':
    main()