#!/usr/bin/env python3
"""
Train all VulnAI models sequentially
"""

import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_training(model_name: str) -> dict:
    """
    Train a single model
    
    Args:
        model_name: Model to train ('codebert', 'graphcodebert', 'codet5')
        
    Returns:
        dict with status and timing info
    """
    print(f"\n{'='*70}")
    print(f"🚀 Training {model_name.upper()}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    try:
        # Run training
        result = subprocess.run(
            [sys.executable, 'src/train.py', '--model', model_name],
            cwd=project_root,
            check=True,
            capture_output=False
        )
        
        elapsed = time.time() - start_time
        
        print(f"\n✅ {model_name.upper()} completed in {elapsed/60:.1f} minutes")
        
        return {
            'model': model_name,
            'status': 'success',
            'time_minutes': elapsed / 60
        }
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {model_name.upper()} failed after {elapsed/60:.1f} minutes")
        
        return {
            'model': model_name,
            'status': 'failed',
            'time_minutes': elapsed / 60,
            'error': str(e)
        }
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
        return {
            'model': model_name,
            'status': 'interrupted',
            'time_minutes': (time.time() - start_time) / 60
        }


def main():
    """Train all models and generate summary"""
    
    print(f"""
{'='*70}
🔥 VulnAI - Train All Models
{'='*70}
Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Models: CodeBERT, GraphCodeBERT, CodeT5
Estimated time: ~5-6 hours (with T4 GPU)
{'='*70}
    """)
    
    models = ['codebert', 'graphcodebert', 'codet5']
    results = []
    
    overall_start = time.time()
    
    # Train each model
    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] Starting {model}...")
        result = run_training(model)
        results.append(result)
        
        # Check if we should continue
        if result['status'] == 'interrupted':
            print("\n⚠️  Training sequence interrupted by user")
            break
        elif result['status'] == 'failed':
            user_input = input(f"\n⚠️  {model} failed. Continue with next model? (y/n): ")
            if user_input.lower() != 'y':
                print("Stopping training sequence")
                break
    
    overall_time = time.time() - overall_start
    
    # Print summary
    print(f"\n{'='*70}")
    print("📊 TRAINING SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"{'Model':<20} {'Status':<15} {'Time (min)':<12}")
    print("-" * 50)
    
    for result in results:
        status_emoji = {
            'success': '✅',
            'failed': '❌',
            'interrupted': '⚠️'
        }[result['status']]
        
        print(f"{result['model']:<20} {status_emoji} {result['status']:<12} {result['time_minutes']:<12.1f}")
    
    print(f"\n{'='*70}")
    print(f"⏱️  Total time: {overall_time/60:.1f} minutes ({overall_time/3600:.1f} hours)")
    
    successful = sum(1 for r in results if r['status'] == 'success')
    print(f"✅ Successful: {successful}/{len(models)}")
    
    if successful == len(models):
        print("\n🎉 All models trained successfully!")
        print("\n📊 Next steps:")
        print("   1. Run evaluation: python scripts/evaluate_all_models.py")
        print("   2. Check results: ls results/metrics/")
        print("   3. Try inference: python -m src.inference --code 'your code here'")
    else:
        print("\n⚠️  Some models failed to train")
        print("   Check logs above for error details")
    
    print(f"\n{'='*70}\n")
    
    # Return exit code
    return 0 if successful == len(models) else 1


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)