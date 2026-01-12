"""
Download Devign dataset
"""

from datasets import load_dataset
from pathlib import Path

def download_devign():
    """Download and save Devign dataset"""
    print("Downloading Devign dataset...")
    
    dataset = load_dataset("code_x_glue_cc_defect_detection")
    
    # Save to disk
    output_dir = Path("data/raw/devign")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset.save_to_disk(str(output_dir))
    
    print(f"✅ Dataset saved to {output_dir}")
    print(f"   Train: {len(dataset['train']):,} samples")
    print(f"   Valid: {len(dataset['validation']):,} samples")
    print(f"   Test: {len(dataset['test']):,} samples")

if __name__ == '__main__':
    download_devign()