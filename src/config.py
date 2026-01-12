"""
Configuration file for VulnAI project
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Ensure directories exist
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, CHECKPOINTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model configurations
MODEL_CONFIGS = {
    'codebert': {
        'model_name': 'microsoft/codebert-base',
        'max_length': 512,
        'batch_size': 8,
        'learning_rate': 2e-5,
        'num_epochs': 3
    },
    'graphcodebert': {
        'model_name': 'microsoft/graphcodebert-base',
        'max_length': 512,
        'batch_size': 8,
        'learning_rate': 2e-5,
        'num_epochs': 3
    },
    'codet5': {
        'model_name': 'Salesforce/codet5-small',
        'max_length': 512,
        'batch_size': 8,
        'learning_rate': 2e-5,
        'num_epochs': 3
    }
}

# Training configuration
TRAIN_CONFIG = {
    'seed': 42,
    'train_split': 0.8,
    'val_split': 0.1,
    'test_split': 0.1,
    'num_workers': 4,
    'gradient_accumulation_steps': 2,
    'warmup_steps': 500,
    'weight_decay': 0.01,
    'fp16': False,
    'early_stopping_patience': 2
}

# Dataset configuration
DATASET_CONFIG = {
    'name': 'code_x_glue_cc_defect_detection',
    'min_code_length': 100,
    'max_code_length': 5000
}