# VulnAI - Multi-Model Code Vulnerability Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Experimental multi-model ensemble system for detecting vulnerabilities in C code using fine-tuned transformer models.

## 🎯 Overview

This project explores using state-of-the-art code understanding models (CodeBERT, GraphCodeBERT, CodeT5) to detect security vulnerabilities in source code. Trained on Microsoft's Devign dataset containing 21K+ labeled C functions.

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| CodeBERT | 66.35% | 68.42% | 47.97% | 56.40% |
| GraphCodeBERT | 61.66% | 55.57% | 77.23% | **64.63%** |
| CodeT5 | 56.68% | 51.71% | 68.14% | 58.80% |

**Baseline comparison:** 12% improvement over regex patterns (54% accuracy)

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/vulnai.git
cd vulnai
pip install -r requirements.txt
```

### Download Dataset
```bash
python data/download_data.py
python data/preprocess.py
```

### Train Models
```bash
# Train individual models
python src/train.py --model codebert
python src/train.py --model graphcodebert
python src/train.py --model codet5

# Or train all at once
bash scripts/train_all_models.sh
```

### Inference
```python
from src.inference import VulnerabilityDetector

detector = VulnerabilityDetector(model_name='graphcodebert')

code = '''
def get_user(user_id):
    query = "SELECT * FROM users WHERE id=" + user_id
    return execute(query)
'''

result = detector.predict(code)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 📁 Project Structure
```
vulnai/
├── data/              # Data download and preprocessing
├── src/               # Core source code
├── notebooks/         # Jupyter experiments
├── scripts/           # Training/evaluation scripts
└── results/           # Saved models and metrics
```

## 🔬 Models

### CodeBERT
- **Architecture:** BERT pre-trained on code
- **Parameters:** 125M
- **Best for:** Keyword-based patterns

### GraphCodeBERT
- **Architecture:** Graph-aware BERT (understands code structure)
- **Parameters:** 125M
- **Best for:** Structural vulnerabilities (highest F1 score)

### CodeT5
- **Architecture:** T5 encoder-decoder adapted for code
- **Parameters:** 220M
- **Best for:** Complex patterns

## 📚 Dataset

**Devign** (Microsoft Research)
- **Size:** 21,854 C functions
- **Source:** Real projects (QEMU, FFmpeg, Linux kernel)
- **Labels:** Binary (vulnerable / safe)
- **Split:** 80% train, 10% val, 10% test (zero leakage verified)

## ⚠️ Limitations

- **Not production-ready:** 66% accuracy insufficient for security-critical applications
- **Language-specific:** Trained only on C code
- **Context-limited:** 512 token window may miss complex vulnerabilities
- **Interpretability:** Black-box predictions (no explanation of why code is vulnerable)

## 🎓 Learnings

Key insights from this project:

1. **Data quality matters more than model size** - GraphCodeBERT (125M) outperformed CodeT5 (220M)
2. **Proper evaluation is critical** - Initial data leakage gave false 100% accuracy
3. **Ensemble isn't magic** - Weighted voting failed due to model disagreement
4. **Domain knowledge required** - Security tools need interpretability, not just accuracy

## 🔮 Future Improvements

- [ ] Graph Neural Networks for AST/CFG analysis
- [ ] Hybrid approach combining static analysis + ML
- [ ] Explainability (attention visualization, LIME)
- [ ] Active learning on uncertain samples
- [ ] Multi-language support (Python, JavaScript, Java)

## 📖 Citation

If you use this code, please cite:
```bibtex
@misc{vulnai2026,
  author = {Your Name},
  title = {VulnAI: Multi-Model Code Vulnerability Detection},
  year = {2026},
  url = {https://github.com/yourusername/vulnai}
}
```


## 🙏 Acknowledgments

- Microsoft Research for the Devign dataset
- HuggingFace for model implementations
- Kaggle for free GPU compute



---

⚠️ **Disclaimer:** This is an experimental research project. Do not use for production security scanning without extensive validation and human review.
