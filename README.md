# VulnAI - Multi-Model Code Vulnerability Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Medium Article](https://img.shields.io/badge/Medium-Article-black)](https://medium.com/@amrgabeerr20/building-a-multi-model-ensemble-for-code-vulnerability-detection-lessons-from-fine-tuning-2da63c7fc279)

Experimental multi-model ensemble system for detecting vulnerabilities in C code using fine-tuned transformer models.

> 📖 **Read the full story:** [Building a Multi-Model Ensemble for Code Vulnerability Detection](https://medium.com/@amrgabeerr20/building-a-multi-model-ensemble-for-code-vulnerability-detection-lessons-from-fine-tuning-2da63c7fc279)

## 🎯 Overview

This project explores using state-of-the-art code understanding models (CodeBERT, GraphCodeBERT, CodeT5) to detect security vulnerabilities in source code. Trained on Microsoft's Devign dataset containing 21K+ labeled C functions.

**Key learnings from this project:**
- Multi-model ensemble challenges and calibration issues
- The critical importance of data quality and proper evaluation
- Why 66% accuracy on a real problem teaches more than 99% on toy datasets
- Production ML is 80% engineering, 20% modeling

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
python scripts/train_all_models.py
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
# Output: Prediction: VULNERABLE, Confidence: 87%
```

**CLI Usage:**
```bash
# Analyze code string
python -m src.inference --code "query = 'SELECT * WHERE id=' + uid"

# Analyze file
python -m src.inference --file vulnerable_code.c --model graphcodebert
```

## 📁 Project Structure
```
vulnai/
├── data/              # Data download and preprocessing
├── src/               # Core source code
│   ├── config.py      # Configuration
│   ├── models.py      # Model wrappers
│   ├── train.py       # Training logic
│   ├── evaluate.py    # Evaluation utilities
│   └── inference.py   # Inference interface
├── notebooks/         # Jupyter experiments
├── scripts/           # Training/evaluation scripts
└── results/           # Saved models and metrics
```

## 🔬 Models

### CodeBERT
- **Architecture:** BERT pre-trained on code (6 programming languages)
- **Parameters:** 125M
- **Strength:** Keyword-based patterns (e.g., `strcpy`, `eval`)
- **Weakness:** Misses structural vulnerabilities

### GraphCodeBERT ⭐ (Best F1)
- **Architecture:** Graph-aware BERT (understands data flow)
- **Parameters:** 125M
- **Strength:** Structural vulnerabilities (loops without bounds, control flow issues)
- **Weakness:** Slower inference due to graph construction

### CodeT5
- **Architecture:** T5 encoder-decoder adapted for code
- **Parameters:** 220M
- **Strength:** Larger capacity for complex patterns
- **Weakness:** Encoder-decoder not ideal for classification tasks

## 📚 Dataset

**Devign** (Microsoft Research, 2019)
- **Size:** 21,854 C functions from real projects
- **Source:** QEMU, FFmpeg, Linux kernel, Pidgin
- **Labels:** Binary (vulnerable / safe)
- **Split:** 80% train, 10% val, 10% test
- **Quality assurance:** Zero data leakage verified (aggressive deduplication)

## ⚠️ Limitations & Lessons Learned

### What Didn't Work:

❌ **Ensemble failed (50% accuracy = random guessing)**
- Root cause: Model disagreement without proper calibration
- Learning: Ensemble methods require careful probability calibration
- Fix needed: Platt scaling or meta-model stacking

❌ **CodeT5 underperformed despite more parameters**
- Root cause: Encoder-decoder architecture not suited for classification
- Learning: Model design matters more than parameter count

❌ **Gap from production tools (80-88% accuracy)**
- Root cause: Limited training data, no domain-specific tuning
- Learning: Commercial tools have years of optimization and larger datasets

### What Worked:

✅ **Proper data preprocessing prevented inflated results**
- Initial 100% accuracy revealed data leakage (296 overlapping samples)
- Aggressive deduplication and different random seeds fixed it

✅ **GraphCodeBERT's structural understanding proved valuable**
- Best F1 score (64.63%) despite same parameter count as CodeBERT
- Confirms: code structure matters for vulnerability detection

✅ **Complete pipeline demonstrates production thinking**
- Data → Training → Evaluation → Inference
- 80% of work was engineering, not modeling

## 🎓 Key Takeaways

> "Production ML is 20% modeling, 80% infrastructure" — This project proved it

1. **Data quality > Model architecture** - Clean, deduplicated data matters more than parameter count
2. **The right metric matters** - Accuracy misled on imbalanced data; F1 told the truth
3. **Ensembles aren't magic** - Require calibration and can fail spectacularly
4. **Ship it** - Perfect is the enemy of done; 66% accuracy teaches more than endless optimization

## 🔮 Future Improvements

If I were to continue this project:

- [ ] Fix ensemble with Platt scaling or stacking
- [ ] Implement Graph Neural Networks for AST/CFG analysis
- [ ] Add explainability (attention visualization, LIME)
- [ ] Combine with static analysis tools (hybrid approach)
- [ ] Expand to multi-language support (Python, JavaScript, Java)
- [ ] Active learning on uncertain samples
- [ ] Longer training (10 epochs instead of 3)

## 📖 Read More

**Full writeup on Medium:**  
👉 [Building a Multi-Model Ensemble for Code Vulnerability Detection: Lessons from Fine-Tuning CodeBERT, GraphCodeBERT, and CodeT5](https://medium.com/@amrgabeerr20/building-a-multi-model-ensemble-for-code-vulnerability-detection-lessons-from-fine-tuning-2da63c7fc279)

The article covers:
- Why the ensemble failed and what I learned
- Data leakage horror story (and how I fixed it)
- Comparison with commercial tools
- Honest discussion of when to ship vs. optimize


## 🙏 Acknowledgments

- **Microsoft Research** for the Devign dataset ([paper](https://arxiv.org/abs/1909.03496))
- **HuggingFace** for transformer implementations
- **Kaggle** for free T4 GPU compute (5 hours of training)

## 📞 Contact

- **Author:** Amr Gaber
- **Medium:** [@amrgabeerr20](https://medium.com/@amrgabeerr20)

## 🔗 Related Papers

- **Devign:** [Graph Neural Networks for Vulnerability Detection](https://arxiv.org/abs/1909.03496)
- **CodeBERT:** [CodeBERT: A Pre-Trained Model for Programming and Natural Languages](https://arxiv.org/abs/2002.08155)
- **GraphCodeBERT:** [GraphCodeBERT: Pre-training Code Representations with Data Flow](https://arxiv.org/abs/2009.08366)
- **CodeT5:** [CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models](https://arxiv.org/abs/2109.00859)

---

⚠️ **Disclaimer:** This is an experimental research project demonstrating ML methodology. Not intended for production security scanning. Always combine automated tools with human security review.

---

<div align="center">

**If this project helped you learn about ML engineering, star it! ⭐**

**Questions? Open an issue or read the [Medium article](https://medium.com/@amrgabeerr20/building-a-multi-model-ensemble-for-code-vulnerability-detection-lessons-from-fine-tuning-2da63c7fc279).**

</div>