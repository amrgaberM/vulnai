"""
Model wrapper classes
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .config import MODEL_CONFIGS

class VulnDetectorModel:
    """Wrapper for vulnerability detection models"""
    
    def __init__(self, model_name: str, device: str = 'cuda'):
        """
        Initialize model
        
        Args:
            model_name: One of 'codebert', 'graphcodebert', 'codet5'
            device: 'cuda' or 'cpu'
        """
        self.model_name = model_name
        self.device = device
        self.config = MODEL_CONFIGS[model_name]
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config['model_name'],
            num_labels=2,
            problem_type="single_label_classification"
        ).to(device)
        
        self.model.eval()
    
    def predict(self, code: str) -> dict:
        """
        Predict if code is vulnerable
        
        Args:
            code: Source code string
            
        Returns:
            dict with prediction, confidence, probabilities
        """
        # Tokenize
        inputs = self.tokenizer(
            code,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config['max_length']
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            prediction = torch.argmax(probs).item()
            confidence = probs[prediction].item()
        
        return {
            'prediction': 'VULNERABLE' if prediction == 1 else 'SAFE',
            'prediction_class': prediction,
            'confidence': confidence,
            'probabilities': {
                'safe': float(probs[0]),
                'vulnerable': float(probs[1])
            }
        }
    
    @classmethod
    def from_pretrained(cls, model_path: str, device: str = 'cuda'):
        """Load from saved checkpoint"""
        # Implementation for loading saved models
        pass