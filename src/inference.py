"""
Inference utilities
"""

from typing import Optional
from pathlib import Path
from .models import VulnDetectorModel
from .config import MODELS_DIR

class VulnerabilityDetector:
    """High-level interface for vulnerability detection"""
    
    def __init__(self, model_name: str = 'graphcodebert', device: str = 'cuda'):
        """
        Initialize detector
        
        Args:
            model_name: Which model to use ('codebert', 'graphcodebert', 'codet5')
            device: 'cuda' or 'cpu'
        """
        self.model = VulnDetectorModel(model_name, device)
    
    def predict(self, code: str, return_details: bool = False) -> dict:
        """
        Analyze code for vulnerabilities
        
        Args:
            code: Source code to analyze
            return_details: Include detailed analysis
            
        Returns:
            Prediction result dictionary
        """
        result = self.model.predict(code)
        
        if return_details:
            result['vulnerability_type'] = self._classify_vuln_type(code, result)
        
        return result
    
    def _classify_vuln_type(self, code: str, prediction: dict) -> Optional[str]:
        """Heuristic vulnerability type classification"""
        if prediction['prediction'] == 'SAFE':
            return None
        
        code_lower = code.lower()
        
        # SQL Injection
        if any(kw in code_lower for kw in ['select', 'insert', 'update', 'delete']):
            if '+' in code or 'concat' in code_lower:
                return 'SQL Injection'
        
        # XSS
        if 'innerhtml' in code_lower or 'eval' in code_lower:
            return 'Cross-Site Scripting (XSS)'
        
        # Buffer Overflow
        if any(kw in code_lower for kw in ['strcpy', 'gets', 'sprintf']):
            return 'Buffer Overflow'
        
        # Command Injection
        if any(kw in code_lower for kw in ['system', 'exec', 'shell']):
            return 'Command Injection'
        
        return 'Unknown Vulnerability'


# Simple CLI interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='VulnAI - Code Vulnerability Detector')
    parser.add_argument('--code', type=str, help='Code to analyze')
    parser.add_argument('--file', type=str, help='File containing code')
    parser.add_argument('--model', type=str, default='graphcodebert',
                       choices=['codebert', 'graphcodebert', 'codet5'])
    
    args = parser.parse_args()
    
    # Get code
    if args.file:
        with open(args.file, 'r') as f:
            code = f.read()
    elif args.code:
        code = args.code
    else:
        print("Error: Provide --code or --file")
        return
    
    # Analyze
    detector = VulnerabilityDetector(model_name=args.model)
    result = detector.predict(code, return_details=True)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.1%}")
    
    if result.get('vulnerability_type'):
        print(f"Type: {result['vulnerability_type']}")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()