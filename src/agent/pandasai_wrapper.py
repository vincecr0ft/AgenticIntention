"""
Wraps PandasAI with Bedrock LLM for tabular analysis
Also routes to Intention model when needed
"""
import pandas as pd
import torch
from pandasai import SmartDataframe
from .bedrock_interface import BedrockLLM
from ..models.intention import load_intention_model

class TabularAgent:
    def __init__(self):
        self.llm = BedrockLLM()
        self.intention_model = None  # Load on demand
    
    def predict_with_intention(self, data):
        """
        INPUT: List of feature vectors or pandas DataFrame
        OUTPUT: Predictions using Intention model
        """
        if self.intention_model is None:
            self.intention_model = load_intention_model()
        
        df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
        # Your Intention model inference
        with torch.no_grad():
            preds = self.intention_model(torch.tensor(df.values))
        return preds.tolist()
    
    def analyze_with_llm(self, query, data):
        """
        INPUT: Natural language query + tabular data
        OUTPUT: LLM-generated analysis using PandasAI
        """
        df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
        sdf = SmartDataframe(df, config={'llm': self.llm})
        return sdf.chat(query)
