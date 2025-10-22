"""
AWS Lambda handler - routes requests to appropriate backend
INPUT: API Gateway event
OUTPUT: JSON response with predictions or analysis
"""
import json
import boto3
from .bedrock_interface import BedrockLLM
from .pandasai_wrapper import TabularAgent

# Initialize once (Lambda reuses containers)
bedrock = BedrockLLM()
agent = TabularAgent()

def lambda_handler(event, context):
    """
    INPUT event format (from API Gateway):
    {
        "body": {
            "query": "What's the default risk for customer X?",
            "data": [[age, job, credit_amount, ...], ...],  # or CSV string
            "mode": "predict" | "analyze" | "explain"
        }
    }
    
    OUTPUT format:
    {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": {
            "result": <prediction or analysis>,
            "method": "intention" | "pandasai",
            "confidence": <optional>,
            "explanation": <optional>
        }
    }
    """
    try:
        # Parse request
        body = json.loads(event.get('body', '{}'))
        query = body.get('query', '')
        data = body.get('data')
        mode = body.get('mode', 'analyze')
        
        # Route to appropriate backend
        if mode == 'predict':
            # Use Intention model directly
            result = agent.predict_with_intention(data)
            method = 'intention'
        else:
            # Use Bedrock + PandasAI for analysis
            result = agent.analyze_with_llm(query, data)
            method = 'pandasai'
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'result': result,
                'method': method,
                'architecture': 'Intention-based Informer'
            })
        }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
