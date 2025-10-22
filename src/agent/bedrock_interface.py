"""
Wraps AWS Bedrock API for LLM calls
"""
import boto3
import json

class BedrockLLM:
    def __init__(self, model_id='anthropic.claude-3-sonnet-20240229-v1:0'):
        self.client = boto3.client('bedrock-runtime', region_name='us-east-1')
        self.model_id = model_id
    
    def invoke(self, prompt):
        """
        INPUT: String prompt
        OUTPUT: LLM response text
        """
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
        })
        
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=body
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
    
    # Make it compatible with PandasAI interface
    def chat(self, prompt):
        return self.invoke(prompt)
