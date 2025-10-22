#!/bin/bash
# Quick AWS Lambda deployment

# Package dependencies
pip install -r deployment/requirements_lambda.txt -t package/
cp -r src/ package/
cd package && zip -r ../lambda_function.zip . && cd ..

# Create/update Lambda (requires AWS CLI configured)
aws lambda create-function \
    --function-name tabular-agent-intention \
    --runtime python3.11 \
    --role arn:aws:iam::YOUR_ACCOUNT:role/lambda-execution-role \
    --handler src.agent.lambda_handler.lambda_handler \
    --zip-file fileb://lambda_function.zip \
    --timeout 30 \
    --memory-size 512

# Create API Gateway endpoint (manual for now - use AWS Console)
echo "Lambda deployed! Create API Gateway endpoint in AWS Console"
