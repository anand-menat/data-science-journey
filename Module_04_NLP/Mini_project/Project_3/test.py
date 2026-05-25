import os
import boto3
from dotenv import load_dotenv

# 1. Force Python to read the local .env file
load_dotenv()

# 2. Initialize the client normally. 
# Boto3 automatically looks for AWS_BEARER_TOKEN_BEDROCK in your environment variables.
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
)

print("🚀 Successfully authenticated natively using your Bedrock API Token!")

# 3. CHOOSE YOUR MODEL:
# Make sure to copy these EXACT strings. A single wrong dash causes a ValidationException.
# Option A (Amazon Nova Micro): "us.amazon.nova-micro-v1:0"
# Option B (Claude 3.5 Haiku):  "us.anthropic.claude-3-5-haiku-20241022-v1:0"

SELECTED_MODEL = "eu.amazon.nova-micro-v1:0"
print(f"Sending request to: {SELECTED_MODEL}...")

try:
    # 4. Invoke the Converse API
    response = bedrock_runtime.converse(
        modelId=SELECTED_MODEL,
        messages=[
            {
                "role": "user",
                "content": [{"text": "Write a one-sentence bedtime story about a unicorn."}]
            }
        ],
        inferenceConfig={
            "temperature": 0.5,
            "maxTokens": 100
        }
    )
    
    # 5. Extract and print the generated text
    output_text = response["output"]["message"]["content"][0]["text"]
    print("\n✨ Response from Bedrock:")
    print(output_text)

except Exception as e:
    print(f"\n❌ Execution Error: {e}")