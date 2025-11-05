import boto3
import json

# 1. Create the client for Bedrock "runtime"
# Use the same region that worked before
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)

# 2. Define the model ID you want to use
# We got this ID from your list:
model_id = 'anthropic.claude-haiku-4-5-20251001-v1:0'

# 3. Define the prompt and parameters
# Claude models use the "Messages" API
prompt = "Olá! Explique o que é Machine Learning em uma frase."

# This is the JSON body format specific for Claude 3/4
body = {
    # Bedrock API version for Anthropic
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 1000,
    "messages": [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]
}

# 4. Invoke the model
try:
    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        contentType='application/json',  # Type of what we are sending
        accept='application/json',      # Type of what we want to receive
        body=json.dumps(body)           # The body needs to be a JSON string
    )

    # 5. Read and interpret the response
    # The response comes in a 'body' field which is a streaming object
    response_body_raw = response.get('body').read()
    response_body = json.loads(response_body_raw)

    # The Claude response text is inside response_body['content']
    resposta_claude = response_body['content'][0]['text']

    print(f"Prompt: {prompt}\n")
    print(f"Resposta do Claude Haiku:\n{resposta_claude}")

except Exception as e:
    print(f"Erro ao invocar o modelo: {e}")
