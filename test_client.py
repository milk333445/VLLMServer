from openai import OpenAI


client = OpenAI(
    base_url="http://10.204.245.170:8947/v1",
    api_key="EMPTY"
)

# test vllm server
completion = client.chat.completions.create(
  model="Qwen2-7B-Instruct",
  messages=[
    {"role": "user", "content": "你好"}
  ]
)

print(completion.choices[0].message)

# test embedding model
text = "The food was delicious and the waiter was friendly."
response = client.embeddings.create(
    input = [text, text],
    model = "m3e-base"
)

print(response.data[0].embedding)