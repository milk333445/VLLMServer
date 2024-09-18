import json
import requests

def clear_lines():
    print('\033[2J')

prompt = []

while True:
    query = input("問題：")
    
    if query.lower() == "quit":
        break
    
    if prompt == []:
        prompt = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
    else:
        prompt.append({"role": "user", "content": query})
    
    response=requests.post('http://localhost:8000/chat',json={
        'prompt':prompt,
        'stream': True
    },stream=True)
    
    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b'\0'):
        if chunk:
            data = json.loads(chunk.decode('utf-8'))
            text = data['text'].rstrip('\r\n') # 去掉尾部的换行符
            
            clear_lines()
            print(text)
            
    prompt.append({"role": "system", "content": text})
    prompt = prompt[-5:]
    
    
