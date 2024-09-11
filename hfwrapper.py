from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import time
import numpy as np

    
class QwenWrapper:
    def __init__(self, model_dir):
        """
        {
        "bos_token_id": 151643,
        "do_sample": true,
        "eos_token_id": [
            151645,
            151643
        ],
        "pad_token_id": 151643,
        "repetition_penalty": 1.1,
        "top_p": 0.8,
        "transformers_version": "4.43.4"
        }
        """
        
        warm_start_time = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)
        
        self.warm_up_time = time.time()-warm_start_time
        self.response_times = []
        self.tokens_response_times = []


        self.tokenizer.eos_token_id = self.generation_config.eos_token_id
        
    def chat(self, query, history=None, system="You are a helpful assistant.", temperature=0.9, max_tokens=512):
        torch.cuda.empty_cache()
        if history is None:
            history = [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ]
        else:
            history.append({"role": "user", "content": query})
        
        end_to_end_start_time = time.time()
        
        text = self.tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        inference_start_time = time.time()
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        inference_time = time.time()-inference_start_time
        self.response_times.append(inference_time)
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        num_tokens_generated = sum(len(ids) for ids in generated_ids)
        tokens_per_second = self.get_token_generation_speed(num_tokens_generated, inference_time)
        self.tokens_response_times.append(tokens_per_second)
        
        end_to_end_time = time.time()-end_to_end_start_time
        self.response_times.append(end_to_end_time)
        
        print(f"Inference Latency: {inference_time:.4f} seconds")
        print(f"End-to-End Latency: {end_to_end_time:.4f} seconds")
        print(f"Current GPU Memory Usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Tokens Per Second: {tokens_per_second:.2f} tokens/second")
        
        history.append(
            {
                "role": "system",
                "content": response,
            }
        )
        del model_inputs
        torch.cuda.empty_cache()
        
        return response, history
    
    def get_average_response_time(self):
        return np.mean(self.response_times)

    def get_response_time_variability(self):
        return np.var(self.response_times)

    def get_token_generation_speed(self, num_tokens, inference_time):
        return num_tokens / inference_time
    
    def get_average_token_generation_speed(self):
        return np.mean(self.tokens_response_times)
    
    
if __name__ == "__main__":
    model_dir = './models'
    qwenwrapper = QwenWrapper(model_dir)
    
    history = [] 
    for i in range(5):
        query = f"這是第{i+1}次對話，請告訴我更多信息。"
        response, history = qwenwrapper.chat(query=query, history=history)
        history = history[:20]
        print(f'第{i+1}次對話結果: {response}')
        
    print(f'Averge Response Time: {qwenwrapper.get_average_response_time()}')
    print(f'Response Time Variability: {qwenwrapper.get_response_time_variability()}')
    print(f'Average Token Generation Speed: {qwenwrapper.get_average_token_generation_speed()}')
    
    
    
    # while True: 
    #     input_text = input("請輸入你的問題: ")  
    #     if input_text.lower() == "quit":  
    #         break
        
    #     response, history = qwenwrapper.chat(input_text, history=history)
    #     history = history[:20]
    #     print("Model response:", response)

