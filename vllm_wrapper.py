from transformers import AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from vllm import LLM, SamplingParams
import os
import copy

        
class vLLMWrapper:
    def __init__(self, 
                 model_dir,
                 tensor_parallel_size=1,
                 gpu_memory_utilization=0.9,
                 dtype='float16',
                 quantization=None):
        self.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.stop_words_ids = self.generation_config.eos_token_id
        
        os.environ['VLLM_USE_MODELSCOPE'] = 'True'
        self.model = LLM(model=model_dir,
                         tokenizer=model_dir,
                         tensor_parallel_size=tensor_parallel_size,
                         trust_remote_code=True,
                         quantization=quantization,
                         gpu_memory_utilization=gpu_memory_utilization,
                         dtype=dtype,
                         max_model_len=6816)
        
    def initialize_history(self, system_message, user_message):
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
    
    def chat(self, query, history=None, system="You are a helpful assistant.", temperature=0.9, max_tokens=512):
        torch.cuda.empty_cache()    
        if history is None:
            history = self.initialize_history(system, query)
        else:
            history.append({"role": "user", "content": query})
        
        text = self.tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer(text, return_tensors="pt")
        
        sampling_params=SamplingParams(stop_token_ids=self.generation_config.eos_token_id, 
                                    early_stopping=False,
                                    top_p=self.generation_config.top_p,
                                    temperature=temperature,
                                    repetition_penalty=self.generation_config.repetition_penalty,
                                    max_tokens=max_tokens)
        input_list = [
            model_inputs.input_ids[0].tolist()
        ]
        
        outputs = self.model.generate(
            prompt_token_ids=input_list,
            sampling_params=sampling_params,
            use_tqdm=False
        )
        response_token_ids = remove_stop_words(outputs[0].outputs[0].token_ids, self.stop_words_ids)
        response = self.tokenizer.decode(response_token_ids, skip_special_tokens=True)
        
        history.append({"role": "system", "content": response})
        return response, history


def remove_stop_words(token_ids,stop_words_ids):
    token_ids=copy.deepcopy(token_ids)
    while len(token_ids)>0:
        if token_ids[-1] in stop_words_ids:
            token_ids.pop(-1)
        else:
            break
    return token_ids
          
        
if __name__ == "__main__":
    model_dir = 'Qwen1.5-1.8B-Chat-GPTQ-Int4'
    qwenwrapper = vLLMWrapper(
        model_dir,
        dtype='float16',
        tensor_parallel_size=1,
        gpu_memory_utilization=0.1
        )
    history=None 
    while True:
        input_text = input("請輸入你的問題: ")  
        if input_text.lower() == "quit":  
            break
        response, history = qwenwrapper.chat(query=input_text,
                                            history=history)
        print(response)
        history=history[:20]