import os 
from vllm import AsyncEngineArgs,AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from transformers import AutoTokenizer
from transformers.generation import GenerationConfig
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
import uuid
import json
import copy



app = FastAPI()

model_dir = './models' # Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4
quantization='gptq'
dtype='float16'
tensor_parallel_size=1
gpu_memory_utilization=0.6
max_model_len = 6816

temperature=0.9
max_tokens = 512

def load_vllm(model_dir, tensor_parallel_size, gpu_memory_utilization, dtype, quantization, max_model_len):
    generation_config = GenerationConfig.from_pretrained(model_dir,trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)

    args = AsyncEngineArgs(model_dir)
    args.worker_use_ray=False
    args.engine_use_ray=False
    args.tokenizer=model_dir
    args.tensor_parallel_size=tensor_parallel_size
    args.trust_remote_code=True
    args.quantization=quantization
    args.gpu_memory_utilization=gpu_memory_utilization
    args.dtype=dtype
    args.max_num_seqs=20
    args.max_model_len=max_model_len

    os.environ['VLLM_USE_MODELSCOPE']='True'
    engine=AsyncLLMEngine.from_engine_args(args)
    return generation_config, tokenizer, engine

generation_config, tokenizer, engine = load_vllm(model_dir, tensor_parallel_size, gpu_memory_utilization, dtype, quantization, max_model_len)

def match_user_stop_words(response_token_ids, user_stop_tokens):
    for stop_token in user_stop_tokens:
        if len(response_token_ids) < len(stop_token):
            continue
        if response_token_ids[-len(stop_token):] == stop_token: # 如果response_token_ids的最后len(stop_token)个token和stop_token一样，则返回True
            return True
    return False

def remove_stop_words(token_ids,stop_words_ids):
    token_ids=copy.deepcopy(token_ids)
    while len(token_ids)>0:
        if token_ids[-1] in stop_words_ids:
            token_ids.pop(-1)
        else:
            break
    return token_ids

@app.post("/chat")
async def chat(request: Request):
    request = await request.json()
    prompt = request.get('prompt', None)
    stream=request.get("stream",False)
    user_stop_words=request.get("user_stop_words",[])
    
    if prompt is None:
        return Response(status_code=502, content="Query is required")
        
    
    user_stop_tokens=[]
    for words in user_stop_words:
        user_stop_tokens.append(tokenizer.encode(words))
        
    text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer(text, return_tensors="pt")
        
    sampling_params=SamplingParams(stop_token_ids=generation_config.eos_token_id, 
                                    early_stopping=False,
                                    top_p=generation_config.top_p,
                                    temperature=temperature,
                                    repetition_penalty=generation_config.repetition_penalty,
                                    max_tokens=max_tokens)
    request_id = str(uuid.uuid4().hex) # hex表示返回一个16進制字符串
    text_tokens = model_inputs.input_ids[0].tolist()
    results_iter = engine.generate(
        prompt=None,
        sampling_params=sampling_params,
        prompt_token_ids=text_tokens,
        request_id=request_id
    )
    
    if stream:
        async def streaming_resp():
            async for result in results_iter:
                token_ids = remove_stop_words(result.outputs[0].token_ids, generation_config.eos_token_id)
                text = tokenizer.decode(token_ids, skip_special_tokens=True)
                yield (
                    json.dumps(
                        {
                            'text': text
                        }
                    ) + '\0'
                ).encode('utf-8')
                if match_user_stop_words(token_ids, user_stop_tokens):
                    await engine.abort(request_id)
                    break
        return StreamingResponse(streaming_resp())
          
    async for result in results_iter:
        token_ids = remove_stop_words(result.outputs[0].token_ids, generation_config.eos_token_id)
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        if match_user_stop_words(token_ids, user_stop_tokens):
            await engine.abort(request_id)
            break
        
    ret = {'text': text}
    return JSONResponse(ret)
                
if __name__=='__main__':
    uvicorn.run(app,
                host=None,
                port=8000,
                log_level="info",)
