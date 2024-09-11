from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = 'Qwen/Qwen1.5-14B-Chat'
model_dir = 'Qwen1.5-14B-Chat'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True).eval()
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)