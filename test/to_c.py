from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# https://huggingface.co/arise-sustech/llm4decompile-6.7b-uo/tree/main
model_path = './model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16).cuda()

with open('test/EnableSlashAtCommands.asm','r') as f:#original file
    asm_func = f.read()
inputs = tokenizer(asm_func, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=500)
c_func_decompile = tokenizer.decode(outputs[0][len(inputs[0]):-1])



with open('test/EnableSlashAtCommands.c', 'w+', encoding='UTF-8') as file :
    file.write(c_func_decompile)
    