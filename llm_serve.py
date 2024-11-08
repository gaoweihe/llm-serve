# app.py
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cuda")
app = FastAPI()
model_name = "meta-llama/Meta-Llama-3-8B"  # or any Hugging Face model name
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the request model
class GenerateRequest(BaseModel):
    prompt: str
    
SYSTEM_PROMPT = "You are a helpful assistant who provides concise, informative responses."

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    prompt = f"{SYSTEM_PROMPT}\n\n{request.prompt}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs, 
        max_new_tokens=50, 
        temperature=0.7)
    generated_text = tokenizer.decode(outputs[0])
    return {"generated_text": generated_text}
