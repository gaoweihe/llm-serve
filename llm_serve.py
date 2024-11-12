# app.py
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
app = FastAPI()
model_name = "microsoft/Phi-3-mini-4k-instruct"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,  # If double quantization is supported in 8-bit, otherwise omit
    bnb_8bit_quant_type="int8",       # Replace with the appropriate quantization type for Q8
    bnb_8bit_compute_dtype=torch.int8
)

model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=torch.bfloat16, quantization_config=bnb_config,trust_remote_code=True)

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
        max_new_tokens=200, 
        temperature=0.7)
    generated_text = tokenizer.decode(outputs[0])
    return {"generated_text": generated_text}
