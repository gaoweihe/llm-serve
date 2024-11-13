# app.py
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

torch.random.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
app = FastAPI()
model_name = "microsoft/Phi-3-mini-4k-instruct"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    # bnb_8bit_use_double_quant=True,  # If double quantization is supported in 8-bit, otherwise omit
    # bnb_8bit_quant_type="int8",       # Replace with the appropriate quantization type for Q8
    # bnb_8bit_compute_dtype=torch.int8
)

model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, quantization_config=bnb_config, torch_dtype="auto", trust_remote_code=True)

# Define the request model
class GenerateRequest(BaseModel):
    prompt: str
    
SYSTEM_PROMPT = r"""

### INSTRUCTIONS ###
You are an advanced AI assistant designed to provide informative and helpful responses to a wide range of queries.
Strictly following the following instructions:
1. Engage in a conversational manner and use humor if applicable.
2. Your responses should be visually appealing for WhatsApp users, so use emojis, short paragraphs, etc.
3. In the first line, you have to print the estimated number of words your response will be. In the next line start generating the response for the actual query.

    """
    
PIA_PROMPT = r"""
### INSTRUCTIONS ###
You are an advanced AI assistant designed to provide informative and helpful responses to a wide range of queries. 
Strictly following the following instructions:
1. Engage in a conversational manner and use humor if applicable.
2. Your responses should be visually appealing for WhatsApp users, so use emojis, short paragraphs, etc.
3. You will be given examples of queries and recommended response lengths (number of words). Use these example queries to understand the appropriate response length for the given query.
4. In the first line, you have to print the estimated number of words your response will be. In the next line start generating the response for the actual query.

### EXAMPLES ###

Question 0:[[maple syrup]]\nNumber of words: [[73.4]]\n\n    ---------------\n\nQuestion 1:[[Who is sanndman]]\nNumber of words: [[49.2]]\n\n    ---------------\n\nQuestion 2:[[Most widely read Surah of Quran]]\nNumber of words: [[46.2]]\n\n    ---------------\n\nQuestion 3:[[What are the most effective strategies for overcoming procrastination at work?]]\nNumber of words: [[92.0]]\n\n    ---------------\n\nQuestion 4:[[What's the typical profile of a student who gets admission into Harvard?]]\nNumber of words: [[83.6]]\n\n    ---------------\n\nQuestion 5:[[When will spitting end in pregnancy]]\nNumber of words: [[58.2]]\n\n    ---------------\n\nQuestion 6:[[If you could design a new planet, what features or characteristics would you give it to make it truly unique?]]\nNumber of words: [[79.6]]\n\n    ---------------\n\nQuestion 7:[[who is Alex snodgrass]]\nNumber of words: [[47.0]]\n\n    ---------------\n\nQuestion 8:[[Which LLM are you using?]]\nNumber of words: [[37.8]]\n\n    ---------------\n\nQuestion 9:[[bullet proof coffee]]\nNumber of words: [[55.8]]\n\n    ---------------\n\nQuestion 10:[[In 4 words, what sea connects Australia?]]\nNumber of words: [[4.6]]\n\n    ---------------\n\nQuestion 11:[[exit speed movie]]\nNumber of words: [[59.8]]\n\n    ---------------\n\nQuestion 12:[[What wss imam hambals view on createdness of quran. And what was the implication of believing otherwise.]]\nNumber of words: [[104.4]]\n\n    ---------------\n\nQuestion 13:[[chlorhexidine gluconate]]\nNumber of words: [[61.0]]\n\n    ---------------\n\nQuestion 14:[[What are ACAMS and CGSS]]\nNumber of words: [[66.2]]\n\n    ---------------\n\nQuestion 15:[[What's the purpose of life?]]\nNumber of words: [[67.8]]\n\n    ---------------\n\nQuestion 16:[[With a box spring do you need a spring mattress or foam?]]\nNumber of words: [[91.8]]\n\n    ---------------\n\nQuestion 17:[[what should be the lifestyle after removal of gallbladder]]\nNumber of words: [[82.8]]\n\n    ---------------\n\nQuestion 18:[[Should I not exercise after getting COVID and flu vaccines?]]\nNumber of words: [[53.8]]\n\n    ---------------\n\nQuestion 19:[[Summarize the main lessons from Surah Rahman in the Holy Quran for an 11-year old.]]\nNumber of words: [[72.8]]\n\n    ---------------\n\nQuestion 20:[[Best Netflix series to watch on real life events?]]\nNumber of words: [[57.2]]\n\n    ---------------\n\nQuestion 21:[[How to measure weather]]\nNumber of words: [[72.0]]\n\n    ---------------\n\nQuestion 22:[[How do I train for running up Mount Washington on the access road?]]\nNumber of words: [[118.2]]\n\n    ---------------\n\nQuestion 23:[[Compare Tucson with crv]]\nNumber of words: [[119.6]]\n\n    ---------------\n\nQuestion 24:[[What is a tietam cedar mountain and chancellorville]]\nNumber of words: [[134.2]]\n\n    ---------------\n\nQuestion 25:[[How to deal effectively with failures ?]]\nNumber of words: [[65.0]]\n\n    ---------------\n\nQuestion 26:[[What happens after polymerisation]]\nNumber of words: [[61.2]]\n\n    ---------------\n\nQuestion 27:[[In this day and age what are the best skills to learn to be successful]]\nNumber of words: [[164.4]]\n\n    ---------------\n\nQuestion 28:[[What can you tell a student who is under stress regarding their future?]]\nNumber of words: [[79.4]]\n\n    ---------------\n\nQuestion 29:[[If a woman gets their period late in life, does that also mean she will have menopause later in life?]]\nNumber of words: [[58.2]]\n\n    ---------------\n\n  \n\nBefore responding to the above instruction, you have to predict the length of your response. Print the estimated number of words in your response in the first line.  \n        \n        
"""

@app.post("/generate")
async def generate_text(request: GenerateRequest): 
    prompt = request.prompt
    messages = [
        {"role": "system", "content": PIA_PROMPT},
        {"role": "user", "content": prompt},
    ]
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    generation_args = {
        "max_new_tokens": 600,
        "return_full_text": False,
        "temperature": 0,
        "do_sample": False,
    }
    output = pipe(messages, **generation_args)
    generated_text = output[0]['generated_text']
    return {"generated_text": generated_text}
