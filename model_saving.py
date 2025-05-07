from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os

HF_TOKEN = ""
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
print(FILE_PATH)
SAVE_PATH = os.path.join(FILE_PATH, 'saved_models')

SAVE_PATH_MODEL = os.path.join(SAVE_PATH, 'model')
SAVE_PATH_TOKENIZER = os.path.join(SAVE_PATH, 'tokenizer')

login(HF_TOKEN)

model_name = "microsoft/rho-math-1b-v0.1"  # e.g., "meta-llama/Llama-2-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.save_pretrained(SAVE_PATH_TOKENIZER)
model.save_pretrained(SAVE_PATH_MODEL)



