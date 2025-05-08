from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os

HF_TOKEN = "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"
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
print(FILE_PATH)
SAVE_PATH = os.path.join(FILE_PATH, 'saved_models')

login(HF_TOKEN)

model_name = "google/gemma-7b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)

