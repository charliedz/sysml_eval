


import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, cache_utils
import torch
import re
import random
import json
import time
import typing
from collections.abc import Sequence
from huggingface_hub import login
import csv
from tqdm import tqdm
from torch.serialization import safe_globals

CUDA_LAUNCH_BLOCKING=1

### CoT Prompt String. 
preamble = "As an expert problem solver, solve step by step the following mathematical questions."
q1 = "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. "\
    "After they are done, there will be 21 trees. How many trees did the grove workers plant today?\n" + \
    "A: Let's think step by step. There are 15 trees originally. Then there were 21 trees after some more were planted. "\
    "So there must have been 21 - 15 = 6. The answer is 6."

q2 = "Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are "\
    "in the parking lot?\n" + \
    "A: Let's think step by step. There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."

q3 = "Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do "\
    "they have left in total?\n" + \
    "A: Let's think step by step. Originally, Leah had 32 chocolates. Her sister had 42. So in total they had " \
    "32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."

q4 = "Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "\
    "How many lollipops did Jason give to Denny?\n" + \
    "A: Let's think step by step. Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "\
    "So he gave Denny 20 - 12 = 8. The answer is 8."

q5 = "Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "\
    "How many toys does he have now?\n" + \
    "A: Let's think step by step. Shawn started with 5 toys. If he got 2 toys each from his mom and dad, "\
    "then that is 4 more toys. 5 + 4 = 9. The answer is 9."

q6 = "Q: There were nine computers in the server room. Five more computers were installed each day, "\
    "from Monday to Thursday. How many computers are now in the server room?\n" + \
    "A: Let's think step by step. There were originally 9 computers. For each of 4 days, 5 more computers were added. "\
    "So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29."

q7 = "Q: Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, " \
    "he lost 2 more. How many golf balls did he have at the end of Wednesday?\n" + \
    "A: Let's think step by step. Michael started with 58 golf balls. After losing 23 on Tuesday, he had 58 - 23 = 35. "\
    "After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."

q8 = "Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\n" + \
    "A: Let's think step by step. Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "\
    "So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."
### / CoT Prompt String. 

COT_QS = (q1, q2, q3, q4, q5, q6, q7, q8)
COT_PROMPT = preamble+'\n'+q1+'\n'+q2+'\n'+q3+'\n'+q4+'\n'+q5+'\n'+q6+'\n'+q7+'\n'+q8+'\n'
LTSBS = "Let's think step by step. "

ANSWER_PHRASE = "The answer is"
NUM_RE = re.compile(r"-?\d+[\d,]*\.?\d*")
TEST_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
MAX_TOKENS = 400
DEVICE = 'cuda'
SAMPLES = 25 #TODO: Change this

EVALUATION_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(EVALUATION_DIR)
GSM_SYMBOLIC_DATA = os.path.join(EVALUATION_DIR, "data/Symbolic_Templates_26-50.json")

GEMMA_DIR = os.path.join(EVALUATION_DIR, "models/gemma-2-2b-it")
MATHSTRAL_DIR = os.path.join(EVALUATION_DIR, "models/Mathstral-7B-v0.1")
RHO_DIR = os.path.join(EVALUATION_DIR, "models/rho-math-1b-v0.1")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-2b-it",
        help="hf model-name",
    )

    parser.add_argument(
        "--in",
        type=str,
        default="./data",
        help="directory with data",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="./timings",
        help="output directory",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable debugging"
    )


    parser.add_argument(
        "--t",
        default='0',
        help='extension to add to saved filename'
    )

    parser.add_argument(
        '--samples',
        type=int,
        help='Set the number of samples',
        default=SAMPLES
    )

    args = parser.parse_args()
    return args

def load_model(model_dir):
    '''Load the desired hf model/tokenizer and return'''    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir,torch_dtype=torch.float16)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer

def create_prompt(question):
    '''
    generate the prompt for a given question 
    (here, question is from the gsm8k json, maybe? really just a string 
    that has as question with no whitespace)
    '''
    # just in case, TODO remove (cdz)
    for q in question: 
        q.strip()
    prompt = []
    for q in question:
        prompt.append(COT_PROMPT + 'Q: ' + q + "\n" + "A: " + LTSBS)
    return prompt


def load_prompt_kv(path: str, device: torch.device):
    # Load the data from the file onto the CPU
    with safe_globals([cache_utils.HybridCache]):
        data = torch.load(path, map_location='cpu', weights_only=False)

    pkv_cpu = data["past_key_values"]
    prompt_len = data["prompt_len"]
    
    # Validate cache sequence length
    if isinstance(pkv_cpu, cache_utils.HybridCache):
        seq_len = pkv_cpu.key_cache[0].shape[2]  # Access seq_len via key_cache
    else:
        raise TypeError(f"Unexpected type for past_key_values: {type(pkv_cpu)}")
    
    if seq_len != prompt_len:
        raise ValueError(f"Cache seq_len {seq_len} does not match prompt_len {prompt_len}")
    
    # Move the cache to the specified device
    try:
        pkv_gpu = pkv_cpu.to(device)
    except Exception:
        pkv_gpu = pkv_cpu  # Already on GPU
    
    print(f"Loaded prompt KV cache from {path}, prompt_len={prompt_len}, cache_shape={pkv_cpu.key_cache[0].shape}")
    return pkv_gpu, prompt_len


def kv_precompute_decode(model,
                         modelname,
                         tokenizer,
                         prompts,
                         max_tokens = MAX_TOKENS
                         ):
    """
    Use precomputed KV cache and then decode
    
    Args:
        model: The loaded language model.
        tokenizer: The tokenizer corresponding to the model.
        prompts: List of question strings to process.
        max_tokens: Maximum number of tokens to generate per response.
    
    Returns:
        List of dictionaries containing timing information for each question.
    """
    timings = []
    # Load precomputed KV cache for CoT prompt and move to GPU
    cache = os.path.join(EVALUATION_DIR, "models", "cached_prompt_kv.pt")
    print(f"debug: cache correct: {os.path.exists(cache)}")
    
    pkv_cot, prompt_len_cot = load_prompt_kv(cache, DEVICE)
    
    for i, question in enumerate(prompts):
        start_time = time.time()
        
        full_prompt = create_prompt([question])[0]
        
        tokenization_start = time.time()
        input_ids_full = tokenizer(full_prompt, return_tensors='pt').input_ids.to(DEVICE)
        tokenization_time = time.time() - tokenization_start
        
        # extract tokens after the CoT prompt to process
        input_ids_new = input_ids_full[:, prompt_len_cot:]
        
        # compute KV cache for the question part using precomputed CoT KV cache
        kv_start = time.time()
        with torch.no_grad():
            outputs = model(input_ids_new, past_key_values=pkv_cot)
        updated_past = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        kv_time = time.time() - kv_start
        
        # decode
        generated_ids = []
        past = updated_past
        rest_gen_start = time.time()
        for _ in range(max_tokens):
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            generated_ids.append(next_token)
            if next_token.item() == tokenizer.eos_token_id:
                break
            input_ids_next = next_token
            with torch.no_grad():
                outputs = model(input_ids=input_ids_next, past_key_values=past)
            past = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
        rest_gen_time = time.time() - rest_gen_start
        
        total_time = time.time() - start_time
        
        question_length = input_ids_new.shape[1]
        timings.append({'index': i,
                        'question_length': question_length,
                        'tokenization_time': tokenization_time,
                        'kv_time': kv_time,
                        'rest_gen_time': rest_gen_time,
                        'generation_time': kv_time + rest_gen_time,
                        'total_time': total_time
                        })
    
    return timings


def main(argvs,i):
    global SAMPLES
    DEBUGGING = argvs.debug
    SAMPLES = argvs.samples
    isDecoder = True if any(x in argvs.model.split('/')[1].lower() for x in
                             ('llama','gemma')) else False

    if argvs.model == "google/gemma-2-2b-it":
        model_path = GEMMA_DIR
    elif argvs.model == "mistralai/Mathstral-7B-v0.1":
        model_path = MATHSTRAL_DIR
    elif argvs.model == "microsoft/rho-math-1b-v0.1":
        model_path = RHO_DIR

    model, tokenizer = load_model(model_path)
    
    split_modname = argvs.model.split('/')
    save_modname = split_modname[1].strip()
    os.makedirs(argvs.out, exist_ok=True)

    tests = []
    with open(GSM_SYMBOLIC_DATA,'r') as f:
        loaded_data=json.load(f)
    # for each of the 25 templates, pick NUM_SAMPS
    for i in range(25):
        j = 0
        indxs = [i for i in range(1,51)]
        while j < SAMPLES:
            choice = random.choice(indxs)
            indxs.remove(choice)
            tests.append((loaded_data[i]['question_'+str(choice)],loaded_data[i]['answer_'+str(choice)]))
            j += 1

    timings = kv_precompute_decode(model, save_modname, tokenizer, [t[0] for t in tests], max_new_tokens=MAX_TOKENS)
        
    csv_dir = os.path.join(argvs.out, "timing_logs")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"{save_modname}_{i}.csv")

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, 
                                fieldnames=['index', 'question_length', 'tokenization_time',
                                            'kv_time', 'rest_gen_time', 'generation_time', 'total_time'])
        writer.writeheader()
        for row in timings:
            writer.writerow(row)

    print('done!')

if __name__ == "__main__":
    ### cli functionality here. 
    argvs = parse_args()
    main(argvs,argvs.t)
