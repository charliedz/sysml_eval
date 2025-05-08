""" BATCHING """

import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
import json
from collections.abc import Sequence
import csv
from tqdm import tqdm

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

MAX_TOKENS = 400
DEVICE = 'cuda'
SAMPLES = 2
BATCH_SIZE = 4

PHRASE = "The answer is"
MAX_TOKENS = 400
DEVICE = 'cuda'
SAMPLES = 2

EVALUATION_DIR = os.path.dirname(os.path.abspath(__file__))
CKPTS_DIR = os.path.join(EVALUATION_DIR, "ckpts")
TIMINGS_DIR = os.path.join(EVALUATION_DIR, "timings")

_BASE_DIR = os.path.dirname(EVALUATION_DIR)
GSM_SYMBOLIC_DATA = os.path.join(EVALUATION_DIR, "data/Symbolic_Templates_26-50.json")

GEMMA_DIR = os.path.join(EVALUATION_DIR, "models/gemma-2-2b-it")
MATHSTRAL_DIR = os.path.join(EVALUATION_DIR, "models/Mathstral-7B-v0.1")
RHO_DIR = os.path.join(EVALUATION_DIR, "models/rho-math-1b-v0.1")
GEMMA_7B_DIR = os.path.join(EVALUATION_DIR, "models/gemma-7b-it")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-2b-it",
        help="hf model-name",
    )

    parser.add_argument(
        '--samples',
        type=int,
        help='Set the number of samples',
        default=SAMPLES
    )

    parser.add_argument(
        '--batch-size', type=int, default=BATCH_SIZE,
        help='Batch size for tokenization and generation',
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

def kv_decode_batched(model,
                      model_name: str,
                      tokenizer,
                      prompts: Sequence[str],
                      max_new_tokens: int = 400,
                      batch_size: int = 1
                      ) -> Sequence[dict]:
    timing_data = []
    total_prompts = len(prompts)

    # create a sample-level progress bar
    progress = tqdm(total=total_prompts, desc="Generating", unit="prompt")

    # process in batches
    for batch_start in range(0, total_prompts, batch_size):
        batch_end = min(batch_start + batch_size, total_prompts)
        batch_prompts = prompts[batch_start:batch_end]
        actual_batch = len(batch_prompts)

        torch.cuda.empty_cache()

        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        # tokenize and pad
        inputs = tokenizer(batch_prompts,
                           return_tensors="pt",
                           padding=True,
                           truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        t1.record()
        torch.cuda.synchronize()
        batch_token_time = t0.elapsed_time(t1) / 1000.0
        token_time_per = batch_token_time / actual_batch

        gen_t0 = torch.cuda.Event(enable_timing=True)
        gen_t1 = torch.cuda.Event(enable_timing=True)
        gen_t0.record()
        _ = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True
        )
        gen_t1.record()
        torch.cuda.synchronize()
        batch_gen_time = gen_t0.elapsed_time(gen_t1) / 1000.0
        gen_time_per = batch_gen_time / actual_batch

        for idx_in_batch in range(actual_batch):
            global_idx = batch_start + idx_in_batch
            q_len = inputs["input_ids"][idx_in_batch].ne(tokenizer.pad_token_id).sum().item()
            timing_data.append({
                'index': global_idx,
                'question_length': q_len,
                'tokenization_time': token_time_per,
                'kv_time': 0.0,
                'rest_gen_time': 0.0,
                'generation_time': gen_time_per,
                'total_time': token_time_per + gen_time_per
            })

        progress.update(actual_batch)

        last_idx = batch_end - 1
        if last_idx != 0 and last_idx % 10 == 0:
            os.makedirs(CKPTS_DIR, exist_ok=True)
            path = os.path.join(CKPTS_DIR, f"{model_name}_ckpt_{last_idx}.csv")
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=timing_data[0].keys())
                writer.writeheader()
                writer.writerows(timing_data)

    progress.close()
    return timing_data

def main(argvs):
    global SAMPLES
    SAMPLES = argvs.samples
    batch_size = argvs.batch_size

    if argvs.model == "google/gemma-2-2b-it":
        model_path = GEMMA_DIR
    elif argvs.model == "mistralai/Mathstral-7B-v0.1":
        model_path = MATHSTRAL_DIR
    elif argvs.model == "microsoft/rho-math-1b-v0.1":
        model_path = RHO_DIR
    elif argvs.model == "google/gemma-7b-it":
        model_path = GEMMA_7B_DIR

    model, tokenizer = load_model(model_path)
    
    split_modname = argvs.model.split('/')
    save_modname = split_modname[1].strip()

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

    timings = kv_decode_batched(model,
                                save_modname, 
                                tokenizer, 
                                [t[0] for t in tests], 
                                max_new_tokens=MAX_TOKENS, 
                                batch_size=batch_size)

    csv_path = os.path.join(TIMINGS_DIR, f"{save_modname}-kv-batch-3080.csv")

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, 
                                fieldnames=['index', 'question_length', 'tokenization_time',
                                            'kv_time', 'rest_gen_time', 'generation_time', 'total_time'])
        writer.writeheader()
        for row in timings:
            writer.writerow(row)

    print('done!')

if __name__ == "__main__":
    argvs = parse_args()
    main(argvs)
