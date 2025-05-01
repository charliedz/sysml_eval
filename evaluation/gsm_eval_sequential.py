''' (cdz) cmsc723 final 2024 '''

# This script batches prompts and infers, rather than going one prompt at a time. 

import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
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
INVALD_RESPONSE = "<inv>"
VALID_CORRECT = "<vc>"
VALID_INCORRECT = "<vi>"
NUM_RE = re.compile(r"-?\d+[\d,]*\.?\d*")
TEST_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
HF_TOKEN = "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" # TODO : don't put this on gh
MAX_TOKENS = 600
DEVICE = 'cuda'
SEED = 58 # pick any here idk 
SAMPLES = 25 #TODO: Change this
GSM8K_DATA = './data/gsm8k_test.jsonl'
GSM_SYMBOLIC_DATA = "./data/Symbolic_Templates_26-50.json"
EVALUATION_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-2b-it",
        help="hf model-name",
    )
    # TODO (cdz) : this isn't used; maybe just use the json path?
    #              or not, with symbolic templates in mind
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
        "--dq",
        action="store_true",
        help="don't quantize model"
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
    parser.add_argument(
        "--l",
        type=int,
        default=0,
        help="number of loops"
    )

    args = parser.parse_args()
    return args

def load_model(hf_name,hf_token=HF_TOKEN,dq=False):
    '''Load the desired hf model/tokenizer and return'''
    login(hf_token)
    print('-'*25+'Loading Model: ' + hf_name + ' '+ '-'*25)
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForCausalLM.from_pretrained(hf_name,torch_dtype=torch.float16)
    if not dq:
        model.half()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Loading device to cuda.... ")
    model.to(DEVICE)
    print("Done loading to cuda!")
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

def infer_sequential(model, tokenizer, prompt: Sequence[str], left=False) -> typing.Tuple[Sequence[str], list]:
    torch.cuda.empty_cache()

    results = []
    timing_data = []

    for i, p in enumerate(tqdm(prompt)):
        # Time tokenization (cpu)
        t_start_tok = time.time()
        if left:
            model_inputs = tokenizer(p, return_tensors="pt", padding_side='left',
                                     padding=True, truncation=True)
        else:
            model_inputs = tokenizer(p, return_tensors="pt",
                                     padding=True, truncation=True)
        t_end_tok = time.time()
        tokenization_time = t_end_tok - t_start_tok
        print(f'Tokenization Time: {tokenization_time}\n')
        input_len = model_inputs['input_ids'].shape[1]

        token_movement_start = time.time()
        model_inputs.to(DEVICE)
        torch.cuda.synchronize()
        token_movement_end = time.time()
        token_movement_time = token_movement_end - token_movement_start
        print(f"token movement timing: {token_movement_time}")

        # Mimic KV Timing.
        start_kv, end_kv = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_kv.record()
        with torch.no_grad():
            _ = model(**model_inputs)
        end_kv.record()
        torch.cuda.synchronize()
        kv_time = start_kv.elapsed_time(end_kv) / 1000.0  # ms to s
        print(f'KV Time: {kv_time}\n')

        # Time generation
        start_gen, end_gen = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_gen.record()
        with torch.no_grad():
            generated_ids = model.generate(**model_inputs,
                                           max_new_tokens=MAX_TOKENS,
                                           do_sample=False,
                                           num_return_sequences=1,
                                           eos_token_id=tokenizer.eos_token_id,
                                           early_stopping=True)
        end_gen.record()
        torch.cuda.synchronize()
        gen_time = start_gen.elapsed_time(end_gen) / 1000.0  # ms to s

        total_time = kv_time + gen_time
        print(f"Generation time: {gen_time}\n")


        timing_data.append({
            'index': i,
            'question_length': input_len,
            'tokenization_time': tokenization_time,
            'kv_time': kv_time,
            'generation_time': gen_time,
            'total_time': total_time,
        })

    return timing_data

def manual_decode_sequential(model,
                             model_name, 
                             tokenizer,
                             prompts: Sequence[str],
                             use_kv_cache: bool = True,
                             left: bool = False,
                             max_new_tokens: int = MAX_TOKENS
                             ) -> typing.Tuple[Sequence[str], list]:

    results = []
    timing_data = []

    for i, prompt in enumerate(tqdm(prompts)):
        torch.cuda.empty_cache()

        # Time tokenization
        t0_tok = torch.cuda.Event(enable_timing=True)
        t1_tok = torch.cuda.Event(enable_timing=True)
        t0_tok.record()

        if left:
            if use_kv_cache:
                model_inputs = tokenizer(prompt, return_tensors="pt", padding_side='right',
                                     padding=True, truncation=True)
            else:
                model_inputs = tokenizer(prompt, return_tensors="pt", padding_side='left',
                                     padding=True, truncation=True)
        else:
            model_inputs = tokenizer(prompt, return_tensors="pt",
                                     padding=True, truncation=True)

        token_movement_start = time.time()
        tokenized = model_inputs.to(DEVICE)
        torch.cuda.synchronize()
        token_movement_end = time.time()
        token_movement_time = token_movement_end - token_movement_start
        print(f"token movement timing: {token_movement_time}")

        t1_tok.record()
        torch.cuda.synchronize()
        tokenization_time = t0_tok.elapsed_time(t1_tok) / 1000.0

        print(f"Tokenization time: {tokenization_time}\n")

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        input_len = input_ids.shape[-1]

        past_key_values = None
        generated = input_ids.clone().to(DEVICE)
        current_input = input_ids.to(DEVICE)
        current_mask = attention_mask.to(DEVICE)

        first_step_time = 0.0
        rest_gen_time = 0.0

        # Time generation
        with torch.no_grad():
            step = 0
            while step < max_new_tokens:
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()

                if step % 25 == 0:
                    print(f"single sequence step: {step}")

                if use_kv_cache:
                    out = model(
                        input_ids=current_input,
                        attention_mask=current_mask,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    past_key_values = out.past_key_values
                else:
                    out = model(
                        input_ids=generated,
                        attention_mask=(generated != tokenizer.pad_token_id),
                        use_cache=False
                    )

                end_ev.record()
                torch.cuda.synchronize()
                elapsed = start_ev.elapsed_time(end_ev) / 1000.0

                if step == 0:
                    first_step_time = elapsed
                else:
                    rest_gen_time += elapsed

                next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True).to(DEVICE)
                generated = torch.cat([generated, next_token], dim=-1)

                # (cdz) Added early stopping at Q: generation. 
                decoded_text = tokenizer.decode(generated[0][input_len:], skip_special_tokens=True)
                if "Q:" in decoded_text:
                    break
                if ANSWER_PHRASE in decoded_text:
                    step = max_new_tokens - 10
                
                if use_kv_cache:
                    current_input = next_token
                    current_mask = torch.cat([current_mask, torch.ones((current_mask.size(0), 1), dtype=current_mask.dtype, device=current_mask.device)], dim=-1)
                else:
                    current_input = generated

                if next_token.item() == tokenizer.eos_token_id:
                    break

                step += 1

        total_gen_time = first_step_time + rest_gen_time
        print(f"Generation time: {total_gen_time}\n")
        total_time = total_gen_time

        timing_data.append({"index": i, "question_length": input_len,
                            "tokenization_time": tokenization_time,
                            "kv_time": first_step_time,
                            "rest_gen_time": rest_gen_time,
                            "generation_time": total_gen_time,
                            "total_time": total_time,})
        
        if i % 10 == 0:
            result = tokenizer.decode(generated[0][input_len:], skip_special_tokens=True)
            print('-'*25 + 'EXAMPLE OUTPUT' + '-'*25)
            print(result)
            print('-'*25 + 'END EXAMPLE OUTPUT' + '-'*25)

            ckpts_dir = os.path.join(EVALUATION_DIR, "ckpts")
            os.makedirs(ckpts_dir, exist_ok=True)
            csv_path = os.path.join(ckpts_dir, f"{model_name}_ckpt_{i}.csv")

            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, 
                                        fieldnames=['index', 'question_length', 'tokenization_time',
                                                    'kv_time', 'rest_gen_time', 'generation_time', 'total_time'])
                writer.writeheader()
                for row in timing_data:
                    writer.writerow(row)

    return timing_data


def remove_prompt(prompts, model_inputs, tokenizer, generated_ids):
    results = []
    for i, prompt in enumerate(prompts):
        input_length = model_inputs['input_ids'][i].shape[0]
        result = tokenizer.decode(generated_ids[i][input_length:])
        results.append(result)
    return results


def main(argvs,i):
    global SAMPLES
    DEBUGGING = argvs.debug
    SAMPLES = argvs.samples
    isDecoder = True if any(x in argvs.model.split('/')[1].lower() for x in
                             ('llama','gemma')) else False

    model, tokenizer = load_model(argvs.model, dq=argvs.dq)
    
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


    # some models are decoder only, so we need to set left-padding for tokenizer
    # or we get issues with repetion of padding tokens!
    # inference timing
    if isDecoder:
        # timings = infer_sequential(model, tokenizer, [t[0] for t in tests], left=True)
        timings = manual_decode_sequential(model, save_modname, tokenizer, [t[0] for t in tests], use_kv_cache=False, left=True)
    else:
        # timings = infer_sequential(model, tokenizer, [t[0] for t in tests])
        timings = manual_decode_sequential(model, save_modname, tokenizer, [t[0] for t in tests], use_kv_cache=False, left=True)

        

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
    loop = argvs.l
    if loop == 0:
        main(argvs,argvs.t)
    else:
        i = 0
        while i < loop:
            print(i)
            main(argvs,i)
            i += 1
