import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import typing

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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

COT_QS = (q1, q2, q3, q4, q5, q6, q7, q8)
COT_PROMPT = preamble+'\n'+q1+'\n'+q2+'\n'+q3+'\n'+q4+'\n'+q5+'\n'+q6+'\n'+q7+'\n'+q8+'\n'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE_DIR, "evaluation/models")

GEMMA_DIR = os.path.join(MODELS, "gemma-2-2b-it")
MATHSTRAL_DIR = os.path.join(MODELS, "Mathstral-7B-v0.1")
RHO_MATH_DIR = os.path.join(MODELS, "rho-math-1b-v0.1")

def save_prompt_kv(path: str,
                   past_key_values: typing.Tuple[typing.Tuple[torch.Tensor,torch.Tensor],...],
                   prompt_mask: torch.Tensor,
                   prompt_len: int):
    """
    Move all tensors to CPU and save the cache dict to `path`.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # detach + cpu
    pkv_cpu = tuple(
        (layer_k.cpu(), layer_v.cpu())
        for (layer_k, layer_v) in past_key_values
    )
    data = {
        "past_key_values": pkv_cpu,
        "prompt_mask": prompt_mask.cpu(),
        "prompt_len": prompt_len
    }
    torch.save(data, path)
    print(f"Saved prompt KV cache to {path}")

def precompute_prompt_kv(model, tokenizer, prompt: str):
    """
    Tokenize the shared prompt and run through the model once to obtain
    `past_key_values` for reuse across multiple generations.
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=True
        )
    # Return both the past_key_values and initial mask/length
    return out.past_key_values, inputs["attention_mask"].clone(), inputs["input_ids"].shape[-1]


def main():
    model_dirs = [GEMMA_DIR, MATHSTRAL_DIR, RHO_MATH_DIR]

    for i in model_dirs:
        tokenizer = AutoTokenizer.from_pretrained(i)
        model = AutoModelForCausalLM.from_pretrained(i,torch_dtype=torch.float16)
        model.to(DEVICE)
        kv, prompt_mask, prompt_len = precompute_prompt_kv(model, tokenizer, COT_PROMPT)
        save_prompt_kv(os.path.join(i,'cached_prompt_kv.pt'), kv, prompt_mask, prompt_len)

