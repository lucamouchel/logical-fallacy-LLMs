from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration, T5Tokenizer
import torch
from utils import *



ckpt = '.cache/root/pythia1_cckg_dpo_2023-12-08_15-17-26_437706/LATEST/policy.pt'

state_dict = torch.load(ckpt, map_location='cpu')
policy = AutoModelForCausalLM.from_pretrained(
        'EleutherAI/pythia-1b', cache_dir='.cache/root/', torch_dtype=torch.float32)

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-1b', cache_dir='.cache/root/')
if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

prompt="Generate a counter argument for the topic: The earth is flat."
tokenized_prompt = tokenizer(prompt, return_tensors='pt', max_length=200, truncation=True)
        # FSDP generation according to

        
policy_output = policy.generate(
tokenized_prompt['input_ids'], attention_mask=tokenized_prompt['attention_mask'], max_length=150, do_sample=True, pad_token_id=tokenizer.pad_token_id)

policy_output = pad_to_length(policy_output, 120, tokenizer.pad_token_id)
policy_output = all_gather_if_needed(policy_output, rank=0, world_size=1)
policy_output_decoded = tokenizer.batch_decode(policy_output, skip_special_tokens=True)

print("policy=",policy_output_decoded)