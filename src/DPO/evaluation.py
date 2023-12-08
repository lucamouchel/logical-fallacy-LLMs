from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration, T5Tokenizer
import torch
from utils import disable_dropout



ckpt = '.cache/root/pythia1_cckg_dpo_2023-12-08_15-17-26_437706/LATEST/policy.pt'

state_dict = torch.load(ckpt, map_location='cpu')

model_kwargs = {'device_map': 'balanced'} 

policy = AutoModelForCausalLM.from_pretrained(
        'EleutherAI/pythia-1b', cache_dir='.cache/root/', **model_kwargs)

