import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
from utils import get_local_dir, get_local_run_dir, disable_dropout
import os
import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf, DictConfig
from typing import Optional, Set
from utils import *

OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs, mkdir=False))

latest_dpo_ckpt = '.cache/root/t5_cckg_dpo_2023-12-09_11-53-56_660827/LATEST/policy.pt'
latest_reference_ckpt = '.cache/root/t5_cckg_sft_2023-12-09_11-40-16_449255/LATEST/policy.pt'

def generate(prompt: str, config: DictConfig, policy: nn.Module, reference_model: Optional[nn.Module] = None, tokenizer: Optional[transformers.PreTrainedTokenizer] = None):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
    
    policy.eval()
    tokenized_prompt = tokenizer(prompt, return_tensors='pt', max_length=200, truncation=True)

    policy_output = policy.generate(
        tokenized_prompt['input_ids'],
        attention_mask=tokenized_prompt['attention_mask'], 
        min_length=10, 
        max_length=150,
        no_repeat_ngram_size=2 , 
        temperature=0.1, 
        do_sample=True, 
        pad_token_id=tokenizer.pad_token_id)
    
    policy_output = pad_to_length(policy_output, config.max_length, tokenizer.pad_token_id)
    policy_output_decoded = tokenizer.decode(policy_output[0], skip_special_tokens=True)
    if reference_model:
        reference_model.eval()
        reference_output = reference_model.generate(
            tokenized_prompt['input_ids'],
            attention_mask=tokenized_prompt['attention_mask'], 
            min_length=10, 
            max_length=150,
            no_repeat_ngram_size=2 , 
            temperature=0.1, 
            do_sample=True, 
            pad_token_id=tokenizer.pad_token_id)
        
        reference_output = pad_to_length(reference_output, config.max_length, tokenizer.pad_token_id)
        reference_output_decoded = tokenizer.decode(reference_output[0], skip_special_tokens=True)
        return (policy_output_decoded, reference_output_decoded)
    
    return policy_output_decoded
@hydra.main(version_base=None, config_path="config", config_name="config_evaluate")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""

    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)
    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")
    
    os.environ['XDG_CACHE_HOME'] = get_local_dir(config.local_dirs)
    policy_dtype = getattr(torch, config.model.policy_dtype)
    if 't5' in config.model.name_or_path.lower():
        policy = transformers.T5ForConditionalGeneration.from_pretrained(
            config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True, torch_dtype=policy_dtype)
    else:
        policy = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True, torch_dtype=policy_dtype)
    disable_dropout(policy)

    reference_ckpt = config.reference_ckpt or latest_reference_ckpt
    if reference_ckpt:
        reference_model_dtype = getattr(torch, config.model.reference_dtype)
        if 't5' in config.model.name_or_path.lower():
            reference_model = transformers.T5ForConditionalGeneration.from_pretrained(
                config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True, torch_dtype=reference_model_dtype)
        else:
            reference_model = transformers.AutoModelForCausalLM.from_pretrained(
                config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True, torch_dtype=reference_model_dtype)
        disable_dropout(reference_model)
    else:
        reference_model = None

    dpo_ckpt = config.dpo_ckpt or latest_dpo_ckpt
    if dpo_ckpt is not None:
        state_dict = torch.load(dpo_ckpt, map_location='cpu')
        policy.load_state_dict(state_dict['state'])
    else:
        raise ValueError("No checkpoint provided")
    
    if reference_ckpt is not None:
        state_dict = torch.load(reference_ckpt, map_location='cpu')
        reference_model.load_state_dict(state_dict['state'])
    
    tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    prefix_counter = "Generate a counter argument for the topic: "
    prefix_support = "Generate a supporting argument for the topic: "
    outs = generate(prompt=prefix_support + "Prostitution should be legal.", config=config, policy=policy, tokenizer=tokenizer)
    print(outs)
if __name__ == '__main__':
    main()