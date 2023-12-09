import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
from utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, get_open_port
import os
import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf, DictConfig
import trainers
import wandb
import json
import socket
from typing import Optional, Set
import resource
from utils import *

OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))


def worker_main(rank: int, world_size: int, config: DictConfig, policy: nn.Module, reference_model: Optional[nn.Module] = None):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
    
  
    tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    prompt = "Generate a counter argument for the topic: Mathematics is important."
    tokenized_prompt = tokenizer(prompt, return_tensors='pt', max_length=200, truncation=True)
    # FSDP generation according to
    print(tokenized_prompt['input_ids'])
    policy_output = reference_model.generate(
        tokenized_prompt['input_ids'], attention_mask=tokenized_prompt['attention_mask'], max_length=150, do_sample=True, pad_token_id=tokenizer.pad_token_id)
    
    policy_output = pad_to_length(policy_output, config.max_length, tokenizer.pad_token_id)
    policy_output = all_gather_if_needed(policy_output, rank, world_size)
    policy_output_decoded = tokenizer.batch_decode(policy_output, skip_special_tokens=True)

    print("policy=",policy_output_decoded)

@hydra.main(version_base=None, config_path="config", config_name="config_evaluate")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""

    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    config_path = os.path.join(config.local_run_dir, 'config_evaluate.yaml')
    
    os.environ['XDG_CACHE_HOME'] = get_local_dir(config.local_dirs)
    print('building policy')
    policy_dtype = getattr(torch, config.model.policy_dtype)
    if 't5' in config.model.name_or_path.lower():
        policy = transformers.T5ForConditionalGeneration.from_pretrained(
            config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True, torch_dtype=policy_dtype)
    else:
        policy = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True, torch_dtype=policy_dtype)
    disable_dropout(policy)

    if config.reference_ckpt:
        print('building reference model')
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

    if config.dpo_ckpt is not None:
        state_dict = torch.load(config.dpo_ckpt, map_location='cpu')
        policy.load_state_dict(state_dict['state'])
    else:
        raise ValueError("No checkpoint provided")
    
    if config.reference_ckpt is not None:
        state_dict = torch.load(config.reference_ckpt, map_location='cpu')
        reference_model.load_state_dict(state_dict['state'])
    
    print('loaded pre-trained weights')
    worker_main(0, 1, config, policy, reference_model)

if __name__ == '__main__':
    main()