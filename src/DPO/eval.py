import torch

torch.backends.cuda.matmul.allow_tf32 = True
from transformers import T5ForConditionalGeneration
import torch.nn as nn
import transformers
from utils import get_local_dir, get_local_run_dir, disable_dropout
import os
import pandas as pd
import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf, DictConfig
from typing import Optional, Set
from utils import *

OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs, mkdir=False))

latest_dpo_ckpt = '.cache/root/t5_cckg_dpo_2023-12-09_14-36-09_394900/LATEST/policy.pt'
latest_reference_ckpt = '.cache/root/t5_cckg_sft_2023-12-09_14-32-43_811574/LATEST/policy.pt'
dpo_model = None
ref_model = None
tkzer = None

@hydra.main(version_base=None, config_path="config", config_name="config_evaluate")
def load_models(config: DictConfig):
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
    
    global dpo_model
    global ref_model
    global tkzer
    dpo_model=policy
    ref_model=reference_model
    tkzer=tokenizer

    return 

load_models()

def generate(prompt: str, model: nn.Module, tokenizer: Optional[transformers.PreTrainedTokenizer] = tkzer):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
    model.eval()
    tokenized_prompt = tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True)
    output = model.generate(
        tokenized_prompt['input_ids'],
        attention_mask=tokenized_prompt['attention_mask'], 
        min_length=10, 
        max_length=150,
        no_repeat_ngram_size=2 , 
        temperature=0.1, 
        do_sample=True, 
        pad_token_id=tokenizer.pad_token_id)
    output = pad_to_length(output, 256, tokenizer.pad_token_id)
    output_decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return output_decoded

external_model = T5ForConditionalGeneration.from_pretrained('models/flan_t5_base_model', cache_dir='models/flan_t5_base_model')
# txt = generate("Generate a supporting argument for the topic: Cannabis should be legal.", dpo_model)
# txt2 = generate("Generate a supporting argument for the topic: Cannabis should be legal.", ref_model)
# txt3 = generate("Generate a supporting argument for the topic: Cannabis should be legal.", external_model)

def evaluate_over_dataset():
    import evaluate
    from tqdm import tqdm
    metric = evaluate.load('bleu')
    dataset = pd.read_json('data/argumentation/test_cckg.json')
    golden = []
    generated_dpo = []
    generated_external = [] ## refiner
    for i, entry in tqdm(dataset[:3].iterrows()):
        topic = entry.topic
        golden_arg = entry.argument
        stance = entry.label
        prompt = f"Generate a {'supporting' if stance==1 else 'counter'} argument for the topic: {topic}."
        dpo_generated = generate(prompt, dpo_model)
        generated_ext = generate(prompt, external_model)
        
        golden.append(golden_arg)
        generated_dpo.append(dpo_generated)
        generated_external.append(generated_ext)
        
    
    rouge_scores = metric.compute(predictions=generated_dpo, references=golden)
    rouge_scores_with_external = metric.compute(predictions=generated_external, references=golden,)

    print(rouge_scores) 
    print("************")
    print(rouge_scores_with_external)
evaluate_over_dataset()