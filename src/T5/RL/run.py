import argparse
from itertools import chain
import json
import logging
import numpy as np
import os
import random
from tqdm import tqdm
import torch
import transformers
import accelerate
from torch.utils.data import IterableDataset, Dataset, DataLoader
from T5RLTrainer import Trainer
FORMAT_BY_TASK = {'cckg': 'bool'}
PART_BY_TASK = {'cckg': 'unseen'}

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def set_seed(seed=19260817, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available() and cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def reduce_sum(value, mask, axis=None):
    if axis is None:
        return torch.sum(value * mask)
    return torch.sum(value * mask, axis)

def reduce_mean(value, mask, axis=None):
    if axis is None:
        return torch.sum(value * mask) / torch.sum(mask)
    return reduce_sum(value, mask, axis) / torch.sum(mask, axis)


accelerator = accelerate.Accelerator()
device = accelerator.device
logging.basicConfig(level=logging.INFO)
log = accelerate.logging.get_logger(__name__, log_level='INFO')
def log_info(s):
    if accelerator.is_main_process:
        log.info(s)


class DeclarativeDataset(Dataset):
    def __init__(self, split, tasks, tokenizer):
        self.split = split
        self.tasks = tasks.split(',')
        self.tokenizer = tokenizer

        self.instances = self.load_datasets()

        if split == 'train':
            random.seed(19260817)
            random.shuffle(self.instances)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]

    def load_datasets(self):
        instances = []
        for task_ix, task in enumerate(self.tasks):
            path = os.path.join('data', task, f'{self.split}.json')
            with open(path) as f:
                js = json.load(f)
            if self.split == 'dev' and len(js) > 2000:
                random.seed(19260817)
                js = random.sample(js, 2000)
            for item in js:
                assert len(item['golds']) <= 1
                distractors = item['distractors'][:3] if self.split == 'train' else item['distractors']
                instance = {
                    'task': task,
                    'split': self.split,
                    'task_ix': task_ix,
                    'sources': item['golds'] + distractors,
                    'first_is_correct': 1 if len(item['golds']) > 0 else 0,
                    'is_mc': 1 if FORMAT_BY_TASK[task] == 'mc' else 0,
                }
                instances.append(instance)
            log_info(f'Loaded dataset for task {task} split {self.split} with {len(js)} instances')
        log_info(f'{self.split} set size = {len(instances)}')
        return instances

    def collate_fn(self, batch):
        task_ixs = torch.tensor([item['task_ix'] for item in batch], dtype=torch.long)
        first_is_corrects = torch.tensor([item['first_is_correct'] for item in batch], dtype=torch.long)
        is_mcs = torch.tensor([item['is_mc'] for item in batch], dtype=torch.long)

        CAP = 4 if batch[0]['split'] == 'train' else 11
        sourcess = [item['sources'] + [''] * (CAP - len(item['sources'])) for item in batch]
        sourcess_tok = [self.tokenizer.batch_encode_plus(
            sources,
            return_tensors='pt', padding='max_length', truncation='longest_first', max_length=self.tokenizer.max_input_len)
            for sources in sourcess]
        sourcess_input_ids = torch.stack([sources_tok.input_ids for sources_tok in sourcess_tok], dim=0)
        sourcess_attention_mask = torch.stack([sources_tok.attention_mask for sources_tok in sourcess_tok], dim=0)

        result = {
            'task_ixs': task_ixs, # (B)
            'sourcess_input_ids': sourcess_input_ids, # (B, C, L)
            'sourcess_attention_mask': sourcess_attention_mask, # (B, C, L)
            'first_is_corrects': first_is_corrects, # (B)
            'is_mcs': is_mcs, # (B)
        }

        return result


def get_args():
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'my_task', 'test'], default='train', help='train or eval?')

    # common
    parser.add_argument('--model_type', type=str, default='google/t5-v1_1-small')
    parser.add_argument('--load_from_ckpt', default=None)
    parser.add_argument('--max_input_len', type=int, default=128)

    # train
    parser.add_argument('--train_tasks', type=str, default='openbookqa,arc_easy,arc_hard,ai2_science_elementary,ai2_science_middle,commonsenseqa,qasc,physical_iqa,social_iqa,winogrande_xl,com2sense_paired,sciq,quarel,quartz,cycic_mc,comve_a,csqa2,symkd_anno,gengen_anno')
    parser.add_argument('--valid_tasks', type=str, default='openbookqa,arc_easy,arc_hard,ai2_science_elementary,ai2_science_middle,commonsenseqa,qasc,physical_iqa,social_iqa,winogrande_xl,com2sense_paired,sciq,quarel,quartz,cycic_mc,comve_a,csqa2,symkd_anno,gengen_anno')
    parser.add_argument('--eval_tasks', type=str, default='openbookqa,arc_easy,arc_hard,ai2_science_elementary,ai2_science_middle,commonsenseqa,qasc,physical_iqa,social_iqa,winogrande_xl,com2sense_paired,sciq,quarel,quartz,cycic_mc,comve_a,csqa2,symkd_anno,gengen_anno,wsc273,copa,numersense,prost,spatial_cs,swag,hellaswag,codah,story_cloze_test,alphanli,strategyqa,creak,rainier_anno')
    parser.add_argument('--total_steps', type=int, default=25000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--loss_fn', type=str, default='mce+bce', choices=['margin', 'bce', 'mce', 'mce+bce'])
    parser.add_argument('--contrastive_loss_type', type=int, default=4)
    parser.add_argument('--contrastive_loss_temp', type=float, default=0.05)
    parser.add_argument('--contrastive_loss_coef', type=float, default=0.1)

    # other
    parser.add_argument('--log_interval', type=int, default=50, help='step interval to log stats')
    parser.add_argument('--save_interval', type=int, default=1000, help='step interval to save model checkpoints')
    parser.add_argument('--eval_interval', type=int, default=1000, help='step interval to do evaluation')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--nosave', default=False, action='store_true')
    parser.add_argument('--nolog', default=False, action='store_true')
    parser.add_argument('--eval_loop_cap', type=int, default=None)

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    set_seed()

    # Set up save directories
    if not args.nosave:
        if args.mode == 'train':
            args.output_dir = 'runs/'
            args.save_dir = os.path.join(args.output_dir, args.run_name)
            args.model_dir = os.path.join(args.save_dir, 'model')
            if accelerator.is_main_process:
                for d in [args.save_dir, args.model_dir]:
                    ensure_dir(d)

        elif args.mode == 'eval':
            if args.load_from_ckpt is not None:
                args.save_dir = os.path.dirname(os.path.dirname(args.load_from_ckpt))
                args.save_dir = args.save_dir.replace('runs', 'eval')
                ckp = args.load_from_ckpt.split('ckp_')[-1].strip('.pth')
                args.save_dir += f'_ckp-{ckp}'
            else:
                log.error('You must provide --load_from_ckpt!')
                exit(-1)
            args.run_name = args.save_dir.split('/')[-1]
            if accelerator.is_main_process:
                for d in [args.save_dir]:
                    ensure_dir(d)
        elif args.mode == 'my_task' or 'test':
            if args.load_from_ckpt is not None:
                args.save_dir = os.path.dirname(os.path.dirname(args.load_from_ckpt))
                args.save_dir = args.save_dir.replace('runs', 'eval')
                ckp = args.load_from_ckpt.split('ckp_')[-1].strip('.pth')
                args.save_dir += f'_ckp-{ckp}'
                print(args.save_dir)
            else:
                log.error('You must provide --load_from_ckpt!')
                exit(-1)
            args.run_name = args.save_dir.split('/')[-1]
            if accelerator.is_main_process:
                for d in [args.save_dir]:
                    ensure_dir(d)
        
        log_info(f'Write to output directory: {args.save_dir}')
        if accelerator.is_main_process:
            with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
                json.dump(args.__dict__, f, indent=2)

    # Load data
    log_info(f'Loading data ...')
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_type)
    tokenizer.max_input_len = args.max_input_len
   
    if args.mode == 'train':
        train_dataset = DeclarativeDataset('train', args.train_tasks, tokenizer)

        eval_dataset = DeclarativeDataset('dev', args.valid_tasks, tokenizer)
        # train ds is shuffled in its constructor
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=train_dataset.collate_fn)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=eval_dataset.collate_fn)
        train_dataloader, eval_dataloader = accelerator.prepare(train_dataloader, eval_dataloader)
    elif args.mode == 'eval':
        train_dataset = None
        eval_dataset = DeclarativeDataset('dev', args.eval_tasks, tokenizer)
        train_dataloader = None
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=eval_dataset.collate_fn)
        eval_dataloader = accelerator.prepare(eval_dataloader)
    elif args.mode == 'my_task' or 'test':
        train_dataset = None
        eval_dataset = DeclarativeDataset(args.mode, args.eval_tasks, tokenizer)
        train_dataloader = None
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=eval_dataset.collate_fn)
        eval_dataloader = accelerator.prepare(eval_dataloader)
        
        
    # Initialize models and optimizer
    log_info(f'Initializing models ...')
    model = transformers.T5EncoderModel.from_pretrained(args.model_type)
    model.D = model.shared.embedding_dim
    linear = torch.nn.Linear(model.D, 1)
    if args.mode == 'train':
        if args.load_from_ckpt is not None:
            checkpoint = torch.load(args.load_from_ckpt, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            linear.load_state_dict(checkpoint['linear'])
            checkpoint.clear()
        model, linear = accelerator.prepare(model, linear)
        optimizer = torch.optim.Adam(chain(model.parameters(), linear.parameters()), lr=args.lr)
        optimizer = accelerator.prepare(optimizer)
    elif args.mode == 'eval':
        if args.load_from_ckpt is not None:
            checkpoint = torch.load(args.load_from_ckpt, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            linear.load_state_dict(checkpoint['linear'])
            step = checkpoint['step']
            checkpoint.clear()
        model, linear = accelerator.prepare(model, linear)
        optimizer = None
    elif args.mode == 'my_task' or 'test':
        if args.load_from_ckpt is not None:
            checkpoint = torch.load(args.load_from_ckpt, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            linear.load_state_dict(checkpoint['linear'])
            step = checkpoint['step']
            checkpoint.clear()
        model, linear = accelerator.prepare(model, linear)
        optimizer = None

    # Set up trainer
    trainer = Trainer(
        args=args,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        tokenizer=tokenizer,
        model=model,
        linear=linear,
        optimizer=optimizer,
    )

    # Train
    if args.mode == 'train':
        steps = list(range(args.total_steps + 1))
        steps = tqdm(steps) if accelerator.is_main_process else steps
        for step in steps:
            trainer.train(step)
    elif args.mode == 'eval':
        trainer.eval(step)
    elif args.mode == 'my_task' or 'test':
        trainer.eval(step)

if __name__ == '__main__':
    main()