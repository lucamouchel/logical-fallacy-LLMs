import argparse
import os
import csv
import json
import re
import token
import tokenize
import sys
import evaluate

sys.path.append(".")
from T5Generation import T5LMClassifier
from src.T5.data_processing.utils import read_labels, get_encoded_code_tokens

import wandb

DATA_FOLDER = 'data'

metric = evaluate.load('rouge')

def training(training_file, dev_file,
             trained_models_dir,
             per_gpu_train_batch_size,
             learning_rate,
             epochs,
             language_model,
             grad_acc,
             sequence_length,
             optimizer_algorithm='adam',
             noisy_file=None,):

    if not os.path.exists(trained_models_dir):
        os.mkdir(trained_models_dir)
    classifier = T5LMClassifier(
            max_seq_length=sequence_length,
            output_model_dir=trained_models_dir,
            cache_dir=os.path.join(DATA_FOLDER, 'pretrained'),
        pretrained_model_name_or_path=language_model
    )
    classifier.train(training_file, dev_file,
                         per_gpu_train_batch_size=per_gpu_train_batch_size,
                         learning_rate=learning_rate,
                         optimizer_algorithm=optimizer_algorithm,
                         num_train_epochs=epochs,
                     noisy_file=noisy_file,
                     gradient_accumulation_steps=grad_acc)



def evaluate(test_file, trained_models_dir, sequence_length,
             per_gpu_eval_batch_size, language_model):
    _classifier = T5LMClassifier(max_seq_length=sequence_length,
                                 output_model_dir=trained_models_dir, 
                                 cache_dir=os.path.join(DATA_FOLDER, 'pretrained'),
                                 pretrained_model_name_or_path=language_model
                                 )

    preds = _classifier.predict(test_file=test_file,
                                per_gpu_eval_batch_size=per_gpu_eval_batch_size,
                                max_generated_tokens=sequence_length)
    
    labels = read_labels(test_file, tag='argument')
    inputs = read_labels(test_file, tag='topic')
    
    labels = [l.lower() for l in labels]
    preds = [p.lower() for p in preds]
    inputs = [i for i in inputs]

    eval_results = metric.compute(predictions=preds, references=labels, use_stemmer=True)

    print(eval_results)
    return eval_results

def parse_args():
    parser = argparse.ArgumentParser(description='Finetune T5 model')

    parser.add_argument('--training-file', dest='training_file', required=False, help='Path to training file',
                        default=None)
    parser.add_argument('--noisy-file', dest='noisy_file', required=False, help='Path to noisy file',
                        default=None)
    parser.add_argument('--validation-file', dest='validation_file', required=False, help='Path to validation file')
    parser.add_argument('--language-model', default='t5-base', help='Can be either some huggingface model or a '
                                                                         'path to a model. If the path is in GCS we '
                                                                         'download it first.')
    parser.add_argument('--model-dir', dest='model_dir', required=True,
                        help='the folder/google bucket in which the model will be stored or loaded from.')
    parser.add_argument('--epochs', default=20,
                        help='number of epochs to train')
    parser.add_argument('--batch-size', default=4,
                        help='batch size')
    parser.add_argument('--val-batch-size', default=4,
                        help='validation batch size')
    parser.add_argument('--lr', default=0.0001,
                        help='learning rate')
    parser.add_argument('--seq_len', default=256,
                        help='sequence length')
    parser.add_argument('--gradient-accumulation', default=4)
    parser.add_argument('--local_rank', default=-1)
    args = parser.parse_args()
    
    return args


def main():
    args = parse_args()
    #   train
    language_model = args.language_model
    if args.training_file and args.validation_file:
        training(training_file=args.training_file, 
                 dev_file=args.validation_file,
                 trained_models_dir=args.model_dir,
                 per_gpu_train_batch_size=int(args.batch_size),
                 epochs=int(args.epochs),
                 learning_rate=float(args.lr),
                 sequence_length=args.seq_len,
                 noisy_file=args.noisy_file,
                 language_model=language_model,
                 grad_acc=int(args.gradient_accumulation))
    
    if args.validation_file:
            evaluation_results = evaluate(test_file=args.validation_file,
                                      trained_models_dir=args.model_dir,
                                      per_gpu_eval_batch_size=int(args.val_batch_size),
                                      sequence_length=args.seq_len,
                                          language_model=language_model
                                    )
if __name__ == '__main__':
    main()
