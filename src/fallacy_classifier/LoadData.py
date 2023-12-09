from cProfile import label
import pandas as pd
from transformers import InputExample, T5Tokenizer
from torch.utils.data import TensorDataset
import torch
import os

def load_data(data_source: str):
    data_dir = f'data/argumentation/'
    train_data = pd.read_json(os.path.join(data_dir, f"train_{data_source}.json"))
    dev_data = pd.read_json(os.path.join(data_dir, f"dev_{data_source}.json"))
    test_data = pd.read_json(os.path.join(data_dir, f"test_{data_source}.json"))
    return train_data, dev_data, test_data

def process_generated_data(generated_data):
    generated = []
    for i, row in generated_data.iterrows():
        data = row[~row.isna()]
        assert data.shape[0] == 3, "wrong shape"

        topic = data['topic']
        fallacy_type = data['fallacy type']
        stance = data.keys()[-1].split(" ")[-1]
        assert stance=='support' or stance=='counter', "invalid stance"

        argument = data.values[2]
        arg = {
            'topic': topic,
            'argument': argument,
            'label' : 1 if stance == 'support' else -1,
            'fallacy type': fallacy_type
        }
        generated.append(arg)
        
    return pd.DataFrame(generated)
def load_generated_data(data_source: str):
    generated_data_dir = f'data/generated'
    generated_train_data = process_generated_data(pd.read_json(os.path.join(generated_data_dir, f"train_{data_source}.json")))
    generated_dev_data = process_generated_data(pd.read_json(os.path.join(generated_data_dir, f"dev_{data_source}.json")))
    generated_test_data = process_generated_data(pd.read_json(os.path.join(generated_data_dir, f"test_{data_source}.json")))
    return generated_train_data, generated_dev_data, generated_test_data


class DatasetLoader():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.generated_train, self.generated_dev, self.generated_test = load_generated_data('cckg')
        self.cckg_train, self.cckg_dev, self.cckg_test = load_data('cckg')
    
    def load_data(self):
        self.generated_train['label'] = 1
        self.generated_dev['label'] = 1
        self.generated_test['label'] = 1

        self.cckg_train['label'] = 0
        self.cckg_dev['label'] = 0
        self.cckg_test['label'] = 0

        train_df = pd.concat([self.generated_train, self.cckg_train]).sample(frac=1, random_state=42)
        dev_df = pd.concat([self.generated_dev, self.cckg_dev]).sample(frac=1, random_state=42)
        test_df = pd.concat([self.generated_test, self.cckg_test]).sample(frac=1, random_state=42)
        return train_df, dev_df, test_df

    def load_dataset(self, split="train"):
        train, dev, test = self.load_data()
        if split == 'train':
            df = train
        elif split == 'dev':
            df = dev
        elif split == 'test': 
            df = test
        else: 
            raise ValueError("split should be in [train, dev, test]")    
        
        examples = []
        labels = []
        from tqdm import tqdm
        for i, entry in tqdm(df.iterrows()):
            argument = entry.argument
            label = entry.label
            guid = str(i)
            ex = InputExample(guid=guid, text_a=argument, text_b=None, label=label)
            examples.append(ex)
            labels.append(label)
        tokenized_inputs = self.tokenizer.batch_encode_plus(
            [ex.text_a for ex in examples],
            padding="longest",
            max_length=150,
            pad_to_max_length = True,
            truncation=True,
            return_tensors="pt",
        )
        l = torch.tensor([ex.label for ex in examples])
        tokenized_targets = {'input_ids': l,
                             'attention_mask' : torch.ones_like(l)}
        
        dataset = TensorDataset(tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'],
                                tokenized_targets['input_ids'], tokenized_targets['attention_mask'] 
                            )
        return dataset, labels
    
