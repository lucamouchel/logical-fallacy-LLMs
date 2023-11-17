import os
import pandas as pd
def load_data(data_source: str):
    data_dir = 'data/argumentation'
    train_data = pd.read_csv(os.path.join(data_dir, f"train_{data_source}.tsv"), sep='\t')[['topic', 'argument', 'label']]
    dev_data = pd.read_csv(os.path.join(data_dir, f"dev_{data_source}.tsv"), sep='\t')[['topic', 'argument', 'label']]
    test_data = pd.read_csv(os.path.join(data_dir, f"test_{data_source}.tsv"), sep='\t')[['topic', 'argument', 'label']]
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
        
def load_augmented_data(data_source: str):
    """
    This function will read the original raw data from data source and will load the generated data with chatpgt and will return the augmented dataset

    Args:
        data_source (str): the data we will use. either cckg or iam
    """
    train_data, dev_data, test_data = load_data(data_source)
    train_data['fallacy type'] = 'no fallacy'
    dev_data['fallacy type'] = 'no fallacy'
    test_data['fallacy type'] = 'no fallacy'
    
    generated_data_dir = 'data/generated'
    generated_train_data = process_generated_data(pd.read_json(os.path.join(generated_data_dir, f"train_{data_source}.json")))
    generated_dev_data = process_generated_data(pd.read_json(os.path.join(generated_data_dir, f"dev_{data_source}.json")))
    generated_test_data = process_generated_data(pd.read_json(os.path.join(generated_data_dir, f"test_{data_source}.json")))
    
    train_data = pd.concat([train_data, generated_train_data]).sample(frac=1)
    dev_data = pd.concat([dev_data, generated_dev_data]).sample(frac=1)
    test_data = pd.concat([test_data, generated_test_data]).sample(frac=1)
    return train_data, dev_data, test_data
        
        