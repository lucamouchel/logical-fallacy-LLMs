from calendar import c
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd 
from data_loader import *

generated_train, generated_dev, generated_test = load_generated_data('cckg')
cckg_train, cckg_dev, cckg_test = load_data('cckg')

print(generated_train)

generated_train['label'] = 1
generated_dev['label'] = 1
generated_test['label'] = 1

cckg_train['label'] = 0
cckg_dev['label'] = 0
cckg_test['label'] = 0

train_df = pd.concat([generated_train, cckg_train]).sample(frac=1)
dev_df = pd.concat([generated_dev, cckg_dev]).sample(frac=1)
test_df = pd.concat([generated_test, cckg_test]).sample(frac=1)


