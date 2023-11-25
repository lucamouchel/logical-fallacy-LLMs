from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch

class Eval:
    def __init__(self, model, tokenizer, label_encoder=None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        
    def eval(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        inputs.to(device='cuda')
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probas = torch.nn.functional.softmax(logits, dim=1)
        return self.label_encoder.inverse_transform([torch.argmax(probas, dim=1).item()])[0]
     
    def eval_data(self, dataset):
        y_test = dataset['fallacy type']
        y_test = self.label_encoder.inverse_transform(dataset['fallacy type'])
        y_pred = []
        for i, text in enumerate(dataset['argument']):
            y_pred.append(self.eval(text)) 
        return y_test, y_pred    
    
    def plot_heatmap(self, y_test, y_pred, title="Classification heatmap"):
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()

        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.iloc[:-1, :3], annot=True, cmap="Blues", fmt=".3f", cbar=True, linewidths=.5)
        plt.title(title)
        plt.show()
    
    
    