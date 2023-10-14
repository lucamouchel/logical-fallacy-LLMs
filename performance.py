import pandas as pd

class PerformanceSaver:
    def __init__(self, csv_file='performance.csv', MSKD='False'):
        self.csv_file = csv_file
        self.MSKD = MSKD

    def save_performance(self, model_name, results):
        accuracy = results['test_accuracy']
        f1 = results['test_f1']
        precision = results['test_precision']
        recall = results['test_recall']
        perf = pd.read_csv(self.csv_file)
        
        if model_name not in perf['model_name'].values:
            perf.loc[len(perf)] = {'model_name': model_name, 'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall, 'MSKD': self.MSKD}
        
        perf.to_csv(self.csv_file, index=False)
