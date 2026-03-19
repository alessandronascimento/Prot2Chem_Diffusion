import moses
import pandas as pd
from glob import glob
import torch

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running on {device}')

    csv_files = glob('unconditional_*.csv')
    df = pd.concat([pd.read_csv(i) for i in csv_files])

    smiles_list = df['generated_smiles'].tolist()
    metrics = moses.get_all_metrics(smiles_list, n_jobs=2, device=device)
    metrics = pd.DataFrame.from_dict(metrics, orient='index')
    metrics.to_csv('moses_metrics.csv')
    print('MOSES metrics written to moses_metrics.csv file.')
