import moses
import pandas as pd
from glob import glob
import torch
from tqdm import tqdm
import random


def read_molecules(mol_file):
    mol_list = []
    with open(mol_file, 'r') as f:
        for line in f:
            mol_list.append(line.split()[0])
    return mol_list

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running on {device}')

    targets = ['ABL1', 'HDAC8', 'LCK', 'PTN1', 'SRC']

    unconditional_csv = glob('../../unconditional_generation/unconditional_*.csv')
    uncond_df = pd.concat([pd.read_csv(i) for i in unconditional_csv])
    uncond_list = uncond_df['generated_smiles'].tolist()
    random.shuffle(uncond_list)
    print(f'Found {len(uncond_df)} unconditionally generated molecules.')

    for target in tqdm(targets):
        csv_files = glob(f'{target}_*.csv')
        df = pd.concat([pd.read_csv(i) for i in csv_files])

        smiles_list = df['generated_smiles'].tolist()
        print(f'Found {len(smiles_list)} conditional molecules for target {target}.')

        actives = read_molecules(f'../../../validation_DUDEZ/{target}/ligands.smi')
        decoys = read_molecules(f'../../../validation_DUDEZ/{target}/decoys.smi')

        random.shuffle(decoys)
        decoys = decoys[:len(actives)]

        random.shuffle(smiles_list)
        smiles_list = smiles_list[:len(actives)]


        metrics_actives = moses.get_all_metrics(smiles_list, test=actives, n_jobs=2, device=device)
        metrics_actives = pd.DataFrame.from_dict(metrics_actives, orient='index')
        metrics_decoys = moses.get_all_metrics(smiles_list, test=decoys, n_jobs=2, device=device)
        metrics_decoys = pd.DataFrame.from_dict(metrics_decoys, orient='index')
        metrics_uncond = moses.get_all_metrics(smiles_list, test=uncond_list[:len(actives)], n_jobs=2, device=device)
        metrics_uncond = pd.DataFrame.from_dict(metrics_uncond, orient='index')
        metrics_moses = moses.get_all_metrics(smiles_list, n_jobs=2, device=device)
        metrics_moses = pd.DataFrame.from_dict(metrics_moses, orient='index')
        
        
        metrics_actives.to_csv(f'moses_metrics_{target}_actives.csv')
        metrics_decoys.to_csv(f'moses_metrics_{target}_decoys.csv')
        metrics_uncond.to_csv(f'moses_metrics_{target}_unconditional.csv')
        metrics_moses.to_csv(f'moses_metrics_{target}_moses.csv')
        print('MOSES metrics written to csv files.')
