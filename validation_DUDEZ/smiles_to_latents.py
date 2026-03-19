import torch
from transformers import AutoTokenizer
import selfies as sf
from rdkit import Chem
from tqdm import tqdm
import sys
sys.path.append('../src')
from vae_model import MolecularVAE
from transformers.modeling_outputs import BaseModelOutput
from utils import encode_to_latent, decode_from_latent


def smiles_to_selfies(smiles_list):
    selfies_list = []
    for smiles in smiles_list:
        try:
            selfies_list.append(sf.encoder(smiles))
        except
            pass
    return selfies_list


if __name__ == '__main__':
    
    targets = ['ABL1', 'HDAC8', 'LCK', 'PTN1', 'SRC']
    