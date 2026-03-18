import os, sys
sys.path.append('../../src')
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from vae_model import MolecularVAE
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import selfies as sf

SPLIT = "train" # train (1.3M), test (118k), or validation (120k)
VAE_CHECKPOINT = "../../train_vae/checkpoints/molgen_vae_epoch_2"
PROTEIN_MODEL = "facebook/esm2_t33_650M_UR50D"       
HF_DATASET = 'alessandronascimento/clustered_jglaser_binding_affinity'
OUTPUT_DIR = f"latents_processed_{SPLIT}"
BATCH_SIZE = 32

# Optimized the mapping function
def smiles_to_selfies(example):
    try:
        # Just encode once directly into the dict
        example['selfies'] = sf.encoder(example['smiles_can'])
    except Exception:
        example['selfies'] = ''
    return example

print("Loading Models...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae = MolecularVAE(model_name="zjunlp/MolGen-large", latent_dim=256)
state_dict = torch.load(f"{VAE_CHECKPOINT}/pytorch_model.bin")
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
vae.load_state_dict(state_dict)
vae.to(device)
vae.eval()

vae_tokenizer = AutoTokenizer.from_pretrained(VAE_CHECKPOINT)

print(f"Loading {PROTEIN_MODEL}...")
esm_tokenizer = AutoTokenizer.from_pretrained(PROTEIN_MODEL)
esm_model = AutoModel.from_pretrained(PROTEIN_MODEL)
esm_model.to(device)
esm_model.eval()

def process_batch(batch):
    with torch.no_grad():
        prot_inputs = esm_tokenizer(
            batch['seq'], 
            padding=True, 
            truncation=True, 
            max_length=1024, 
            return_tensors="pt"
        ).to(device)
        
        prot_outputs = esm_model(**prot_inputs)
        attention_mask = prot_inputs['attention_mask'].unsqueeze(-1)
        prot_embeddings = (prot_outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
        
        lig_inputs = vae_tokenizer(
            batch['selfies'], 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        ).to(device)
        
        encoder_outputs = vae.bart.model.encoder(
            input_ids=lig_inputs['input_ids'],
            attention_mask=lig_inputs['attention_mask']
        )
        pooled_output = encoder_outputs.last_hidden_state[:, 0, :] 
        mu = vae.fc_mu(pooled_output) 

        affinities = batch['neg_log10_affinity_M']
        if not isinstance(affinities, torch.Tensor):
            affinities = torch.tensor(affinities, dtype=torch.float32)
        
    return prot_embeddings.cpu().half(), mu.cpu().half(), affinities.cpu().half()

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print('Loading dataset.....')
    dataset = load_dataset(HF_DATASET, split=SPLIT) 
    
    print('Mapping SMILES to SELFIES...')
    dataset = dataset.map(smiles_to_selfies, num_proc=8)
    
    print('Filtering out empty SELFIES...')
    dataset = dataset.filter(lambda example: example['selfies'] != '')
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    all_prot_embs = []
    all_ligand_mus = []
    all_affinities = []
    
    print(f"Processing {len(dataset)} pairs...")
    
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        prots, ligs, affinities = process_batch(batch)
        all_prot_embs.append(prots)
        all_ligand_mus.append(ligs)
        all_affinities.append(affinities)
        
        # Save combined dictionary every 1000 batches (32,000 pairs)
        if (i + 1) % 1000 == 0:
             chunk_data = {
                 'protein_embeddings': torch.cat(all_prot_embs),
                 'ligand_latents': torch.cat(all_ligand_mus),
                 'affinities' : torch.cat(all_affinities)
             }
             torch.save(chunk_data, f"{OUTPUT_DIR}/latent_chunk_{i}.pt")
             all_prot_embs = []
             all_ligand_mus = []
             all_affinities = []

    # Final Save for remaining pairs
    if all_prot_embs:
        chunk_data = {
             'protein_embeddings': torch.cat(all_prot_embs),
             'ligand_latents': torch.cat(all_ligand_mus),
             'affinities' : torch.cat(all_affinities)
        }
        torch.save(chunk_data, f"{OUTPUT_DIR}/latent_chunk_final.pt")

    print("Done! Data frozen.")