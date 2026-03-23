import torch
from tqdm import tqdm
import selfies as sf
from rdkit import Chem
from utils import decode_from_latent
from transformers import AutoTokenizer, AutoModel
import argparse
import sys
from diffuser_lightning import Prot2Chem_Diffusion
from vae_model import MolecularVAE
import pandas as pd
from utils.load_model import load_pretrained_models


def generate_target_embeddings(target_seq='', model_name='facebook/esm2_t33_650M_UR50D', device='cuda'):
    esm_tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    inputs = esm_tokenizer(target_seq, padding=True, truncation=True, max_length=1024, return_tensors='pt').to(device)
    with torch.no_grad():        
        prot_outputs = model(**inputs)
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        prot_embeddings = (prot_outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
    return prot_embeddings

def generate_molecules_batched(
    diffusion_model, vae_model, tokenizer, 
    protein_embeddings, target_affinities, 
    scale_factor, steps=100, guidance_scale=4.0, device='cuda'
):
    """
    protein_embeddings: Tensor of shape [Batch_Size, 1280]
    target_affinities: Tensor of shape [Batch_Size, 1]
    """
    diffusion_model.eval()
    vae_model.eval()
    diffusion_model.to(device)
    vae_model.to(device)
    
    batch_size = protein_embeddings.size(0)
    
    # Setup Conditions for CFG
    protein_cond = protein_embeddings.to(device)
    affinity_cond = target_affinities.to(device)
    
    uncond_protein = torch.zeros_like(protein_cond).to(device)
    uncond_affinity = torch.zeros_like(affinity_cond).to(device)

    latents = torch.randn((batch_size, 256), device=device)
    
    scheduler = diffusion_model.scheduler
    scheduler.set_timesteps(num_inference_steps=steps)
    
    print(f"Generating {batch_size} molecules... (Steps: {steps}, CFG: {guidance_scale})")
    
    # Denoising Loop
    with torch.no_grad():
        for t in tqdm(scheduler.timesteps, desc="Denoising"):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            noise_pred_cond = diffusion_model(latents, t_batch, protein_cond, affinity_cond)
            
            noise_pred_uncond = diffusion_model(latents, t_batch, uncond_protein, uncond_affinity)
            
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            
    true_latents = latents / scale_factor
    
    generated_smiles = []
    print("Decoding latents to molecules...")
    with torch.no_grad():
        generated_ids = decode_from_latent(vae_model, true_latents)
        reconstructed_selfies = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        for decoded_selfie in reconstructed_selfies:
            try:
                smiles = sf.decoder(decoded_selfie)
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    generated_smiles.append(smiles)
                else:
                    generated_smiles.append(f"INVALID_SMILES: {smiles}")
            except Exception:
                generated_smiles.append(f"INVALID_SELFIES: {decoded_selfie}")
                
    return generated_smiles

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    parser = argparse.ArgumentParser()
    parser.add_argument("--protein_seq", type=str, default="", help="Target protein sequence")
    parser.add_argument("--protein_model", type=str, default="facebook/esm2_t33_650M_UR50D", help="ESM-2 model name")
    parser.add_argument("--target_affinity", type=float, default=9.0, help="Desired binding affinity (e.g., pKd)")
    parser.add_argument("--steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="Classifier-Free Guidance scale")
    parser.add_argument("--batch_size", type=int, default=64, help="Number of molecules to generate in a batch")
    parser.add_argument('--output_prefix', type=str, default='generated_molecules', help='Prefix for output files')
    parser.add_argument('--seed', type=int, default=23, help='Random seed for reproducibility')
    args = parser.parse_args()

    batch_size = args.batch_size

    torch.manual_seed(args.seed)

    if args.protein_seq != "":
        protein_embeddings = generate_target_embeddings(args.protein_seq, args.protein_model)
        desired_affinities = torch.full((batch_size, 1), args.target_affinity)
    else: # Unconditional generation
        print('Running unconditional generation (no protein sequence provided)...')
        protein_embeddings = torch.zeros(1280)
        desired_affinities = torch.zeros(batch_size, 1)

    protein_embeddings = protein_embeddings.repeat(batch_size, 1)
    
    SCALE_FACTOR = 2.15709

    vae_model, diffusion_model = load_pretrained_models()

    tokenizer = AutoTokenizer.from_pretrained("zjunlp/MolGen-large")

    new_molecules = generate_molecules_batched(
        diffusion_model, vae_model, tokenizer, 
        protein_embeddings, desired_affinities, scale_factor=SCALE_FACTOR, 
        guidance_scale=args.guidance_scale, steps=args.steps, 
        device=device)
    
    df = pd.DataFrame(new_molecules, columns=['generated_smiles'])
    df.to_csv(f"{args.output_prefix}_batch_{batch_size}_guidance_{args.guidance_scale}.csv")