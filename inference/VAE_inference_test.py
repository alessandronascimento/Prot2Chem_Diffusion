import torch
from transformers import AutoTokenizer
import selfies as sf
from rdkit import Chem
from tqdm import tqdm
import sys
sys.path.append('../src')
from vae_model import MolecularVAE
from datasets import load_dataset
from transformers.modeling_outputs import BaseModelOutput
from utils import encode_to_latent, decode_from_latent

def test_vae_validity_batched(vae_model, tokenizer, inputs, device='cuda'):
    
    z = encode_to_latent(vae_model, inputs, use_mean=True)
    generated_ids = decode_from_latent(vae_model, z)    
    reconstructed_selfies = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    valid_count = 0

    for decoded_selfie in reconstructed_selfies:
        try:
            smiles = sf.decoder(decoded_selfie)
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_count += 1
        except Exception:
            pass
                    
    return valid_count, reconstructed_selfies


if __name__ == '__main__':
    
    VAE_CHECKPOINT = "../train_vae/checkpoints/molgen_vae_epoch_2"
    batch_size = 128
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    vae = MolecularVAE(model_name="zjunlp/MolGen-large", latent_dim=256)
    state_dict = torch.load(f"{VAE_CHECKPOINT}/pytorch_model.bin")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    vae.load_state_dict(state_dict)
    vae.to(device)
    vae.eval()

    vae_tokenizer = AutoTokenizer.from_pretrained(VAE_CHECKPOINT)

    test_dataset = load_dataset('alessandronascimento/zinc20_chembl36', split='test')
    test_dataset = test_dataset.shuffle(seed=42).select(range(100000)) 

    total_valid_count = 0
    total_reconstructed = 0

    for i in tqdm(range(0, len(test_dataset), batch_size), desc="Testing VAE Batches"):
        batch_selfies = test_dataset[i:i + batch_size]['selfies']
        inputs = vae_tokenizer(batch_selfies, padding='max_length', truncation=True, return_tensors='pt').to(device)
        valid_count, reconstructed_selfies = test_vae_validity_batched(vae, vae_tokenizer, inputs, device)
        total_valid_count += valid_count
        total_reconstructed += len(reconstructed_selfies)
    
    overall_validity_rate = (total_valid_count / total_reconstructed) * 100
    print(f"Overall Validity Rate: {overall_validity_rate:.2f}%")
