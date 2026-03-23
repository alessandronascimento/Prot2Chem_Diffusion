from huggingface_hub import hf_hub_download
import torch
from vae_model import MolecularVAE
from diffuser_lightning import Prot2Chem_Diffusion

def load_pretrained_models(repo_id="alessandronascimento/prot2chemdiff"):
    """
    Automatically downloads weights from HF and loads the models.
    """
    print("Fetching weights from HuggingFace...")
    
    # This downloads the file if it isn't cached, or uses the local cache if it is!
    vae_path = hf_hub_download(repo_id=repo_id, filename="vae_weights.ckpt")
    diff_path = hf_hub_download(repo_id=repo_id, filename="diffusion_weights.ckpt")
    
    # Load Models
    vae_model = MolecularVAE(model_name="zjunlp/MolGen-large", latent_dim=256)
    state_dict = torch.load("vae_weights.ckpt")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    vae_model.load_state_dict(state_dict)
    diffuser = Prot2Chem_Diffusion.load_from_checkpoint(diff_path)
    
    return vae, diffuser