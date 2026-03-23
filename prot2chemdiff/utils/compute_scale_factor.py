import torch
from torch.utils.data import DataLoader
import tqdm
from prot2chemdiff.utils.latent_dataset import StreamingLatentDataset

if __name__ == "__main__":
    data_dir = './'
    dataset = StreamingLatentDataset(data_dir, scale_factor=1.0, shuffle_files=True, shuffle_chunks=True)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=2)

    total_variance = 0.0
    total_samples = 0

    for batch in tqdm.tqdm(dataloader):
        latents, proteins, affinities = batch
        total_variance += (latents ** 2).sum().item()
        total_samples += latents.numel() # Total number of latent scalars
    
    global_std = (total_variance / total_samples) ** 0.5
    print(f"True Global Latent STD: {global_std}")

    global_scale_factor = 1.0 / global_std
    print(f"Perfect Global Scale Factor: {global_scale_factor}")


