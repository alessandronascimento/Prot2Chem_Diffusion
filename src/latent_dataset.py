import os
from glob import glob
import torch
import random
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

class StreamingLatentDataset(IterableDataset):
    def __init__(self, data_dir, scale_factor=2.15709, shuffle_files=True, shuffle_chunks=True):
        """
        Args:
            data_dir (str): Folder containing the precomputed .pt chunk files.
            scale_factor (float): Multiplier to ensure latents have ~1.0 standard deviation.
            shuffle_files (bool): If True, randomizes the order chunks are loaded.
            shuffle_chunks (bool): If True, randomizes the order of samples inside each chunk.
        """
        super().__init__()
        self.data_dir = data_dir
        self.scale_factor = scale_factor
        self.shuffle_files = shuffle_files
        self.shuffle_chunks = shuffle_chunks
        
        self.pt_files = sorted(glob(os.path.join(data_dir, "*.pt")))
        if len(self.pt_files) == 0:
            raise ValueError(f"No .pt files found in {data_dir}!")
            
        print(f"Dataset initialized with {len(self.pt_files)} chunks.")

    def __iter__(self):
        worker_files = list(self.pt_files)
        if self.shuffle_files:
            random.shuffle(worker_files)

        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            worker_files = worker_files[worker_id::num_workers]

        for pt_file in worker_files:
            chunk_data = torch.load(pt_file, map_location='cpu', weights_only=False)
            
            proteins = chunk_data['protein_embeddings']
            ligands = chunk_data['ligand_latents']
            affinities = chunk_data['affinities']
            
            num_samples = proteins.size(0)
            indices = list(range(num_samples))
            
            if self.shuffle_chunks:
                random.shuffle(indices)
                
            for idx in indices:
                protein = proteins[idx].float()
                latent = ligands[idx].float()
                affinity = affinities[idx].float()
                
                scaled_latent = latent * self.scale_factor
                
                yield scaled_latent, protein, affinity