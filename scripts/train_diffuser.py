import torch
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
from prot2chemdiff.utils.latent_dataset import StreamingLatentDataset
from torch.utils.data import DataLoader
from prot2chemdiff.diffuser_lightning import Prot2Chem_Diffusion


if __name__ == "__main__":

    torch.set_float32_matmul_precision('medium')

    train_dataset = StreamingLatentDataset('../data/clustered_jglaser_binding_affinity/latents_processed_train', shuffle_files=True, shuffle_chunks=True)
    val_dataset   = StreamingLatentDataset('../data/clustered_jglaser_binding_affinity/latents_processed_validation', shuffle_files=True, shuffle_chunks=True)
    
    train_loader = DataLoader(train_dataset, batch_size=256, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=256, num_workers=2, pin_memory=True)

    model = Prot2Chem_Diffusion(learning_rate=1e-4, drop_prob=0.1)

    wandb_logger = WandbLogger(project="Prot2Chem_Diffusion")

    trainer = pl.Trainer(
        max_epochs=300,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        log_every_n_steps=200,
        val_check_interval=1000,
        logger=wandb_logger,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)