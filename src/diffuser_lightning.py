import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from diffusers import DDPMScheduler
from diffuser import ConditionalDiT

class Prot2Chem_Diffusion(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, drop_prob=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.model = ConditionalDiT(num_blocks=6)
        
        # Hugging Face Scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon"
        )

    def forward(self, noisy_latents, timesteps, proteins, affinities):
        return self.model(noisy_latents, timesteps, proteins, affinities)

    def _shared_step(self, batch, batch_idx, is_train=True):
        latents, proteins, affinities = batch
        bsz = latents.shape[0]

        if affinities.dim() == 1:
            affinities = affinities.view(-1, 1)
            
        noise = torch.randn_like(latents)

        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (bsz,), 
            device=self.device, dtype=torch.long
        )

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        if is_train and self.hparams.drop_prob > 0:
            drop_mask = torch.rand(bsz, device=self.device) < self.hparams.drop_prob
            
            proteins = torch.where(drop_mask.view(-1, 1), torch.zeros_like(proteins), proteins)
            affinities = torch.where(drop_mask.view(-1, 1), torch.zeros_like(affinities), affinities)

        noise_pred = self.model(noisy_latents, timesteps, proteins, affinities)

        loss = F.mse_loss(noise_pred, noise)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, is_train=True)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, is_train=False)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4)
        return optimizer

