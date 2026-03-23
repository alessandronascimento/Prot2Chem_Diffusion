#@title VAE Model
from transformers import BartForConditionalGeneration, AutoTokenizer
import torch
import torch.nn as nn

class MolecularVAE(nn.Module):
    def __init__(self, model_name="zjunlp/MolGen-large", latent_dim=256):
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(model_name)
        self.hidden_dim = self.bart.config.d_model # 1024

        # VAE
        self.fc_mu = nn.Linear(self.hidden_dim, latent_dim)
        self.fc_var = nn.Linear(self.hidden_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.hidden_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input_ids, attention_mask, labels=None):
        # ENCODER
        encoder_outputs = self.bart.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = encoder_outputs.last_hidden_state[:, 0, :]

        # VAE
        mu = self.fc_mu(pooled_output)
        logvar = self.fc_var(pooled_output)
        z = self.reparameterize(mu, logvar) # [Batch, 256]

        # DECODER
        # Expand z back to hidden size
        z_projected = self.fc_decode(z) # [Batch, 1024]

        encoder_hidden_states = z_projected.unsqueeze(1) # [Batch, 1, 1024]
        encoder_outputs_tuple = (encoder_hidden_states,)

        outputs = self.bart(
            input_ids=None, # BART handles this internally if labels are provided
            encoder_outputs=encoder_outputs_tuple,
            labels=labels
        )

        return outputs.loss, mu, logvar