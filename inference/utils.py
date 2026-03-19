import torch
from transformers.modeling_outputs import BaseModelOutput

def encode_to_latent(vae_model, inputs, use_mean=True):
    """
    Extracts the latent vector from the VAE.
    use_mean=True: Returns the deterministic center (best for reconstruction).
    use_mean=False: Returns a noisy sample (best for exploring the space).
    """
    with torch.no_grad():
        encoder_outputs = vae_model.bart.model.encoder(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        pooled_output = encoder_outputs.last_hidden_state[:, 0, :]
        
        mu = vae_model.fc_mu(pooled_output)
        
        if use_mean:
            return mu
        else:
            logvar = vae_model.fc_var(pooled_output)
            return vae_model.reparameterize(mu, logvar)

def decode_from_latent(vae_model, z, max_length=128):
    """
    Takes a latent vector [Batch, 256] and auto-regressively generates tokens.
    """
    with torch.no_grad():
        z_projected = vae_model.fc_decode(z) # [Batch, 1024]
    
        encoder_hidden_states = z_projected.unsqueeze(1) # [Batch, 1, 1024]
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
        
        generated_ids = vae_model.bart.generate(
            encoder_outputs=encoder_outputs,
            max_length=max_length,
            bos_token_id=vae_model.bart.config.bos_token_id,
            eos_token_id=vae_model.bart.config.eos_token_id,
            pad_token_id=vae_model.bart.config.pad_token_id,
        )
        
        return generated_ids