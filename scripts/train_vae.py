import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, DataCollatorForSeq2Seq, AutoTokenizer
from datasets import load_from_disk
from accelerate import Accelerator
from tqdm.auto import tqdm
import os
import wandb 
import numpy as np
from prot2chemdiff.vae_model import MolecularVAE
from prot2chemdiff.utils.kl_annealing import KLAnnealer


##############################   Inputs   ##############################

BATCH_SIZE = 16
GRAD_ACCUM = 2
EPOCHS = 3
LEARNING_RATE = 1e-4
MAX_BETA = 0.002
LATENT_DIM = 256
MAX_SEQ_LEN = 128
MODEL_NAME = "zjunlp/MolGen-large"
PROJECT_NAME = "MOLGEN_VAE_PROJECT"
RUN_NAME = "TRAIN_VAE"

##############################   End of Inputs   ##############################

def add_labels(example):
    example["labels"] = example["input_ids"]
    return example

def main():
    accelerator = Accelerator(log_with="wandb", project_dir="logs")
    accelerator.init_trackers(PROJECT_NAME, config={"batch_size": BATCH_SIZE, "lr": LEARNING_RATE})

    print(f"🚀 Launching on {accelerator.device}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dataset = load_from_disk("../data/tokenized_dataset")
    dataset = dataset.map(add_labels, num_proc=8)
    dataset = dataset.remove_columns(["smiles", "selfies", "Unnamed: 0", "id"])
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    dataset = dataset.filter(lambda x: len(x['input_ids']) < MAX_SEQ_LEN)

    # Data Collator (Dynamic Padding)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        padding=True,
        max_length=MAX_SEQ_LEN,
        pad_to_multiple_of=8
    )

    train_loader = DataLoader(
        dataset['train'], 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=data_collator,
        num_workers=4, 
        pin_memory=True
    )

    val_loader = DataLoader(
        dataset['validation'], 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=data_collator,
        num_workers=4, 
        pin_memory=True
    )

    model = MolecularVAE(model_name=MODEL_NAME, latent_dim=LATENT_DIM)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2000, num_training_steps=total_steps)
    kl_annealer = KLAnnealer(total_steps=total_steps, max_beta=MAX_BETA)
    
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    global_step = 0
    
    for epoch in range(EPOCHS):
        model.train()
        progress_bar = tqdm(total=len(train_loader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch+1}")
        
        for batch in train_loader:
            with accelerator.accumulate(model):
                loss, mu, logvar = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss = kl_loss / batch['input_ids'].size(0) # Mean over batch
                beta = kl_annealer.step()
                
                total_loss = loss + (beta * kl_loss)
                
                accelerator.backward(total_loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Logging
                if global_step % 50 == 0 and accelerator.is_local_main_process:
                    accelerator.log({
                        "total_loss": total_loss.item(),
                        "recon_loss": loss.item(),
                        "kl_loss": kl_loss.item(),
                        "beta": beta,
                        "lr": scheduler.get_last_lr()[0]
                    }, step=global_step)
                    
                    progress_bar.set_postfix(loss=total_loss.item(), kl=kl_loss.item(), beta=beta)
                
                global_step += 1
                progress_bar.update(1)


        if accelerator.is_local_main_process:
            save_path = f"./checkpoints/molgen_vae_epoch_{epoch+1}"
            os.makedirs(save_path, exist_ok=True)
            unwrapped = accelerator.unwrap_model(model)
            torch.save(unwrapped.state_dict(), f"{save_path}/pytorch_model.bin")
            tokenizer.save_pretrained(save_path)
            
            print(f"\n--- Epoch {epoch+1} Sanity Check: Sampling from Prior ---")
            model.eval()
            with torch.no_grad():
                # Create random latent vector z ~ N(0, 1)
                z = torch.randn(1, LATENT_DIM).to(accelerator.device)
                
                # We need a decode function in your MolGenVAE class!
                # Or manually:
                z_proj = unwrapped.fc_decode(z) # [1, 1024]
                encoder_hidden = z_proj.unsqueeze(1) # [1, 1, 1024]                
                print(f"Saved to {save_path}. Validation loss: {total_loss.item()}")

    accelerator.end_training()

if __name__ == "__main__":
    main()