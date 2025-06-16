import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, get_linear_schedule_with_warmup
import wandb
from torch.optim import AdamW

# Paths
tokenized_path = "/home/user/cs336-a4-data/tokenized_outputs/combined.npy"
validation_path = "/home/user/data/paloma/tokenized_paloma_c4_100_domains_validation.bin"
model_save_path = "./gpt2-training"

# Hyperparameters
block_size = 128
batch_size = 16
epochs = 1
lr = 1e-4

# Load tokenized data
all_ids = np.memmap(tokenized_path, dtype=np.uint16, mode="r")
val_ids = np.memmap(validation_path, dtype=np.uint16, mode="r")
print(f"Loaded {len(all_ids)} training tokens.")
print(f"Loaded {len(val_ids)} validation tokens.")

# Dataset
class TokenDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) // self.block_size

    def __getitem__(self, idx):
        i = idx * self.block_size
        x = torch.tensor(self.data[i:i+self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[i+1:i+self.block_size+1], dtype=torch.long)
        return x, y

dataset = TokenDataset(all_ids, block_size)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

val_subset_size = 10000  # or any number you prefer
val_ids_subset = val_ids[:val_subset_size]
val_dataset = TokenDataset(val_ids_subset, block_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model & Tokenizer
config = GPT2Config(
    vocab_size=50257,  # GPT-2's vocab size
    n_positions=1024,  # context size
    n_ctx=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
)
model = GPT2LMHeadModel(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.train()

optimizer = AdamW(model.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=10, num_training_steps=len(loader)*epochs
)

wandb.init(project="gpt2-pretraining", name="gpt2-small-from-scratch")

def evaluate(model, val_loader, device):
    model.eval()
    eval_losses = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(input_ids=x, labels=y)
            eval_losses.append(outputs.loss.item())
    eval_loss = sum(eval_losses) / len(eval_losses)
    return eval_loss

# Training loop
for epoch in range(epochs):
    model.train()
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        outputs = model(input_ids=x, labels=y)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if step % 100 == 0:
            print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")
            wandb.log({"train/loss": loss.item(), "step": epoch * len(loader) + step, "epoch": epoch})
        # Evaluate every 200 steps
        if step % 50 == 0 and step > 0:
            eval_loss = evaluate(model, val_loader, device)
            print(f"Epoch {epoch} Step {step} Eval Loss {eval_loss:.4f}")
            wandb.log({"eval/loss": eval_loss, "step": epoch * len(loader) + step, "epoch": epoch})
    # Evaluation at end of epoch
    eval_loss = evaluate(model, val_loader, device)
    print(f"Epoch {epoch} Eval Loss {eval_loss:.4f}")
    wandb.log({"eval/loss": eval_loss, "epoch": epoch})

# Save model
model.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")
