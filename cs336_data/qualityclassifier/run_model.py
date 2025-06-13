import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
import wandb

# Paths
tokenized_path = "/home/user/cs336-a4-data/tokenized_outputs/tokenized_example.npy"
model_save_path = "./gpt2-training"

# Hyperparameters
block_size = 128
batch_size = 2
epochs = 1
lr = 5e-5

# Load tokenized data
all_ids = np.memmap(tokenized_path, dtype=np.uint16, mode="r")
print(f"Loaded {len(all_ids)} tokens.")

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
    # Evaluation at end of epoch
    model.eval()
    eval_losses = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(input_ids=x, labels=y)
            eval_losses.append(outputs.loss.item())
    eval_loss = sum(eval_losses) / len(eval_losses)
    print(f"Epoch {epoch} Eval Loss {eval_loss:.4f}")
    wandb.log({"eval/loss": eval_loss, "epoch": epoch})

# Save model
model.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")
