import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from tqdm import tqdm, trange
import os

# # Load IMDb dataset (binary classification)
# dataset = load_dataset("imdb")

# # Load GPT-2 tokenizer
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn’t have a padding token

# # Tokenization function
# def tokenize_data(example):
#     return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")

# # Apply tokenization
# dataset = dataset.map(tokenize_data, batched=True)
# dataset = dataset.rename_column("label", "labels")  # Rename for consistency
# dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# # Custom PyTorch Dataset wrapper
# class IMDbDataset(Dataset):
#     def __init__(self, dataset):
#         self.dataset = dataset

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         return {key: self.dataset[idx][key] for key in ["input_ids", "labels"]}

# # Create PyTorch DataLoaders
# batch_size = 8
# train_dataloader = DataLoader(IMDbDataset(dataset["train"]), batch_size=batch_size, shuffle=True)
# val_dataloader = DataLoader(IMDbDataset(dataset["test"]), batch_size=batch_size, shuffle=False)

# # Load GPT-2 into transformer_lens
# model = HookedTransformer.from_pretrained("gpt2", device="cuda:4")

# class GPT2Classifier(nn.Module):
#     def __init__(self, transformer, num_classes=2):
#         super().__init__()
#         self.transformer = transformer
#         self.classifier = nn.Linear(transformer.cfg.d_model, num_classes)  # d_model = 768

#     def forward(self, input_ids):
#         _, cache = self.transformer.run_with_cache(input_ids)  # Get cache

#         # Extract final hidden states from residual stream
#         hidden_states = cache["resid_post", -1]  # Shape: [batch, seq_len, hidden_dim]

#         # Use last token’s hidden state for classification
#         logits = self.classifier(hidden_states[:, -1, :])  # Shape: [batch, num_classes]
#         return logits

    
# # Initialize model and optimizer
# num_classes = 2
# classifier = GPT2Classifier(model, num_classes).to("cuda:4")

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(classifier.parameters(), lr=5e-5)

# # Enable gradients for transformer parameters
# for param in model.parameters():
#     param.requires_grad_(True)

# # Training loop
# num_epochs = 3

# for epoch in trange(num_epochs):
#     classifier.train()
#     total_loss = 0

#     for batch in tqdm(train_dataloader):
#         input_ids, labels = batch["input_ids"].to("cuda:4"), batch["labels"].to("cuda:4")

#         optimizer.zero_grad()
#         logits = classifier(input_ids)
#         loss = criterion(logits, labels)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     avg_loss = total_loss / len(train_dataloader)
#     print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")


# save_dir = "finetuned_gpt2"
# os.makedirs(save_dir, exist_ok=True)

# # Save after training
# torch.save(classifier.state_dict(), os.path.join(save_dir, "classifier.pth"))
# torch.save(model.state_dict(), os.path.join(save_dir, "transformer.pth"))

# print("Model saved successfully.")

# Load model
save_dir = "finetuned_gpt2"
device = "cuda:4"

# Reload GPT-2 as a HookedTransformer
from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained("gpt2", device=device)
model.load_state_dict(torch.load(os.path.join(save_dir, "transformer.pth")))
# print(model.mod_dict.keys())
# Reload Classifier Head
import torch.nn as nn

class GPT2Classifier(nn.Module):
    def __init__(self, transformer, num_classes=2):
        super().__init__()
        self.transformer = transformer
        self.classifier = nn.Linear(transformer.cfg.d_model, num_classes)  # d_model = 768

    def forward(self, input_ids):
        _, cache = self.transformer.run_with_cache(input_ids)  # Get cache
        hidden_states = cache["resid_post", -1]  # Final hidden states
        logits = self.classifier(hidden_states[:, -1, :])  # Last token
        return logits

# Load classifier
classifier = GPT2Classifier(model, num_classes=2).to(device)
classifier.load_state_dict(torch.load(os.path.join(save_dir, "classifier.pth")))
classifier.eval()

print("Model loaded successfully.")

input_ids = torch.randint(0, 50257, (2, 128)).to("cuda")  # Fake batch of 2
logits = classifier(input_ids)
print("Logits shape:", logits.shape)  # Should be [2, num_classes]


# Load IMDb dataset
dataset = load_dataset("imdb")

# Load GPT-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 needs padding token

# Tokenization function
def tokenize_batch(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")

# Apply tokenization
dataset = dataset.map(tokenize_batch, batched=True)

# Rename 'label' → 'labels' to match model expectations
dataset = dataset.rename_column("label", "labels")

# Set correct format for PyTorch
dataset.set_format(type="torch", columns=["input_ids", "labels"])

# Create DataLoader for evaluation
val_dataloader = DataLoader(dataset["test"], batch_size=1, shuffle=False)

# Fetch a single batch
batch = next(iter(val_dataloader))
input_ids = batch["input_ids"].to("cuda")
original_logits = classifier(input_ids)
original_pred = torch.argmax(original_logits, dim=-1).item()

print(f"Original Prediction: {original_pred}")

# Iterate over all layers and heads
num_layers = model.cfg.n_layers  # Number of layers
num_heads = model.cfg.n_heads  # Number of heads per layer

for layer in range(num_layers):
    for head in range(num_heads):
        
        def zero_out_head(attn_output, hook):
            attn_output[:, :, head, :] = 0  # Zero out current head

        # Add the hook
        hook_name = f"blocks.{layer}.attn.hook_z"
        model.add_hook(hook_name, zero_out_head, "fwd")

        # Run inference with modified attention
        with torch.no_grad():
            ablated_logits = classifier(input_ids)
            ablated_pred = torch.argmax(ablated_logits, dim=-1).item()
        
        # Print result
        print(f"Layer {layer}, Head {head} - Prediction: {ablated_pred}")

        # Remove the hook after running inference
        model.reset_hooks()


# def zero_out_heads(attn_output, hook):
#     attn_output[:, 0, :, :] = 0  # Zero out head 0 across all tokens

# classifier.eval()
# with torch.no_grad():
#     for batch in val_dataloader:
#         input_ids = batch["input_ids"].to("cuda")
#         logits = classifier(input_ids)
#         preds = torch.argmax(logits, dim=-1)
#         print("Predictions:", preds.tolist())
#         break  # Print only one batch

# # Add the hook before inference
# model.add_hook("blocks.1.attn.hook_z", zero_out_heads, "fwd")

# # Evaluate on a batch
# classifier.eval()
# with torch.no_grad():
#     for batch in val_dataloader:
#         input_ids = batch["input_ids"].to("cuda")
#         logits = classifier(input_ids)
#         preds = torch.argmax(logits, dim=-1)
#         print("Predictions:", preds.tolist())
#         break  # Print only one batch
