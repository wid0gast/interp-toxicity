from tqdm import tqdm, trange

from datasets import load_dataset
import pandas as pd
import functools
import sys
from pathlib import Path
from typing import Callable

# import circuitsvis as cv
import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import eindex
# from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    HookedEncoderDecoder,
    HookedEncoder,
    utils,
)
from transformer_lens.hook_points import HookPoint

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"


dataset = load_dataset("csv", data_files={"train": 'jigsaw/train.csv', "test": 'jigsaw/test.csv'})

# Load GPT-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn’t have a padding token

# Tokenization function
def tokenize_data(example):
    return tokenizer(example["comment_text"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")

# Apply tokenization
dataset = dataset.map(tokenize_data, batched=True)
dataset = dataset.rename_column("toxic", "labels")  # Rename for consistency
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


# Custom PyTorch Dataset wrapper
class JigsawDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {key: self.dataset[idx][key] for key in ["input_ids", "labels"]}

# Create PyTorch DataLoaders
batch_size = 256
train_dataloader = DataLoader(JigsawDataset(dataset["train"]), batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(JigsawDataset(dataset["test"]), batch_size=batch_size, shuffle=False)


device = "cuda" if torch.cuda.is_available() else "cpu"

# Load GPT-2 into transformer_lens
model = HookedTransformer.from_pretrained("gpt2", device=device)

class GPT2Classifier(nn.Module):
    def __init__(self, transformer, num_classes=2):
        super().__init__()
        self.transformer = transformer
        self.classifier = nn.Linear(transformer.cfg.d_model, num_classes)  # d_model = 768

    def forward(self, input_ids):
        _, cache = self.transformer.run_with_cache(input_ids)  # Get cache

        # Extract final hidden states from residual stream
        hidden_states = cache["resid_post", -1]  # Shape: [batch, seq_len, hidden_dim]

        # Use last token’s hidden state for classification
        logits = self.classifier(hidden_states[:, -1, :])  # Shape: [batch, num_classes]
        return logits

    def run_with_hooks(self, tokens, fwd_hooks):
        logits = self.transformer.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
        return self.classifier(logits)


# Initialize model and optimizer
num_classes = 2
classifier = GPT2Classifier(model, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=5e-5)

# Training loop
num_epochs = 1
best_epoch = -1
best_loss = 100

for epoch in num_epochs:
    print(f"Epoch {epoch+1}")
    classifier.train()
    total_train_loss = 0
    total_val_loss = 0
    correct_preds = 0

    for batch in tqdm(train_dataloader):
        input_ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)

        optimizer.zero_grad()
        logits = classifier(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
    classifier.eval()
    for batch in val_dataloader:
        input_ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)

        logits = classifier(input_ids)
        loss = criterion(logits, labels)

        total_val_loss += loss.item()
        correct_preds += sum(logits.argmax(dim=1) == labels).item()

    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_val_loss = total_val_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss}")
    save_dir = "finetuned_gpt2"
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        best_epoch = epoch
        torch.save(classifier.state_dict(), os.path.join(save_dir, f"classifier.pth"))
        torch.save(model.state_dict(), os.path.join(save_dir, f"transformer.pth"))

def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
    logprobs = logits.log_softmax(dim=-1)
    # We want to get logprobs[b, s, tokens[b, s+1]], in eindex syntax this looks like:
    correct_logprobs = eindex(logprobs, tokens, "b s [b s+1]")
    return correct_logprobs

def head_zero_ablation_hook(
    z: Float[Tensor, "batch seq n_heads d_head"],
    hook: HookPoint,
    head_index_to_ablate: int,
) -> None:
    z[:, :, head_index_to_ablate, :] = 0.0

def get_ablation_scores(
    classifier: GPT2Classifier,
    tokens: Int[Tensor, "batch seq"],
    ablation_function: Callable = head_zero_ablation_hook,
) -> Float[Tensor, "n_layers n_heads"]:
    """
    Returns a tensor of shape (n_layers, n_heads) containing the increase in cross entropy loss from ablating the output
    of each head.
    """
    # Initialize an object to store the ablation scores
    ablation_scores = t.zeros((classifier.transformer.cfg.n_layers, classifier.transformer.cfg.n_heads), device=classifier.transformer.cfg.device)

    # Calculating loss without any ablation, to act as a baseline
    classifier.transformer.reset_hooks()
    seq_len = (tokens.shape[1] - 1) // 2
    logits = classifier(tokens, return_type="logits")
    loss_no_ablation = -get_log_probs(logits, tokens)[:, -(seq_len - 1) :].mean()

    for layer in tqdm(range(classifier.transformer.cfg.n_layers)):
        for head in range(classifier.transformer.cfg.n_heads):
            # Use functools.partial to create a temporary hook function with the head number fixed
            temp_hook_fn = functools.partial(ablation_function, head_index_to_ablate=head)
            # Run the model with the ablation hook
            ablated_logits = classifier.run_with_hooks(tokens, fwd_hooks=[(utils.get_act_name("z", layer), temp_hook_fn)])
            # Calculate the loss difference (= negative correct logprobs), only on the last `seq_len` tokens
            loss = -get_log_probs(ablated_logits.log_softmax(-1), tokens)[:, -(seq_len - 1) :].mean()
            # Store the result, subtracting the clean loss so that a value of zero means no change in loss
            ablation_scores[layer, head] = loss - loss_no_ablation

    return ablation_scores

tokens = dataset['test']['input_ids'][:100]
ablation_scores = get_ablation_scores(classifier, tokens, head_zero_ablation_hook)
torch.save(ablation_scores, "ablation_scores")
imshow(
    ablation_scores,
    labels={"x": "Head", "y": "Layer", "color": "Logit diff"},
    title="Loss Difference After Ablating Heads",
    text_auto=".2f",
    width=900,
    height=350,
)