# %%
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm, trange
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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

# %%
import torch
from contextlib import contextmanager

def make_zero_head_hook(head_idx: int, num_heads: int, head_dim: int):
    def hook(module, input, output):
        # In BERT's self-attention, the output is a tuple of (context_layer, attention_probs)
        context_layer, attention_probs = output

        # Reshape attention probabilities to separate heads
        # attention_probs shape: (batch_size, num_heads, seq_len, seq_len)
        # Zero out the attention weights for the specified head
        attention_probs[:, head_idx, :, :] = 0.0
        
        # Get the value projections from the input
        value_layer = input[0]  # This is the value projection
        b, s, h = value_layer.shape
        value_layer = value_layer.view(b, s, num_heads, head_dim)
        
        # Transpose value_layer to match attention_probs dimensions
        # From (batch_size, seq_len, num_heads, head_dim) to (batch_size, num_heads, seq_len, head_dim)
        value_layer = value_layer.transpose(1, 2)
        
        # Compute new context layer with zeroed attention weights
        # attention_probs: (batch_size, num_heads, seq_len, seq_len)
        # value_layer: (batch_size, num_heads, seq_len, head_dim)
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Transpose back to original shape
        context_layer = context_layer.transpose(1, 2)
        context_layer = context_layer.reshape(b, s, h)
        
        return (context_layer, attention_probs)

    return hook


@contextmanager
def zero_head(model, layer_idx: int, head_idx: int):
    """
    Context-manager that installs the hook, yields, then removes it.
    """
    sa = model.bert.encoder.layer[layer_idx].attention.self
    hd = model.config.hidden_size // model.config.num_attention_heads
    handle = sa.register_forward_hook(
        make_zero_head_hook(head_idx, model.config.num_attention_heads, hd)
    )
    try:
        yield
    finally:
        handle.remove()


# %%
criterion = nn.CrossEntropyLoss()

# %%
def get_ablation_scores_plain(
    classifier: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
):
    n_layers     = classifier.config.num_hidden_layers
    n_heads      = classifier.config.num_attention_heads
    d_head       = classifier.config.hidden_size // n_heads
    device       = input_ids.device
    scores       = torch.zeros(n_layers, n_heads, device=device)
    predictions  = {}

    # baseline
    with torch.no_grad():
        base_logits = classifier(input_ids, attention_mask)
        base_preds = torch.argmax(base_logits.logits, dim=-1)
        base_loss   = criterion(base_logits.logits, labels)

    # per-head ablation
    for L in trange(n_layers):
        for H in range(n_heads):
            with zero_head(classifier, L, H):
                with torch.no_grad():
                    logits = classifier(input_ids)
                    loss   = criterion(logits.logits, labels)
                    preds = torch.argmax(logits.logits, dim=-1)
            scores[L, H] = loss - base_loss        # Δ-loss
            predictions[(L, H)] = preds.detach().cpu()
    return scores, base_preds, predictions


# %%
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer  = AutoTokenizer.from_pretrained("bert-base-uncased")

# %%
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    output_attentions=True      # optional
).to(device)
model.eval()

# %%
model.load_state_dict(torch.load('bert_classifier_vanilla/model_epoch_2_acc_0.9236.pt', map_location=device))

# %%
# dataset = load_dataset("csv", data_files={'logs/input_reduction_log.csv'})
df = pd.read_csv('data/raw/jigsaw/test.csv')
dataset = Dataset.from_pandas(df.groupby('toxic').sample(n=5000).reset_index(drop=True))
# dataset = load_dataset('csv', data_files={'test': 'toxigen_alice.csv'})
# Tokenization function
def tokenize_data(example):
    return tokenizer(example["comment_text"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")

# Apply tokenization
dataset = dataset.map(tokenize_data, batched=True)
dataset = dataset.rename_column("toxic", "labels")  # Rename for consistency
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# %%
class JigsawDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {key: self.dataset[idx][key] for key in ["input_ids", "labels", "attention_mask"]}

# Create PyTorch DataLoaders
batch_size = 16
val_dataloader = DataLoader(JigsawDataset(dataset), batch_size=batch_size, shuffle=False)

# %%
print('patching')
ablation_scores = torch.zeros(12,12, device=device)
base_preds = []
ablated_preds = {layer: {head : [] for head in range(model.config.num_attention_heads)} for layer in range(model.config.num_hidden_layers)}
for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
    input_ids, labels, attention_mask = batch["input_ids"].to(device), batch["labels"].to(device), batch["attention_mask"].to(device)
    tmp_ablation_scores, base_pred, ablated_pred_list = get_ablation_scores_plain(model, input_ids, attention_mask, labels)
    ablation_scores += tmp_ablation_scores
    base_preds += base_pred.tolist()
    for layer in range(model.config.num_hidden_layers):
        for head in range(model.config.num_attention_heads):
            ablated_preds[layer][head] += ablated_pred_list[(layer, head)].tolist()
    if i % 16 == 0 or i == len(val_dataloader) - 1:
        torch.save(ablation_scores / ((i+1) * batch_size), "bert_ablation_scores_jigsaw_perturbed.pth")
        with open('bert_ablated_preds_jigsaw_perturbed.json', 'w') as f:
            json.dump(ablated_preds, f)
        with open('bert_base_preds_jigsaw_perturbed.json', 'w') as f:
            json.dump(base_preds, f)


