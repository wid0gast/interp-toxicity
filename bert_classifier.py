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
from eindex import eindex
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
import json

# os.environ['CUDA_VISIBLE_DEVICES'] = "7"


dataset = load_dataset("csv", data_files={'input_reduction_log.csv'})
print(len(dataset['train']))
# dataset = load_dataset('csv', data_files={'test': 'toxigen_alice.csv'})

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenization function
def tokenize_data(example):
    return tokenizer(example["original_text"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")

# Apply tokenization
dataset = dataset.map(tokenize_data, batched=True)
dataset = dataset.rename_column("ground_truth_output", "labels")  # Rename for consistency
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
batch_size = 64
val_dataloader = DataLoader(JigsawDataset(dataset["train"]), batch_size=batch_size, shuffle=True)
# val_dataloader = DataLoader(JigsawDataset(dataset["test"]), batch_size=batch_size, shuffle=False)


device = "cuda:6" if torch.cuda.is_available() else "cpu"

class BERTClassifier(nn.Module):
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
        with torch.no_grad():
            with self.transformer.hooks(fwd_hooks=fwd_hooks):
                _, cache = self.transformer.run_with_cache(tokens)
            hidden_states = cache["resid_post", -1] 
            logits = self.classifier(hidden_states[:, -1, :])
        return logits


# # Initialize model and optimizer
# num_classes = 2
# model = HookedEncoder.from_pretrained("bert-base-uncased", device=device)
# # model.load_state_dict(torch.load("finetuned_gpt2/transformer.pth", map_location=device))
# model.to(device)
# classifier = BERTClassifier(model, num_classes)
# # classifier.load_state_dict(torch.load("finetuned_gpt2/classifier.pth", map_location=device))
# classifier.to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(classifier.parameters(), lr=5e-5)

# # Training loop
# num_epochs = 10
# best_epoch = -1
# best_loss = 100

# for epoch in range(num_epochs):
#     print(f"Epoch {epoch+1}")
#     classifier.train()
#     total_train_loss = 0
#     total_val_loss = 0
#     correct_preds = 0

#     for batch in tqdm(train_dataloader):
#         input_ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)

#         optimizer.zero_grad()
#         logits = classifier(input_ids)
#         loss = criterion(logits, labels)
#         loss.backward()
#         optimizer.step()

#         total_train_loss += loss.item()
#     classifier.eval()
#     for batch in val_dataloader:
#         input_ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)

#         logits = classifier(input_ids)
#         loss = criterion(logits, labels)

#         total_val_loss += loss.item()
#         correct_preds += sum(logits.argmax(dim=1) == labels).item()

#     avg_train_loss = total_train_loss / len(train_dataloader)
#     avg_val_loss = total_val_loss / len(train_dataloader)
#     print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss}")
#     save_dir = "finetuned_bert/jigsaw"
#     if avg_val_loss < best_loss:
#         best_loss = avg_val_loss
#         best_epoch = epoch
#         torch.save(classifier.state_dict(), os.path.join(save_dir, f"classifier.pth"))
#         torch.save(model.state_dict(), os.path.join(save_dir, f"transformer.pth"))


# Initialize model and optimizer
num_classes = 2
model = HookedEncoder.from_pretrained("bert-base-uncased", device=device)
model.load_state_dict(torch.load("finetuned_bert/jigsaw/transformer.pth", map_location=device))
model.to(device)
classifier = BERTClassifier(model, num_classes)
classifier.load_state_dict(torch.load("finetuned_bert/jigsaw/classifier.pth", map_location=device))
classifier.to(device)

criterion = nn.CrossEntropyLoss()

def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
    logprobs = logits.log_softmax(dim=-1)
    # We want to get logprobs[b, s, tokens[b, s+1]], in eindex syntax this looks like:
    print(logits.shape, logprobs.shape, tokens.shape)
    correct_logprobs = eindex(logprobs, tokens, "b s [b s+1]")
    return correct_logprobs

def head_zero_ablation_hook(
    z: Float[Tensor, "batch seq n_heads d_head"],
    hook: HookPoint,
    head_index_to_ablate: int,
) -> None:
    z[:, :, head_index_to_ablate, :] = 0.0

def get_ablation_scores(
    classifier: BERTClassifier,
    tokens: Int[Tensor, "batch seq"],
    labels,
    ablation_function: Callable = head_zero_ablation_hook,
) -> Float[Tensor, "n_layers n_heads"]:
    """
    Returns a tensor of shape (n_layers, n_heads) containing the increase in cross entropy loss from ablating the output
    of each head.
    """
    # Initialize an object to store the ablation scores
    ablation_scores = t.zeros((classifier.transformer.cfg.n_layers, classifier.transformer.cfg.n_heads), device=classifier.transformer.cfg.device)
    ablated_pred_list = {}
    # Calculating loss without any ablation, to act as a baseline
    classifier.transformer.reset_hooks()
    logits = classifier(tokens)
    preds = logits.argmax(dim=-1)
    loss_no_ablation = criterion(logits, labels)

    for layer in range(classifier.transformer.cfg.n_layers):
        ablated_pred_list[layer] = {}
        for head in range(classifier.transformer.cfg.n_heads):
            # Use functools.partial to create a temporary hook function with the head number fixed
            temp_hook_fn = functools.partial(ablation_function, head_index_to_ablate=head)
            # Run the model with the ablation hook
            ablated_logits = classifier.run_with_hooks(tokens, fwd_hooks=[(utils.get_act_name("z", layer), temp_hook_fn)])
            # Calculate the loss difference (= negative correct logprobs), only on the last `seq_len` tokens
            # loss = -get_log_probs(ablated_logits.log_softmax(-1), tokens)[:, -(seq_len - 1) :].mean()
            loss = criterion(ablated_logits, labels)
            # Store the result, subtracting the clean loss so that a value of zero means no change in loss
            ablation_scores[layer, head] = loss - loss_no_ablation
            ablated_preds = ablated_logits.argmax(dim=-1)
            ablated_pred_list[layer][head] = ablated_preds

    return ablation_scores, preds, ablated_pred_list

print('patching')
ablation_scores = torch.zeros(12,12, device=device)
base_preds = []
ablated_preds = {layer: {head : [] for head in range(classifier.transformer.cfg.n_heads)} for layer in range(classifier.transformer.cfg.n_layers)}
for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
    input_ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
    tmp_ablation_scores, base_pred, ablated_pred_list = get_ablation_scores(classifier, input_ids, labels, head_zero_ablation_hook)
    base_preds += base_pred.tolist()
    for layer in range(classifier.transformer.cfg.n_layers):
        for head in range(classifier.transformer.cfg.n_heads):
            ablated_preds[layer][head] += ablated_pred_list[layer][head].tolist()
    ablation_scores += tmp_ablation_scores
    if i % 16 == 0 or i == len(val_dataloader) - 1:
        torch.save(ablation_scores / ((i+1) * batch_size), "bert_ablation_scores_jigsaw.pth")
        with open('bert_ablated_preds_jigsaw.json', 'w') as f:
            json.dump(ablated_preds, f)
        with open('bert_base_preds_jigsaw.json', 'w') as f:
            json.dump(base_preds, f)

ablation_scores /= len(dataset)
torch.save(ablation_scores, "bert_ablation_scores_jigsaw.pth")