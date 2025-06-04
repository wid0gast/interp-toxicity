import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
import textattack
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
from tqdm import tqdm, trange
from multiprocessing import freeze_support, set_start_method
import torch.multiprocessing as mp
import argparse
from langdetect import detect
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
# from transformer_lens import (
#     ActivationCache,
#     FactoredMatrix,
#     HookedTransformer,
#     HookedTransformerConfig,
#     HookedEncoderDecoder,
#     HookedEncoder,
#     utils,
# )
# from transformer_lens.hook_points import HookPoint

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

from textattack.attack_recipes.textfooler_jin_2019 import TextFoolerJin2019
from textattack.attack_recipes.bert_attack_li_2020 import BERTAttackLi2020
from textattack.transformations import WordSwapMaskedLM
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.transformations import WordSwapEmbedding
from textattack.search_methods import GreedyWordSwapWIR
from textattack.goal_functions import UntargetedClassification

class CustomTextFooler(TextFoolerJin2019):
    @staticmethod
    def build(model_wrapper):
        transformation = WordSwapEmbedding(
            max_candidates=50
        )
        
        constraints = [
            WordEmbeddingDistance(
                min_cos_sim=0.5,
                cased=False,
                include_unknown_words=True,
                compare_against_original=True
            ),
            PartOfSpeech(
                tagger_type="nltk",
                tagset="universal",
                allow_verb_noun_swap=True,
                compare_against_original=True
            ),
            RepeatModification(),
            StopwordModification()
        ]
        
        goal_function = UntargetedClassification(model_wrapper)
        search_method = GreedyWordSwapWIR(wir_method="delete")
        
        return CustomTextFooler(
            goal_function=goal_function,
            constraints=constraints,
            transformation=transformation,
            search_method=search_method,
        )

class CustomBERTAttack(BERTAttackLi2020):
    @staticmethod
    def build(model_wrapper):
        transformation = WordSwapMaskedLM(
            method="bert-attack",
            max_candidates=48,
            min_confidence=5e-4,
        )
        
        constraints = [
            MaxWordsPerturbed(max_percent=0.4),
            RepeatModification(),
            StopwordModification()
        ]
        
        goal_function = UntargetedClassification(model_wrapper)
        search_method = GreedyWordSwapWIR(wir_method="unk")
        
        return CustomBERTAttack(
            goal_function=goal_function,
            constraints=constraints,
            transformation=transformation,
            search_method=search_method,
        )


def get_attack(model_wrapper, attack_name):
    attacks = {
        "deepword": textattack.attack_recipes.deepwordbug_gao_2018.DeepWordBugGao2018.build(model_wrapper),
        "input_reduction": textattack.attack_recipes.input_reduction_feng_2018.InputReductionFeng2018.build(model_wrapper),
        "pwws": textattack.attack_recipes.pwws_ren_2019.PWWSRen2019.build(model_wrapper),
        "pruthi": textattack.attack_recipes.pruthi_2019.Pruthi2019.build(model_wrapper),
        "textfooler": CustomTextFooler.build(model_wrapper),
        "bert_attack": CustomBERTAttack.build(model_wrapper)
    }
    
    if attack_name not in attacks:
        raise ValueError(f"Unknown attack name: {attack_name}. Available attacks: {list(attacks.keys())}")
    
    return attacks[attack_name]

def main():
    # Parse command line arguments
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parser = argparse.ArgumentParser(description='Run text attack with specified attack method')
    parser.add_argument('--attack_name', type=str, required=True, 
                      help='Name of the attack to use (deepword, input_reduction, pwws, or pruthi)')
    args = parser.parse_args()

    print("Available CUDA devices:", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")

    ## Attack
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
    model.load_state_dict(torch.load('bert_classifier_vanilla/model_epoch_2_acc_0.9236.pt', map_location=device))
    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
    df = pd.read_csv('jigsaw/test.csv')
    df.comment_text = df.comment_text.str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
    df.comment_text = df.comment_text.str.strip()
    drop_indices = []
    for i in trange(len(df)):
        try:
            if detect(df.comment_text[i]) != 'en':
                drop_indices.append(i)
        except:
            drop_indices.append(i)
    df = df.drop(drop_indices).reset_index(drop=True)
    jigsaw_data = list(zip(df['comment_text'], df['toxic']))
    dataset = textattack.datasets.dataset.Dataset(jigsaw_data)

    # Get the specified attack
    attack = get_attack(model_wrapper, args.attack_name)
    attack_args = textattack.AttackArgs(
        num_examples=50000, 
        log_to_csv=f"{args.attack_name}_log.csv", 
        disable_stdout=True, 
        parallel=True,
        checkpoint_dir="checkpoints", 
        checkpoint_interval=1000, 
        shuffle=True,
    )
    attacker = textattack.Attacker(attack, dataset, attack_args)
    attacker.attack_dataset()
    # jigsaw_aug = pd.read_csv(f"{args.attack_name}_log.csv")
    # jigsaw_aug.drop(jigsaw_aug.columns.difference(["perturbed_text", "ground_truth_output"]), axis=1).to_csv(f"{args.attack_name}_jigsaw.csv", index=False)
    # dataset = load_dataset("csv", data_files={f'{args.attack_name}_jigsaw.csv'})
    # print(len(dataset['train']))
    # # dataset = load_dataset('csv', data_files={'test': 'toxigen_alice.csv'})

    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # print(tokenizer.pad_token)

    # # Tokenization function
    # def tokenize_data(example):
    #     return tokenizer(example["perturbed_text"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    # # Apply tokenization
    # dataset = dataset.map(tokenize_data, batched=True)
    # dataset = dataset.rename_column("ground_truth_output", "labels")  # Rename for consistency
    # dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    # class JigsawDataset(Dataset):
    #     def __init__(self, dataset):
    #         self.dataset = dataset

    #     def __len__(self):
    #         return len(self.dataset)

    #     def __getitem__(self, idx):
    #         return {key: self.dataset[idx][key] for key in ["input_ids", "labels"]}

    # # Create PyTorch DataLoaders
    # batch_size = 64
    # val_dataloader = DataLoader(JigsawDataset(dataset['train']), batch_size=batch_size, shuffle=False)
    # class BERTClassifier(nn.Module):
    #     def __init__(self, transformer, num_classes=2):
    #         super().__init__()
    #         self.transformer = transformer
    #         self.classifier = nn.Linear(transformer.cfg.d_model, num_classes)  # d_model = 768

    #     def forward(self, input_ids):
    #         _, cache = self.transformer.run_with_cache(input_ids)  # Get cache

    #         # Extract final hidden states from residual stream
    #         hidden_states = cache["resid_post", -1]  # Shape: [batch, seq_len, hidden_dim]

    #         # Use last token's hidden state for classification
    #         logits = self.classifier(hidden_states[:, -1, :])  # Shape: [batch, num_classes]
    #         return logits

    #     def run_with_hooks(self, tokens, fwd_hooks):
    #         with torch.no_grad():
    #             with self.transformer.hooks(fwd_hooks=fwd_hooks):
    #                 _, cache = self.transformer.run_with_cache(tokens)
    #             hidden_states = cache["resid_post", -1] 
    #             logits = self.classifier(hidden_states[:, -1, :])
    #         return logits


    # # Initialize model and optimizer
    # num_classes = 2
    # model = HookedEncoder.from_pretrained("bert-base-uncased", device=device)
    # model.load_state_dict(torch.load("finetuned_bert/jigsaw/transformer.pth", map_location=device))
    # model.to(device)
    # classifier = BERTClassifier(model, num_classes)
    # classifier.load_state_dict(torch.load("finetuned_bert/jigsaw/classifier.pth", map_location=device))
    # classifier.to(device)

    # criterion = nn.CrossEntropyLoss()
    # def get_log_probs(
    #     logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
    # ) -> Float[Tensor, "batch posn-1"]:
    #     logprobs = logits.log_softmax(dim=-1)
    #     # We want to get logprobs[b, s, tokens[b, s+1]], in eindex syntax this looks like:
    #     print(logits.shape, logprobs.shape, tokens.shape)
    #     correct_logprobs = eindex(logprobs, tokens, "b s [b s+1]")
    #     return correct_logprobs

    # def head_zero_ablation_hook(
    #     z: Float[Tensor, "batch seq n_heads d_head"],
    #     hook: HookPoint,
    #     head_index_to_ablate: int,
    # ) -> None:
    #     z[:, :, head_index_to_ablate, :] = 0.0

    # def get_ablation_scores(
    #     classifier: BERTClassifier,
    #     tokens: Int[Tensor, "batch seq"],
    #     labels,
    #     ablation_function: Callable = head_zero_ablation_hook,
    # ) -> Float[Tensor, "n_layers n_heads"]:
    #     """
    #     Returns a tensor of shape (n_layers, n_heads) containing the increase in cross entropy loss from ablating the output
    #     of each head.
    #     """
    #     # Initialize an object to store the ablation scores
    #     ablation_scores = t.zeros((classifier.transformer.cfg.n_layers, classifier.transformer.cfg.n_heads), device=classifier.transformer.cfg.device)
    #     ablated_pred_list = {}
    #     # Calculating loss without any ablation, to act as a baseline
    #     classifier.transformer.reset_hooks()
    #     logits = classifier(tokens)
    #     preds = logits.argmax(dim=-1)
    #     loss_no_ablation = criterion(logits, labels)

    #     for layer in range(classifier.transformer.cfg.n_layers):
    #         ablated_pred_list[layer] = {}
    #         for head in range(classifier.transformer.cfg.n_heads):
    #             # Use functools.partial to create a temporary hook function with the head number fixed
    #             temp_hook_fn = functools.partial(ablation_function, head_index_to_ablate=head)
    #             # Run the model with the ablation hook
    #             ablated_logits = classifier.run_with_hooks(tokens, fwd_hooks=[(utils.get_act_name("z", layer), temp_hook_fn)])
    #             # Calculate the loss difference (= negative correct logprobs), only on the last `seq_len` tokens
    #             # loss = -get_log_probs(ablated_logits.log_softmax(-1), tokens)[:, -(seq_len - 1) :].mean()
    #             loss = criterion(ablated_logits, labels)
    #             # Store the result, subtracting the clean loss so that a value of zero means no change in loss
    #             ablation_scores[layer, head] = loss - loss_no_ablation
    #             ablated_preds = ablated_logits.argmax(dim=-1)
    #             ablated_pred_list[layer][head] = ablated_preds

    #     return ablation_scores, preds, ablated_pred_list

    # print('patching')
    # ablation_scores = torch.zeros(12,12, device=device)
    # base_preds = []
    # ablated_preds = {layer: {head : [] for head in range(classifier.transformer.cfg.n_heads)} for layer in range(classifier.transformer.cfg.n_layers)}
    # for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
    #     input_ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
    #     tmp_ablation_scores, base_pred, ablated_pred_list = get_ablation_scores(classifier, input_ids, labels, head_zero_ablation_hook)
    #     base_preds += base_pred.tolist()
    #     for layer in range(classifier.transformer.cfg.n_layers):
    #         for head in range(classifier.transformer.cfg.n_heads):
    #             ablated_preds[layer][head] += ablated_pred_list[layer][head].tolist()
    #     ablation_scores += tmp_ablation_scores
    #     if i % 16 == 0 or i == len(val_dataloader) - 1:
    #         torch.save(ablation_scores / ((i+1) * batch_size), f"{args.attack_name}/bert_ablation_scores_jigsaw_perturbed.pth")
    #         with open(f"{args.attack_name}/bert_ablated_preds_jigsaw_perturbed.json", 'w') as f:
    #             json.dump(ablated_preds, f)
    #         with open(f"{args.attack_name}/bert_base_preds_jigsaw_perturbed.json", 'w') as f:
    #             json.dump(base_preds, f)

    # ablation_scores /= len(dataset)
    # torch.save(ablation_scores, f"{args.attack_name}/bert_ablation_scores_jigsaw_perturbed.pth")

    # ## Analysis
    # ablation_scores = torch.load(f"{args.attack_name}/bert_ablation_scores_jigsaw_perturbed.pth", map_location=torch.device('cpu'))
    # ablation_scores.shape
    # tensor_np = ablation_scores.cpu().detach().numpy()

    # # Plot the heatmap
    # plt.figure(figsize=(8, 6))  # Adjust figure size
    # sns.heatmap(tensor_np, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

    # # Add labels
    # plt.title("Ablation Score by Head")
    # plt.xlabel("Head")
    # plt.ylabel("Layer")

    # plt.savefig(f"{args.attack_name}/bert_jigsaw_perturbed_ablation_scores_by_head.png")
    # # Show the plot
    # plt.show()
    # base_preds = json.load(open(f"{args.attack_name}/bert_base_preds_jigsaw_perturbed.json"))
    # ablated_preds = json.load(open(f"{args.attack_name}/bert_ablated_preds_jigsaw_perturbed.json"))

    # df = pd.read_csv(f"{args.attack_name}_jigsaw.csv")
    # base_acc = (df.ground_truth_output == base_preds).mean()
    # acc_scores = np.zeros((12,12))
    # for i in range(12):
    #     for j in range(12):
    #         acc_scores[i][j] = (df.ground_truth_output == ablated_preds[str(i)][str(j)]).mean()
    # acc_scores -= base_acc
    # base_acc
    # # Plot the heatmap
    # plt.figure(figsize=(8, 6))  # Adjust figure size
    # sns.heatmap(acc_scores, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

    # # Add labels
    # plt.title("Accuracy Score by Head")
    # plt.xlabel("Head")
    # plt.ylabel("Layer")

    # plt.savefig(f"{args.attack_name}/bert_jigsaw_perturbed_accuracy_scores_by_head.png")
    # # Show the plot
    # plt.show()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    freeze_support()
    main()
