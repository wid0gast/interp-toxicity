import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,6,7"
import textattack
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd

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
import json
import matplotlib.pyplot as plt
import seaborn as sns

print("Available CUDA devices:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.load_state_dict(torch.load('bert_classifier_vanilla/model_epoch_1_acc_0.9372.pt', map_location=device))
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

df = pd.read_csv('jigsaw/test.csv')
df.comment_text = df.comment_text.str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
df.comment_text = df.comment_text.str.strip()
df = df[df.comment_text != ''].reset_index(drop=True)
drop_indices = []
for i in trange(len(df)):
    if len(df.comment_text) < 10:
        drop_indices.append(i)
df = df.drop(drop_indices).reset_index(drop=True)
jigsaw_data = list(zip(df['comment_text'], df['toxic']))
dataset = textattack.datasets.dataset.Dataset(jigsaw_data)

attacks = {
    "deepword": textattack.attack_recipes.deepwordbug_gao_2018.DeepWordBugGao2018.build(model_wrapper),
    "input_reduction": textattack.attack_recipes.input_reduction_feng_2018.InputReductionFeng2018.build(model_wrapper),
    "pwws": textattack.attack_recipes.pwws_ren_2019.PWWSRen2019.build(model_wrapper),
    "pruthi": textattack.attack_recipes.pruthi_2019.Pruthi2019.build(model_wrapper)
}

for attack_name, attack in attacks.items():
    print(attack_name)
    try:
        attack_args = textattack.AttackArgs(num_examples=50000, log_to_csv=f"{attack_name}_log.csv", disable_stdout=True, parallel=True, shuffle=True)
        attacker = textattack.Attacker(attack, dataset, attack_args)
        attacker.attack_dataset()
    except:
        print(f"Attack {attack_name} Failed!!")
        