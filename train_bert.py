import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)
from transformer_lens.hook_points import HookPoint

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

bert: HookedTransformer = HookedTransformer.from_pretrained("bert-base-uncased")