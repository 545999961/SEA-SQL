import sys
import time
from collections import Counter
from typing import Optional, List, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, AutoModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.idefics.modeling_idefics import LLAMA_INPUTS_DOCSTRING, _CONFIG_FOR_DOC
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings
import torch.distributed as dist
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

class PreModel(nn.Module):
    def __init__(self,
                 model: AutoModel = None,
                 ):
        super().__init__()
        self.model = model

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self, **kwargs):
        self.model.enable_input_require_grads(**kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
             v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)