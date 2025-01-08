import sys

import math
import os.path
import random
from dataclasses import dataclass
from typing import List, Tuple

import datasets
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import trange
from transformers import DataCollatorWithPadding, DataCollatorForSeq2Seq
from transformers import PreTrainedTokenizer, BatchEncoding

from arguments import DataArguments


class TrainDataset(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer
    ):
        self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train', cache_dir='local_data')

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        inputs = self.dataset[item]['input']
        outputs = self.dataset[item]['output']
        # if 'deepseek' in self.tokenizer.name_or_path:
        #     inputs = inputs.replace('Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n',
        #                             'You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacyssues, and other non-computer science questions, you will refuse to answer')

        tokenized_full_prompt = self.tokenizer(inputs + outputs,
                                               truncation=True,
                                               max_length=self.args.cutoff_len,
                                               padding=False,
                                               return_tensors=None,
                                               add_special_tokens=True)
        tokenized_user_prompt = self.tokenizer(inputs,
                                               truncation=True,
                                               max_length=self.args.cutoff_len,
                                               padding=False,
                                               return_tensors=None,
                                               add_special_tokens=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"]) + 1
        tokenized_full_prompt["labels"] = tokenized_full_prompt["input_ids"].copy()
        tokenized_full_prompt["labels"] = [-100]* user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        # print(tokenized_user_prompt['input_ids'])
        return tokenized_full_prompt


@dataclass
class EmbedCollator(DataCollatorForSeq2Seq):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    cutoff_len: int = 4096

    def __call__(self, features, return_tensors='pt'):
        if return_tensors is None:
            return_tensors = self.return_tensors

        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        inputs = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.cutoff_len,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # print(inputs['input_ids'].shape)

        return inputs