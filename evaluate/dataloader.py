import logging
import os
import csv
import pickle
import numpy as np
import random

import torch
import torch.utils.data as util_data
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, DataCollatorForLanguageModeling


class PairSamples(Dataset):
    def __init__(self, train_x1, train_x2, pairsimi):
        assert len(pairsimi) == len(train_x1) == len(train_x2)
        self.train_x1 = train_x1
        self.train_x2 = train_x2
        self.pairsimi = pairsimi

    def __len__(self):
        return len(self.pairsimi)

    def __getitem__(self, idx):
        return {'seq1': self.train_x1[idx], 'seq2': self.train_x2[idx], 'pairsimi': self.pairsimi[idx]}


'''
Assumed data format:
DNA sequence1, DNA sequence2
'''


def pair_loader_csv(args, load_train=True):
    delimiter = ","
    if load_train:
        with open(os.path.join(args.datapath, args.train_dataname)) as csvfile:
            data = list(csv.reader(csvfile, delimiter=delimiter))
    else:
        with open(os.path.join(args.datapath, args.val_dataname)) as csvfile:
            data = list(csv.reader(csvfile, delimiter=delimiter))
    if args.con_method == "same_species":
        seq1 = [d[0] for d in data]
        seq2 = [d[1] for d in data]
    else:
        seq1 = [d[0] for d in data] + [d[1] for d in data]
        seq2 = seq1
    pairsimi = [1 for _ in seq1]

    dataset = PairSamples(seq1, seq2, pairsimi)
    if load_train:
        loader = util_data.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    else:
        loader = util_data.DataLoader(dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=4)
    return loader


class GeneStructureDataset(Dataset):
    def __init__(
            self,
            dest_path: str,
            tokenizer: PreTrainedTokenizerBase,
            dataset_name: str = "8K",
            split: str = "train",
            max_length: int = 512,
            seed: int = 42,
            **kwargs
    ):
        super(GeneStructureDataset, self).__init__()
        np.random.seed(seed)
        random.seed(seed)

        data_path = os.path.join(dest_path, dataset_name, split + ".txt")
        assert os.path.exists(data_path), f"Data path {data_path} does not exist."

        with open(data_path, "r") as f:
            data_tmp = f.read().strip().split("\n")
        data = []
        for path in data_tmp:
            if os.path.exists(path):
                data.append(path)
            else:
                logging.warning(f"File {path} does not exist.")
        # down sample in val dataset
        if "val" in split and len(data) > 1000:
            logging.info(f"Down sampling {len(data)} to 1000, from {split} dataset")
            data = np.random.choice(data, 1000, replace=False).tolist()
        self.sequences = data
        self.max_length = max_length
        self.labels = None
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.mlm = kwargs.get("mlm", False)
        self.mlm_probability = kwargs.get("mlm_probability", 0.15)
        # self.label_name = self.get_label_name
        self._data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=self.mlm,
                                                              mlm_probability=self.mlm_probability)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        with open(self.sequences[idx], "rb") as f:
            data = pickle.load(f)
        label = os.path.basename(self.sequences[idx]).split("_")[-1].split(".")[0]
        # TODO down sample with 0.5%
        assert "seq" in data, f"seq not in data: {data.keys()}"
        seq = data["seq"][:self.max_length]

        return {
            "input_ids": seq,
            'labels': label,
        }


class GeneStructureContrastDataset(GeneStructureDataset):

    def __getitem__(self, idx):
        seq_file1 = self.sequences[idx]
        seq_file2 = self.sequences[np.random.choice(len(self.sequences), 1, replace=False)[0]]
        with open(seq_file1, "rb") as f:
            data1 = pickle.load(f)
            assert "seq" in data1, f"seq not in data: {data1.keys()}"
            seq1 = data1["seq"][:self.max_length]
            # seq1_ids = self.tokenizer(seq1)["input_ids"]
        with open(seq_file2, "rb") as f:
            data2 = pickle.load(f)
            assert "seq" in data2, f"seq not in data: {data2.keys()}"
            seq2 = data2["seq"][:self.max_length]
            # seq2_ids = self.tokenizer(seq2)["input_ids"]
        # TODO down sample with 0.5%
        seq1_type = os.path.basename(seq_file1).split("_")[0]
        seq2_type = os.path.basename(seq_file2).split("_")[0]
        if seq1_type == seq2_type:
            pairsmi = torch.tensor(1, dtype=torch.long)
        else:
            pairsmi = torch.tensor(0, dtype=torch.long)

        return {'seq1': seq1, 'seq2': seq2, 'pairsimi': pairsmi}
