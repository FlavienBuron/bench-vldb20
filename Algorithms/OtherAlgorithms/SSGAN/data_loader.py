import ujson as json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

choose = 0
missing_rate = 50
dataset = "AirQuality"
dimension = 36


class MySet(Dataset):
    def __init__(self):
        super(MySet, self).__init__()
        self.content = open("./json/json").readlines()
        indices = np.arange(len(self.content))
        val_indices = np.random.choice(indices, len(self.content) // 5)
        self.val_indices = set(val_indices.tolist())

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rec = json.loads(self.content[idx])
        if idx in self.val_indices:
            rec["is_train"] = 0
        else:
            rec["is_train"] = 1
        return rec


class MyTrainSet(Dataset):
    def __init__(self, input):
        super(MyTrainSet, self).__init__()
        self.content = open(input).readlines()
        indices = np.arange(len(self.content))
        val_indices = np.random.choice(indices, len(self.content) // 5)
        self.val_indices = set(val_indices.tolist())

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rec = json.loads(self.content[idx])
        if idx in self.val_indices:
            rec["is_train"] = 0
        else:
            rec["is_train"] = 1
        return rec


class MyTestSet(Dataset):
    def __init__(self, input):
        super(MyTestSet, self).__init__()
        self.content = open(
            "./json/" + dataset + "/" + str(missing_rate) + "_test.json"
        ).readlines()
        print(self.content)
        indices = np.arange(len(self.content))
        val_indices = np.random.choice(indices, len(self.content) // 5)
        self.val_indices = set(val_indices.tolist())

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rec = [json.loads(self.content[idx])]
        return rec


def collate_fn(recs):
    forward = map(lambda x: x["forward"], recs)
    backward = map(lambda x: x["backward"], recs)

    def to_tensor_dict(recs):
        values = torch.FloatTensor(list(map(lambda r: r["values"], recs)))
        masks = torch.FloatTensor(list(map(lambda r: r["masks"], recs)))
        deltas = torch.FloatTensor(list(map(lambda r: r["deltas"], recs)))
        forwards = torch.FloatTensor(list(map(lambda r: r["forwards"], recs)))
        evals = torch.FloatTensor(list(map(lambda r: r["evals"], recs)))
        eval_masks = torch.FloatTensor(list(map(lambda r: r["eval_masks"], recs)))
        return {
            # "values": values.permute(0, 2, 1),
            "values": values,
            # "forwards": forwards.permute(0, 2, 1),
            "forwards": forwards,
            # "masks": masks.permute(0, 2, 1),
            "masks": masks,
            # "deltas": deltas.permute(0, 2, 1),
            "deltas": deltas,
            # "evals": evals.permute(0, 2, 1),
            "evals": evals,
            # "eval_masks": eval_masks.permute(0, 2, 1),
            "eval_masks": eval_masks,
        }

    ret_dict = {
        "forward": to_tensor_dict(forward),
        "backward": to_tensor_dict(backward),
    }
    ret_dict["labels"] = torch.FloatTensor(list(map(lambda x: x["label"], recs)))
    ret_dict["is_train"] = torch.FloatTensor(list(map(lambda x: x["is_train"], recs)))
    return ret_dict


def get_loader(batch_size=64, shuffle=True):
    data_set = MySet()
    data_iter = DataLoader(
        dataset=data_set,
        batch_size=batch_size,
        num_workers=1,
        shuffle=shuffle,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return data_iter


def get_train_loader(input, batch_size=32, shuffle=True):
    data_set = MyTrainSet(input)
    data_iter = DataLoader(
        dataset=data_set,
        batch_size=batch_size,
        num_workers=1,
        shuffle=shuffle,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return data_iter


def get_test_loader(batch_size=32, shuffle=False):
    data_set = MyTestSet()
    data_iter = DataLoader(
        dataset=data_set,
        batch_size=batch_size,
        num_workers=1,
        shuffle=shuffle,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return data_iter
