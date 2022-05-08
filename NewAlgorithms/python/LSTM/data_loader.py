from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import ujson as json


class MySet(Dataset):
    def __init__(self, filename):
        super(MySet, self).__init__()
        self.content = open(filename).readlines()

        indices = np.arange(len(self.content))
        val_indices = np.random.choice(indices, len(self.content) // 5)

        self.val_indices = set(val_indices.tolist())

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rec = json.loads(self.content[idx])
        if idx in self.val_indices:
            rec['is_train'] = 0
        else:
            rec['is_train'] = 1
        return rec


def collate_fn(recs):
    forward = map(lambda x: x['forward'], recs)

    def to_tensor_dict(recs):
        values = torch.FloatTensor(map(lambda r: r['values'], recs))
        masks = torch.FloatTensor(map(lambda r: r['masks'], recs))
        deltas = torch.FloatTensor(map(lambda r: r['deltas'], recs))
        forwards = torch.FloatTensor(map(lambda r: r['forwards'], recs))
        evals = torch.FloatTensor(map(lambda r: r['evals'], recs))
        eval_masks = torch.FloatTensor(map(lambda r: r['eval_masks'], recs))

        return {
            'values': values,
            'forwards': forwards,
            'masks': masks,
            'deltas': deltas,
            'evals': evals,
            'eval_masks': eval_masks
        }
    
    ret_dict = {
        'forward': to_tensor_dict(forward),
    }

    ret_dict['labels'] = torch.FloatTensor(map(lambda x: x['label'], recs))
    ret_dict['is_train'] = torch.FloatTensor(map(lambda x:x['is_train'], recs))

    return ret_dict

def get_loader(filename, batch_size=64, shuffle=True):
    data_set = MySet(filename)
    data_iter = DataLoader(
                        dataset=data_set,
                        batch_size=batch_size,
                        num_workers=4,
                        shuffle=shuffle,
                        pin_memory=True,
                        collate_fn=collate_fn
                        )
    return data_iter