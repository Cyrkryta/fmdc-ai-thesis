# Import dependencies
import os
import random
from torch.utils.data import Sampler

def get_project_key(dataset, idx):
    """
    Custom function for retrieving a designated project key
    """
    return dataset.project_keys[idx]

class FMRICustomSampler(Sampler):
    """
    Custom sampler to only sample elements from the same projects
    """
    def __init__(self, dataset, batch_size, key_fn):
        self.dataset = dataset
        self.batch_size = batch_size
        self.key_fn = key_fn
        self.groups = {}

        # Group the dataset indices
        for idx in range(len(dataset)):
            key = self.key_fn(dataset, idx)
            if key not in self.groups:
                self.groups[key] = []
            self.groups[key].append(idx)

        # Prepare the batches to only contain the groups
        self.batches = []
        for key, indices in self.groups.items():
            random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                self.batches.append(indices[i : i + batch_size])
        
        # Shuffle the final batches
        random.shuffle(self.batches)

    # Iter
    def __iter__(self):
        for batch in self.batches:
            yield batch

    # Retrieve length
    def __len__(self):
        return len(self.batches)
    pass