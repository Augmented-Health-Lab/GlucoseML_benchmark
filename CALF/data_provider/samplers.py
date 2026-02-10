from torch.utils.data import Sampler
import random

class SubjectBalancedSampler(Sampler):
    def __init__(self, dataset, samples_per_epoch=None):
        self.dataset = dataset
        self.subject_to_indices = dataset.subject_to_indices
        self.subjects = list(self.subject_to_indices.keys())

        if samples_per_epoch is None:
            samples_per_epoch = len(dataset)

        self.samples_per_epoch = samples_per_epoch

    def __iter__(self):
        indices = []

        while len(indices) < self.samples_per_epoch:
            # sample subject uniformly
            sid = random.choice(self.subjects)
            # sample window from that subject
            idx = random.choice(self.subject_to_indices[sid])
            indices.append(idx)

        return iter(indices)

    def __len__(self):
        return self.samples_per_epoch

class SubsampleWindowSampler(Sampler):
    """
    Randomly subsample a fixed number of windows per epoch.
    """
    def __init__(self, dataset, max_windows_per_epoch, seed=0):
        self.dataset = dataset
        self.max_windows = max_windows_per_epoch
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        all_indices = list(range(len(self.dataset)))

        if len(all_indices) <= self.max_windows:
            return iter(all_indices)

        sampled = rng.sample(all_indices, self.max_windows)
        return iter(sampled)

    def __len__(self):
        return min(len(self.dataset), self.max_windows)