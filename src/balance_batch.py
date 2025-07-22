"""Copyright 2018- The Hugging Face team. All rights reserved.
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
"""

import torch
from torch.utils.data import Sampler
import math
import torch.distributed as dist


def get_length_grouped_indices(
    lengths, batch_size, indices=None, mega_batch_mult=None, generator=None
):  # CHANGES:
    # - added the argument `indices`
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - sorted by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """
    # Default for mega_batch_mult: 50 or the number to get 4 megabatches, whichever is smaller.
    if mega_batch_mult is None:
        mega_batch_mult = min(len(lengths) // (batch_size * 4), 50)
        # Just in case, for tiny datasets
        if mega_batch_mult == 0:
            mega_batch_mult = 1

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    _indices = torch.randperm(len(lengths), generator=generator)
    if indices is None:
        indices = _indices
    else:
        indices = indices[_indices]
    megabatch_size = mega_batch_mult * batch_size
    megabatches = [
        indices[i : i + megabatch_size].tolist()
        for i in range(0, len(lengths), megabatch_size)
    ]
    megabatches = [
        sorted(megabatch, key=lambda i: lengths[i], reverse=True)
        for megabatch in megabatches
    ]

    # The rest is to get the biggest batch first.
    # Since each megabatch is sorted by descending length, the longest element is the first
    megabatch_maximums = [lengths[megabatch[0]] for megabatch in megabatches]
    max_idx = torch.argmax(torch.tensor(megabatch_maximums)).item()
    # Switch to put the longest element in first position
    megabatches[0][0], megabatches[max_idx][0] = (
        megabatches[max_idx][0],
        megabatches[0][0],
    )

    return [i for megabatch in megabatches for i in megabatch]


class LengthGroupedSampler(Sampler):  # CHANGES:
    # - removed the automatic inference of lengths from the dataset
    # - added call to super
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(self, batch_size: int, lengths: list[int], generator=None):
        super().__init__()
        self.batch_size = batch_size
        self.lengths = lengths
        self.generator = generator

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_length_grouped_indices(
            self.lengths, self.batch_size, generator=self.generator
        )
        return iter(indices)


class DistLengthGroupedSampler(Sampler):
    """
    Distributed version of LengthGroupedSampler.
    Ensures that samples are grouped by length and distributed across multiple processes.
    """

    def __init__(
        self, lengths, batch_size, num_replicas=None, rank=None, generator=None
    ):
        """
        Args:
            lengths (list[int]): List of lengths of the dataset samples.
            batch_size (int): Batch size.
            num_replicas (int, optional): Number of processes participating in distributed training.
            rank (int, optional): Rank of the current process within num_replicas.
            shuffle (bool, optional): Whether to shuffle the data. Default is True.
            seed (int, optional): Random seed for shuffling. Default is 0.
        """
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")

        self.lengths = lengths
        self.batch_size = batch_size
        self.num_replicas = num_replicas or dist.get_world_size()
        self.rank = rank or dist.get_rank()
        self.generator = generator
        self.epoch = 0

        # Calculate the number of samples per replica
        self.num_samples = self.batch_size * int(
            math.ceil(len(self.lengths) / (self.num_replicas * self.batch_size))
        )
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        """
        Returns an iterator over the indices for the current process.
        """
        # Generate indices grouped by length
        indices = get_length_grouped_indices(
            self.lengths, self.batch_size, generator=self.generator
        )
        # Pad indices to make them evenly divisible
        indices += indices[: (self.total_size - len(indices))]

        # Subsample for the current process
        # indices = indices[self.rank : self.total_size : self.num_replicas]
        indices = indices[
            self.rank * self.num_samples : (self.rank + 1) * self.num_samples
        ]
        return iter(indices)

    def __len__(self):
        """
        Returns the number of samples for the current process.
        """
        return self.num_samples


class DistDefaultSampler(Sampler):
    """
    A default distributed sampler that evenly splits the dataset indices across multiple processes.
    """

    def __init__(
        self, dataset_size, num_replicas=None, rank=None, shuffle=True, seed=0
    ):
        """
        Args:
            dataset_size (int): Total size of the dataset.
            num_replicas (int, optional): Number of processes participating in distributed training.
            rank (int, optional): Rank of the current process within num_replicas.
            shuffle (bool, optional): Whether to shuffle the data. Default is True.
            seed (int, optional): Random seed for shuffling. Default is 0.
        """
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")

        self.dataset_size = dataset_size
        self.num_replicas = num_replicas or dist.get_world_size()
        self.rank = rank or dist.get_rank()
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Calculate the number of samples per replica
        self.num_samples = int(math.ceil(self.dataset_size / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch):
        """
        Sets the epoch for deterministic shuffling.
        """
        self.epoch = epoch

    def __iter__(self):
        """
        Returns an iterator over the indices for the current process.
        """
        # Generate a list of all indices
        indices = list(range(self.dataset_size))

        # Shuffle indices if required
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.dataset_size, generator=g).tolist()

        # Pad indices to make them evenly divisible
        indices += indices[: (self.total_size - len(indices))]

        # Subsample for the current process
        indices = indices[
            self.rank * self.num_samples : (self.rank + 1) * self.num_samples
        ]
        return iter(indices)

    def __len__(self):
        """
        Returns the number of samples for the current process.
        """
        return self.num_samples
