import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class GenericUCIDataset(Dataset):
    X: np.ndarray
    y: np.ndarray

    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert len(X) == len(y)
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        return torch.tensor(self.X[idx]), self.y[idx]
