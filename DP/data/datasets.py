import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class DiseasePredDataset(Dataset):
    def __init__(self, path="data/disease_pred/Training.csv"):
        df = pd.read_csv(path)
        self.labels = df[df.columns[-1]]
        self.features = df[df.columns[:-1]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        label = self.labels[index]
        features = self.features.iloc[index]
        sample = (features, label)
        return sample


class MnistDataset(Dataset):
    def __init__(self, path="data/mnist/Training.csv"):
        df = pd.read_csv(path)
        self.labels = df[df.columns[-1]]
        self.features = df[df.columns[:-1]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        label = self.labels[index]
        features = self.features.iloc[index]
        sample = (features, label)
        return sample

class Cifar10Dataset(Dataset):
    def __init__(self, path="data/cifar10/Training.csv"):
        df = pd.read_csv(path)
        self.labels = df[df.columns[-1]]
        self.features = df[df.columns[:-1]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        label = self.labels[index]
        features = self.features.iloc[index]
        sample = (features, label)
        return sample