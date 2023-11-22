import torch
from torch.utils.data import Dataset
import os
from PIL import Image


class SaveFeatures:
    """
    This class wraps around a pytorch module and saves the output of the module to a class variable.
    """
    def __init__(self, module: torch.nn.Module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.features = object()

    def hook_fn(self, module, i, output):
        self.features = output.clone().requires_grad_(True).cuda()

    def close(self):
        self.hook.remove()


class DirectoryImageset(Dataset):
    def __init__(self, path: str, transform=None):
        self.path = path
        self.transform = transform
        self.image_names = os.listdir(path)
        self.image_names.sort()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(os.path.join(self.path, self.image_names[idx]))
        if self.transform:
            image = self.transform(image)
        return self.image_names[idx], image


class csvImageset(Dataset):
    def __init__(self, csv_path: str, basename=os.path.abspath("./"), img_col="img", score_col="memorability", transform=None):
        import pandas as pd
        self.csv_path = csv_path
        self.basename = basename
        self.transform = transform
        self.csv = pd.read_csv(csv_path)
        self.img_col = img_col
        self.score_col = score_col

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(os.path.join(self.basename, self.csv[self.img_col][idx]))
        if self.transform:
            image = self.transform(image)
        return image, self.csv[self.score_col][idx]
