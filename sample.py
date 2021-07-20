import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.dataset import random_split
import torch
from scipy.stats import spearmanr
from resmem import ResMem, transformer
from matplotlib import pyplot as plt
import seaborn as sns
from torchvision import transforms
import pandas as pd
from PIL import Image
import tqdm
from csv import reader
from torch import nn
from torchvision.transforms.transforms import CenterCrop
import glob

ORDINAL = 1
class MemCatDataset(Dataset):
    def __init__(self, loc='./Sources/memcat/', transform=transformer):
        self.loc = loc
        self.transform = transform
        with open(f'{loc}data/memcat_image_data.csv', 'r') as file:
            r = reader(file)
            next(r)
            data = [d for d in r]
            self.memcat_frame = np.array(data)

    def __len__(self):
        return len(self.memcat_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.memcat_frame[idx, 1]
        cat = self.memcat_frame[idx, 2]
        scat = self.memcat_frame[idx, 3]
        img = Image.open(f'{self.loc}images/{cat}/{scat}/{img_name}').convert('RGB')
        y = self.memcat_frame[idx, 12]
        y = torch.Tensor([float(y)])
        image_x = self.transform(img)
        return [image_x, y, img_name]


class LamemDataset(Dataset):
    def __init__(self, loc='./Sources/lamem/', transform=transformer):
        self.lamem_frame = np.array(np.loadtxt(f'{loc}splits/full.txt', delimiter=' ', dtype=str))
        self.loc = loc
        self.transform = transform

    def __len__(self):
        return self.lamem_frame.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.lamem_frame[idx, 0]
        image = Image.open(f'{self.loc}/images/{img_name}')
        image = image.convert('RGB')
        y = self.lamem_frame[idx, 1]
        y = torch.Tensor([float(y)])
        image_x = self.transform(image)
        return [image_x, y, img_name]


dt = ConcatDataset((LamemDataset(), MemCatDataset()))
_, d_test = random_split(dt, [63741, 5000])
d_test = DataLoader(d_test, batch_size=32, num_workers=4, pin_memory=True)
model = ResMem(pretrained=True).cuda(ORDINAL)

distvis='ResMem with Feature Retraining'
model.eval()
if len(d_test):
    model.eval()
    # If you're using a seperate database for testing, and you aren't just splitting stuff out
    with torch.no_grad():
        rloss = 0
        preds = []
        ys = []
        names = []
        t = 1
        for batch in d_test:
            x, y, name = batch
            ys += y.squeeze().tolist()
            bs, c, h, w = x.size()
            ypred = model.forward(x.cuda(ORDINAL).view(-1, c, h, w)).view(bs, -1).mean(1)
            preds += ypred.squeeze().tolist()
            names += name
        rcorr = spearmanr(ys, preds)[0]
        loss = ((np.array(ys) - np.array(preds)) ** 2).mean()
        if distvis:
            sns.distplot(ys, label='Ground Truth')
            sns.distplot(preds, label='Predictions')
            plt.title(f'{distvis} prediction distribution on {len(d_test)*32} samples')
            plt.legend()
            plt.savefig(f'{distvis}.png', dpi=500)
