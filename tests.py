import numpy as np
from torch.utils.data import Dataset, DataLoader
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


d_test = LamemDataset()
d_test = DataLoader(d_test, batch_size=8, num_workers=4, pin_memory=True)
model = ResMem(pretrained=True).cuda()
model.eval()
with torch.no_grad():
    rloss = 0
    preds = []
    ys = []
    names = []
    t = 1
    for batch in tqdm.tqdm(d_test):
        x, y, name = batch
        ys += y.squeeze().tolist()
        bs, c, h, w = x.size()
        ypred = model.forward(x.cuda()).view(bs, -1).mean(1)
        preds += ypred.squeeze().tolist()
        names += name

    df = pd.DataFrame({'Name': names, 'Y': ys, "Y_Pred": preds})
    rcorr = spearmanr(ys, preds)[0]
    loss = ((np.array(ys) - np.array(preds)) ** 2).mean()
    print('Loss is ', loss)
    print('Rank Correlation is ', rcorr)
    sns.distplot(ys, label='Ground Truth')
    sns.distplot(preds, label='Predictions')
    plt.title(f'prediction distribution on {len(d_test)} samples')
    plt.legend()
    plt.savefig(f'restest.png', dpi=500)
    df.to_csv('Test.csv')
