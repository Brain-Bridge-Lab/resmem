import math
import os
import shutil
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset
from resmem.model import ResMem
from resmem.utils import SaveFeatures
from torchvision.transforms import ToTensor, ToPILImage, Normalize, Resize
from PIL import Image, ImageFilter
from tqdm import tqdm



def get_gaussian_kernel(kernel_size=3, sigma=1, channels=3):
    """
    https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/7
    """
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float().cuda()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) /
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    gaussian_kernel.cuda()

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False)
    gaussian_filter.cuda()
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


class Viz:
    """
    This class will be used to generate a maximally-activating image for a given filter.
    """
    def __init__(self, size=67, upscaling_steps=16, upscaling_factor=1.2, device='cuda', branch='alex', layer=0):
        self.size, self.upscaling_steps, self.upscaling_factor = size, upscaling_steps, upscaling_factor
        self.device = device
        self.model = ResMem(pretrained=True).to(device).eval()
        if branch == 'alex':
            self.target = self.model.alexnet_layers[layer]
        elif branch == 'resnet':
            self.target = self.model.resnet_layers[layer]
        else:
            raise ValueError('Branch must be alex or resnet.')

        self.output = None
        self.normer = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def visualize(self, filt, lr=1e-2, opt_steps=36):
        sz = self.size
        img = Image.fromarray(np.uint8(np.random.uniform(150, 180, (sz, sz, 3))))
        activations = SaveFeatures(self.target)
        gaussian_filter = get_gaussian_kernel()
        self.model.zero_grad()
        for outer in tqdm(range(self.upscaling_steps), leave=False):
            img_var = torch.unsqueeze(ToTensor()(img), 0).to(self.device).requires_grad_(True)
            img_var.requires_grad_(True).to(self.device)
            optimizer = torch.optim.Adam([img_var], lr=lr, weight_decay=1e-6)

            pbar = tqdm(range(opt_steps), leave=False)
            for n in pbar:
                optimizer.zero_grad()
                self.model.conv_forward(img_var)
                loss = -activations.features[0, filt].mean() + 0.00*torch.norm(img_var)
                loss.backward()
                pbar.set_description(f'Loss: {loss.item()}')
                optimizer.step()

            sz = int(sz * self.upscaling_factor)
            img = ToPILImage()(img_var.squeeze(0))
            if outer != self.upscaling_steps:
                img = img.resize((sz, sz))
                img = img.filter(ImageFilter.BoxBlur(2))
            self.output = img.copy()
        activations.close()

    def save(self, layer, filt):
        if self.branch == 'alex':
            self.output.save(f'alex/layer_{layer}_filter_{filt}.jpg', 'JPEG')
        else:
            self.output.save(f'resnet/layer_{self.rn_address}-{layer}_filter_{filt}.jpg', 'JPEG')

class ImgActivations:
    """
    This class can be used to extract the top 9 images in a dataset that activate a given filter the most.
    """
    def __init__(self, dset: Dataset, branch='alex', layer=0):
        self.branch = branch
        self.model = ResMem(pretrained=True).cuda().eval()
        self.target = self.model
        self.output = []
        self.dset = dset

        if branch == 'alex':
            self.target = self.model.alexnet_layers[layer]
        elif branch == 'resnet':
            self.target = self.model.resnet_layers[layer]
        else:
            raise ValueError('Branch must be alex or resnet.')


    def calculate(self, filt):
        activations = SaveFeatures(self.target)
        levels = []
        names = []
        for img in self.dset:
            x, y, name = img
            self.model(x.cuda().view(-1, 3, 227, 227))
            levels.append(activations.features[0, filt].mean().detach().cpu())
            names.append(name)

        self.output = np.array(names)[np.argsort(levels)[-9:]].flatten()

    def draw(self, filt):
        if not os.path.exists(f'./figs/{self.rn_address}-{filt}'):
            os.mkdir(f'./figs/{self.rn_address}-{filt}')
        for f in self.output:
            shortfname = f.split('/')[-1]
            shutil.copy2(f, f'./figs/{self.rn_address}-{filt}/'+shortfname)

