from torch import nn
from torchvision.models import resnet152
from torchvision import transforms
import torch
import torch.nn.functional as f
from pickle import UnpicklingError

transformer = transforms.Compose((
    transforms.Resize((256, 256)),
    transforms.CenterCrop(227),
    transforms.ToTensor()
    )
)


class ResMem(nn.Module):
    def __init__(self, learning_rate=1e-5, momentum=.9, cruise_altitude=384, pretrained=False):
        super().__init__()

        self.features = resnet152(pretrained=True)
        for param in self.features.parameters():
            param.requires_grad = False

        self.conv1 = nn.Conv2d(3, 48, kernel_size=11, stride=4)
        self.pool1 = nn.MaxPool2d(3, 2)
        self.lrn1 = nn.LocalResponseNorm(5)
        self.conv2 = nn.Conv2d(48, 256, kernel_size=5, stride=1, padding=2, groups=2)
        self.pool2 = nn.MaxPool2d(3, 2)
        self.lrn2 = nn.LocalResponseNorm(5)
        self.conv3 = nn.Conv2d(256, cruise_altitude, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2)
        self.conv4 = nn.Conv2d(cruise_altitude, cruise_altitude, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               groups=2)
        self.conv5 = nn.Conv2d(cruise_altitude, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.fc6 = nn.Linear(in_features=9216 + 1000, out_features=4096, bias=True)
        self.drp6 = nn.Dropout(p=0.5, inplace=False)
        self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.drp7 = nn.Dropout(p=0.5, inplace=False)
        self.fc8 = nn.Linear(in_features=4096, out_features=2048, bias=True)
        self.drp8 = nn.Dropout(p=0.5, inplace=False)
        self.fc9 = nn.Linear(in_features=2048, out_features=1024, bias=True)
        self.fc10 = nn.Linear(in_features=1024, out_features=512, bias=True)
        self.fc11 = nn.Linear(in_features=512, out_features=256, bias=True)
        self.fc12 = nn.Linear(in_features=256, out_features=1, bias=True)

        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)

        self.learning_rate = learning_rate
        if pretrained:
            try:
                self.load_state_dict(torch.load('resmem/model.pt'))
            except UnpicklingError:
                raise TypeError("Could not find the model, try running git lfs pull. If you haven't installed git lfs, "
                                "you can here: https://git-lfs.github.com/")

    def forward(self, x):
        cnv = f.relu(self.conv1(x))
        cnv = self.pool1(cnv)
        cnv = self.lrn1(cnv)
        cnv = f.relu(self.conv2(cnv))
        cnv = self.pool2(cnv)
        cnv = self.lrn2(cnv)
        cnv = f.relu(self.conv3(cnv))
        cnv = f.relu(self.conv4(cnv))
        cnv = f.relu(self.conv5(cnv))
        cnv = self.pool5(cnv)
        feat = cnv.view(-1, 9216)
        resfeat = self.features(x)

        catfeat = torch.cat((feat, resfeat), 1)

        hid = f.relu(self.fc6(catfeat))
        hid = self.drp6(hid)
        hid = f.relu(self.fc7(hid))
        hid = self.drp8(hid)
        hid = f.relu(self.fc8(hid))
        hid = f.relu(self.fc9(hid))
        hid = f.relu(self.fc10(hid))
        hid = f.relu(self.fc11(hid))

        pry = torch.sigmoid(self.fc12(hid))

        return pry
