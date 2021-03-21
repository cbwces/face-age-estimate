import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import torchvision


class NormCost(object):

    def __init__(self, loss_kw, alpha=0.25, gamma=2):
        if loss_kw in ['normal', 'focal']:
            self.loss_kw = loss_kw
            if loss_kw == 'focal':
                self.alpha = alpha
                self.gamma = gamma
        else:
            raise KeyError

    def __call__(self, logits, levels):
        if self.loss_kw == 'normal':
            val = (-torch.sum((F.logsigmoid(logits)*levels + (F.logsigmoid(logits)-logits)*(1-levels)), dim=1))
            return torch.mean(val)
        else:
            total_sigmoid = torch.sigmoid(logits)
            one_logsigmoid = - torch.log(total_sigmoid+1e-7) * levels
            zero_logsigmoid = - (torch.log(1 - total_sigmoid+1e-7)) * (1 - levels)
            add_logsigmoid = (one_logsigmoid*((1-total_sigmoid)**self.gamma) + zero_logsigmoid*(total_sigmoid**self.gamma)) * self.alpha 
            return torch.mean(torch.sum(add_logsigmoid, dim=1))

class StnModule(nn.Module):

    def __init__(self, img_size):
        super(StnModule, self).__init__()
        self.img_size = img_size
        self.fc = nn.Sequential(
            nn.Linear(in_features=img_size*img_size*3, out_features=1000),
            nn.Dropout(0.5),
            nn.ReLU(True),
            nn.Linear(in_features=1000, out_features=20),
            nn.ReLU(True),
            nn.Linear(in_features=20, out_features=6),
        )
        bias = torch.Tensor([1, 0, 0, 0, 1, 0])

        nn.init.constant_(self.fc[5].weight, 0)
        self.fc[5].bias.data.copy_(bias)

    def forward(self, img):
        batch_size = img.size(0)
        theta = self.fc(img.view(batch_size, -1))
        theta[:, [0, 1, 3, 4]] = F.tanh(theta[:, [0, 1, 3, 4]])
        theta = theta.view(batch_size, 2, 3)

        grid = F.affine_grid(theta, torch.Size((batch_size, 3, self.img_size, self.img_size)))
        img_transform = F.grid_sample(img, grid)

        return img_transform

class MainModel(nn.Module):

    def __init__(self, backbone, num_classes, pretrain=False):
        super(MainModel, self).__init__()
        self.backbone = backbone
        if backbone == 'E':
            if pretrain == True:
                self.model = EfficientNet.from_pretrained('efficientnet-b4')
            else:
                self.model = EfficientNet.from_name('efficientnet-b4')
            self.model._fc = nn.Linear(self.model._fc.in_features, num_classes-1, bias=False)
            self.last_bias = nn.Parameter(torch.zeros(num_classes-1).float())

        elif backbone == 'R':
            if pretrain == True:
                self.model = torchvision.models.resnet101(pretrained=True, num_classes=num_classes-1)
            else:
                self.model = torchvision.models.resnet101(num_classes=num_classes-1)
        else:
            raise KeyError

    def forward(self, x):
        x = self.model(x)
        if self.backbone == 'E':
            x = x + self.last_bias
        return x

