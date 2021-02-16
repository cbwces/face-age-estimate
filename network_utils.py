import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.Linear(in_features=img_size*img_size*3, out_features=20),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(in_features=20, out_features=6),
            nn.Tanh(),
        )
        bias = torch.Tensor([1, 0, 0, 0, 1, 0])

        nn.init.constant_(self.fc[3].weight, 0)
        self.fc[3].bias.data.copy_(bias)

    def forward(self, img):
        batch_size = img.size(0)
        theta = self.fc(img.view(batch_size, -1)).view(batch_size, 2, 3)

        grid = F.affine_grid(theta, torch.Size((batch_size, 3, self.img_size, self.img_size)))
        img_transform = F.grid_sample(img, grid)

        return img_transform

