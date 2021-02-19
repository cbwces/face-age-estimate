import sys
import yaml
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from torch.utils.data import DataLoader
from image_utils import AgeData
from efficientnet_pytorch import EfficientNet

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

f = open(sys.argv[1], 'r')
args = yaml.safe_load(f.read())
f.close()

if args['cuda'] == True:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

test_line = []
test_label = []
f = open(args['data_file'], 'r')
for line in f.read().strip().split('\n'):
    test_line.append(line)
    test_label.append(-1)
test_pair = (test_line, test_label)
f.close()
test_set = AgeData(test_pair, is_train=False, normal_aug=args['test_preprocess'], test_time_aug=args['test_time_augmentation'], img_size=args['img_size'], num_classes=args['num_classes'], mode=args['tta_mode'], crop_info=args['data_file_info'])

test_loader = DataLoader(test_set, batch_size=1)

if args['stn'] == True:
    stem_model = EfficientNet.from_name('efficientnet-b0')
    stem_model._fc = nn.Linear(stem_model._fc.in_features, args['num_classes']-1)
    model = nn.Sequential(StnModule(img_size=args['img_size']), stem_model)
else:
    model = EfficientNet.from_name('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, args['num_classes']-1)

model = model.to(DEVICE)
model.load_state_dict(torch.load(args['model']))

model.eval()
if args['tta_mode'] >= 2:
    pred_value_chunk = torch.zeros((args['tta_mode'], args['num_classes']-1)).to(DEVICE)
with torch.no_grad():
    for i, (X, y) in enumerate(test_loader):
        y = y.to(DEVICE)
        if args['tta_mode'] < 2:
            X = X.to(DEVICE).float()
            pred_value = torch.sum(torch.sigmoid(model(X)) > 0.5, dim=1) + 1
        else:
            for no_aug_img, aug_X in enumerate(X):
                aug_X = aug_X.to(DEVICE).float()
                pred_value_chunk[no_aug_img] = torch.sigmoid(model(aug_X))
            pred_value = torch.sum((torch.sum(pred_value_chunk, dim=0) / args['tta_mode']) > 0.5, dim=0) + 1
        pred_value = pred_value.to('cpu').numpy()
        if args['cuda'] == True:
            torch.cuda.empty_cache()
        print("img: " + test_set.img_paths[i])
        print("age: " + str(pred_value))
