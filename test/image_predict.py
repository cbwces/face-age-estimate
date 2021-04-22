import os
import sys
import yaml
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from torch.utils.data import DataLoader
sys.path.append('/home/cbw233/python/age_estimate/')
from image_utils import AgeData
from network_utils import StnModule, MainModel

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
if args['label'] == False:
    for line in f.read().strip().split('\n'):
        if len(line.split(" ")) > 1:
            test_line.append(" ".join(line.split(" ")))
        else:
            test_line.append(line)
        test_label.append(-1)
else:
    for line in f.read().strip().split('\n'):
        if len(line.split(" ")) > 2:
            test_line.append(" ".join(line.split(" ")[:-1]))
        else:
            test_line.append(line.split(" ")[0])
        test_label.append(int(line.split(" ")[-1]))
test_pair = (test_line, test_label)
f.close()
test_set = AgeData(test_pair, is_train=False, normal_aug=args['test_preprocess'], test_time_aug=args['test_time_augmentation'], img_size=args['img_size'], num_classes=args['num_classes'], mode=args['tta_mode'], crop_info=args['data_file_info'], crop_margin=args['margin'], is_affine=args['affine'])

test_loader = DataLoader(test_set, batch_size=args['batch_size'])

if args['stn'] == True:
    stem_model = MainModel(args['backbone'], args['num_classes'])
    model = nn.Sequential(StnModule(img_size=args['img_size']), stem_model)
else:
    model = MainModel(args['backbone'], args['num_classes'])

model = model.to(DEVICE)
model.load_state_dict(torch.load(os.path.join(args['save_dir'], 'age_model.pth')))

model.eval()
if args['tta_mode'] >= 2:
    pred_value_chunk = torch.zeros((args['tta_mode'], args['num_classes']-1)).to(DEVICE)

mae = 0
with torch.no_grad():
    for i, (X, y) in enumerate(test_loader):
        y = y.numpy()
        if args['tta_mode'] < 2:
            X = X.to(DEVICE).float()
            pred_value = torch.sum(torch.sigmoid(model(X)) > 0.5, dim=1) + 1
            pred_value = pred_value.to('cpu').numpy()
        else:
            for no_aug_img, aug_X in enumerate(X):
                aug_X = aug_X.to(DEVICE).float()
                pred_value_chunk[no_aug_img] = torch.sigmoid(model(aug_X))
            pred_value = torch.sum((torch.sum(pred_value_chunk, dim=0) / args['tta_mode']) > 0.5, dim=0) + 1
            pred_value = pred_value.to('cpu').numpy()
        mae += np.sum(np.abs(y - pred_value))
        if args['cuda'] == True:
            torch.cuda.empty_cache()
    print("MAE: ", mae / len(test_pair[1]))
