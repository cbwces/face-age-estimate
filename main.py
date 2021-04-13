import os
import gc
import sys
import yaml
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error
from image_utils import AgeData, SplitDataset, save_split_to_file, load_split_from_file
from network_utils import NormCost, StnModule, MainModel

f = open(sys.argv[1], 'r')
args = yaml.safe_load(f.read())
f.close()
if not os.path.exists(args['save_dir']):
    os.mkdir(args['save_dir'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
if args['cuda'] == True:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
if args['snap'] == False:
    spliter = SplitDataset(args['data_file'])
    train_split_pair, test_split_pair = spliter.transform(shuffle=True, test_size=args['test_size'], stratify=True)
    save_split_to_file(train_split_pair, test_split_pair, file_path=args['save_dir'])
else:
    train_split_pair, test_split_pair = load_split_from_file(args['save_dir'])
train_set = AgeData(train_split_pair, is_train=True, normal_aug=args['train_augmentation'], num_classes=args['num_classes'], img_size=args['img_size'], crop_info=args['data_file_info'], crop_margin=args['margin'], is_affine=args['affine'])
test_set = AgeData(test_split_pair, is_train=False, normal_aug=args['test_preprocess'], test_time_aug=None, num_classes=args['num_classes'], img_size=args['img_size'], crop_info=args['data_file_info'], crop_margin=args['margin'], is_affine=args['affine'])

train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'])
test_loader = DataLoader(test_set, batch_size=args['batch_size']*2, num_workers=args['num_workers'])
if args['stn'] == True:
    stem_model = MainModel(args['backbone'], args['num_classes'], args['pretrain'])
    model = nn.Sequential(StnModule(img_size=args['img_size']), stem_model)
else:
    model = MainModel(args['backbone'], args['num_classes'], args['pretrain'])
args['pretrain'] = False

loss_cal = NormCost(args['loss'])

model = model.to(DEVICE)
if args['snap'] == True:
    model.load_state_dict(torch.load(os.path.join(args['save_dir'], 'age_model.pth')))
optimation = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=0.00001)
if args['patience'] != -1:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimation, min_lr=args['lr'] * 0.001, patience=args['patience'])

##################################################
# model train and test
##################################################
best_mae = 100.0

if args['snap'] == True:
    model.eval()
    with torch.no_grad():
        correct_label = np.zeros(len(test_set))
        pred_label = np.zeros(len(test_set))
        for i, (X, y) in enumerate(test_loader):
            y = y.to(DEVICE)

            X = X.to(DEVICE).float()
            pred_value = torch.sum(torch.sigmoid(model(X)) > 0.5, dim=1) + 1
            if args['cuda'] == True:
                torch.cuda.empty_cache()

            correct_label[i*args['batch_size']*2:(i+1)*args['batch_size']*2] = y.to('cpu').numpy()
            pred_label[i*args['batch_size']*2:(i+1)*args['batch_size']*2] = pred_value.to('cpu').numpy()
    mae = mean_absolute_error(correct_label, pred_label)
    best_mae = mae
    scheduler.num_bad_epochs = args['num_no_boost']

    print("restart from MAE:", mae)
else:
    args['num_no_boost'] = 0
args['snap'] = True

EPOCH = args['epoch']
for epk in range(EPOCH):

    model.train()
    for i, (X, y) in enumerate(train_loader):
        X = X.to(DEVICE).float()
        y = y.to(DEVICE)
        logit = model(X)
        loss_value = loss_cal(logit, y)
        optimation.zero_grad()
        loss_value.backward()
        optimation.step()
        if args['cuda'] == True:
            torch.cuda.empty_cache()
        gc.collect()
        if i % 200 == 0:
            print(f"Loss in " + str(i) + " iters: " + str(loss_value.to('cpu').detach().numpy()))

    model.eval()
    with torch.no_grad():
        correct_label = np.zeros(len(test_set))
        pred_label = np.zeros(len(test_set))
        for i, (X, y) in enumerate(test_loader):
            y = y.to(DEVICE)
            X = X.to(DEVICE).float()
            pred_value = torch.sum(torch.sigmoid(model(X)) > 0.5, dim=1) + 1
            if args['cuda'] == True:
                torch.cuda.empty_cache()

            correct_label[i*args['batch_size']*2:(i+1)*args['batch_size']*2] = y.to('cpu').numpy()
            pred_label[i*args['batch_size']*2:(i+1)*args['batch_size']*2] = pred_value.to('cpu').numpy()
    mae = mean_absolute_error(correct_label, pred_label)
    scheduler.step(mae)
    args['lr'] = scheduler._last_lr[0]
    args['epoch'] -= 1
    with open(os.path.join(args['save_dir'], 'parameters.yml'), 'w') as f:
        yaml.safe_dump(args, f, default_flow_style=False)
    if mae < best_mae:
        best_mae = mae
        torch.save(model.state_dict(), os.path.join(args['save_dir'], 'age_model.pth'))
        args['num_no_boost'] = 0
    else:
        args['num_no_boost'] += 1
        if args['early_stop'] != -1:
            if args['num_no_boost'] >= args['early_stop']:
                break

    print("current MAE:", mae, "| best MAE:", best_mae)
