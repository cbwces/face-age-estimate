import gc
import sys
import yaml
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error
from image_utils import AgeData, SplitDataset 
from network_utils import NormCost, StnModule
from efficientnet_pytorch import EfficientNet

##################################################
# Backend setting
##################################################
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

##################################################
# Parameters convert
##################################################
f = open(sys.argv[1], 'r')
args = yaml.safe_load(f.read())
f.close()

if args['cuda'] == True:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

spliter = SplitDataset(args['data_file'])
train_split_pair, test_split_pair = spliter.transform(shuffle=True, test_size=args['test_size'], stratify=True)
train_set = AgeData(train_split_pair, is_train=True, normal_aug=args['train_augmentation'], num_classes=args['num_classes'], img_size=args['img_size'], crop_info=args['data_file_info'], crop_margin=args['margin'], is_affine=args['affine'])
test_set = AgeData(test_split_pair, is_train=False, normal_aug=args['test_preprocess'], test_time_aug=None, num_classes=args['num_classes'], img_size=args['img_size'], crop_info=args['data_file_info'], crop_margin=args['margin'], is_affine=args['affine'])

train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'])
test_loader = DataLoader(test_set, batch_size=args['batch_size']*2, num_workers=args['num_workers'])
if args['stn'] == True:
    if args['pretrain'] == False:
        stem_model = EfficientNet.from_name('efficientnet-b0')
    else:
        stem_model = EfficientNet.from_pretrained('efficientnet-b0')
    stem_model._fc = nn.Linear(stem_model._fc.in_features, args['num_classes']-1)
    model = nn.Sequential(StnModule(img_size=args['img_size']), stem_model)
else:
    if args['pretrain'] == False:
        model = EfficientNet.from_name('efficientnet-b0')
    else:
        model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, args['num_classes']-1)

loss_cal = NormCost(args['loss'])

model = model.to(DEVICE)
if args['snap_model']:
    model.load_state_dict(torch.load(args['snap_model']))
optimation = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=0.99, weight_decay=0.00001)
if args['patience'] != -1:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimation, min_lr=args['lr'] * 0.001, patience=args['patience'])

##################################################
# model train and test
##################################################
best_mae = 100.0
num_no_boost = 0

if args['snap_model']:
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
    scheduler.step(mae)

    print("restart from MAE:", mae)

for epk in range(args['epoch']):

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
    # for i, (X, y_list) in enumerate(test_loader):
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
    if mae < best_mae:
        best_mae = mae
        torch.save(model.state_dict(), './age_model.pth')
        num_no_boost = 0
    else:
        num_no_boost += 1
        if args['early_stop'] != -1:
            if num_no_boost >= args['early_stop']:
                break

    print("current MAE:", mae, "| best MAE:", best_mae)

