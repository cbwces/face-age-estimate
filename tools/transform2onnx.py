import yaml
import argparse
import torch
from torch import nn as nn

from network_utils import StnModule, MainModel

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_classes', type=int, help='number of classes of output')
parser.add_argument('-c', '--cuda', type=bool, default=True, help='if model with CUDA')
parser.add_argument('-s', '--stn', type=bool, default=False, help='if model using stn module')
parser.add_argument('-b', '--backbone', type=str, help='model backbone')
parser.add_argument('-I', '--input', type=str, help='input model parameter dir')
parser.add_argument('-O', '--output', type=str, help='output onnx model dir')
parser.add_argument('-i', '--img_size', type=int, help='input image size')
args = parser.parse_args()

if args.stn == True:
    stem_model = MainModel(args.backbone, args.num_classes, static=True)
    model = nn.Sequential(StnModule(img_size=args.img_size), stem_model)
else:
    model = MainModel(args.backbone, args.num_classes, static=True)
model.load_state_dict(torch.load(args.input))

toy_data = torch.randn(1, 3, args.img_size, args.img_size)
if args.cuda == True:
    model = model.to('cuda')
    toy_data = toy_data.to('cuda')
model.eval()
with torch.no_grad():
    out = model(toy_data)

torch.onnx.export(model, toy_data, args.output, input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_szie'}, 'output': {0: 'batch_size'}})
