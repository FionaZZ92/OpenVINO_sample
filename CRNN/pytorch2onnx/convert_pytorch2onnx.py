import numpy as np
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import models.crnn as crnn

model_path = './crnn.pth'
img_path = './demo.png'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
width=256
height=37
batch=32
channel=1
torch_model = crnn.CRNN(batch, channel, height, width)
torch_model.load_state_dict(torch.load(model_path))

x = torch.randn(batch, channel, height, width, requires_grad=True)
torch_out = torch.onnx._export(torch_model,x,"crnn.onnx",export_params=True)
