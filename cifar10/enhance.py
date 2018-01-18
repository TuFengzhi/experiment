import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from models import *

import os
import filters

class Enhanced(nn.Module):
    def __init__(self):
        super(Enhanced, self).__init__()
        self.filter_num = 2

        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        
        self.original_model = GoogLeNet()
        self.original_model = checkpoint['net']
        self.fc1 = nn.Linear(self.filter_num * 10, 10)

    def forward(self, x):
        x0 = Variable(filters.filter1(x), requires_grad = False)
        x1 = Variable(filters.filter2(x), requires_grad = False)
        y0 = Variable(self.original_model(x0), requires_grad = False)
        y1 = Variable(self.original_model(x1), requires_grad = False)

        out = torch.cat([y1, y2], 1)
        out = self.fc1(out)

        return out

