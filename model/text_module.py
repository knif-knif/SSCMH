import torch.nn as nn
import torch

class TextModule(nn.Module):
    def __init__(self, text_dim, code_length):
        super(TextModule, self).__init__()
        self.lay1 = nn.Sequential(
            nn.Linear(text_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.lay2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.lay3 = nn.Sequential(
            nn.Linear(1024, code_length),
            nn.BatchNorm1d(code_length),
            nn.Tanh()
        )

        self.classify = nn.Linear(code_length, 21)

    def forward(self, y):
        y = self.lay1(y)
        y = self.lay2(y)
        y = self.lay3(y)
        pred = self.classify(y)
        return y, pred

class TxtNet(nn.Module):
    def __init__(self, text_dim, bit, norm=True, mid_num1=1024*8, mid_num2=1024*8, num_classes=21, hiden_layer=2):
        super(TxtNet, self).__init__()
        self.module_name = 'txt_model'
        mid_num1 = mid_num1 if hiden_layer > 1 else bit
        modules = [nn.Linear(text_dim, mid_num1)]
        if hiden_layer >= 2:
            modules += [nn.ReLU(inplace=True)]
            pre_num = mid_num1
            for i in range(hiden_layer - 2):
                if i == 0:
                    modules += [nn.Linear(mid_num1, mid_num2), nn.ReLU(inplace=True)]
                else:
                    modules += [nn.Linear(mid_num2, mid_num2), nn.ReLU(inplace=True)]
                pre_num = mid_num2
            modules += [nn.Linear(pre_num, bit)]
        self.fc = nn.Sequential(*modules)
        self.norm = norm
        self.classify = nn.Linear(bit, num_classes)

    def forward(self, x):
        out = self.fc(x)
        code = out.tanh()
        pred = self.classify(code)
        if self.norm:
            norm_x = torch.norm(code, dim=1, keepdim=True)
            code = code / norm_x
        return code, pred