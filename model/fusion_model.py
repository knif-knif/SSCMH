import torch
import torch.nn as nn


class FusionModule(nn.Module):
    def __init__(self, code_length):
        super(FusionModule, self).__init__()

        self.gconv1 = nn.Linear(2 * code_length, 1024)
        self.BN1 = nn.BatchNorm1d(1024)
        self.act1 = nn.ReLU()

        self.gconv2 = nn.Linear(1024, 1024)
        self.BN2 = nn.BatchNorm1d(1024)
        self.act2 = nn.ReLU()

        self.fc = nn.Linear(1024, code_length)
        self.BN3 = nn.BatchNorm1d(code_length)
        self.act3 = nn.Tanh()

    def forward(self, graph, latent):
        f = torch.mm(graph, latent)
        f = self.gconv1(f)
        f = self.BN1(f)
        f = self.act1(f)
        r = f

        f = torch.mm(graph, f)
        f = self.gconv2(f)
        f = self.BN2(f)
        f += r
        f = self.act2(f)

        f = self.fc(f)
        f = self.BN3(f)
        f = self.act3(f)

        return f