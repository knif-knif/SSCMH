import torch.nn as nn
import torch
import torchvision
import torchvision.models as tms
import math

class ResNet(nn.Module):
    def __init__(self, pretrain):
        super(ResNet, self).__init__()
        self.pretrained = tms.resnet50(pretrained=pretrain)
        self.children_list = []
        for n,c in self.pretrained.named_children():
            self.children_list.append(c)
            if n == 'avgpool':
                break

        self.net = nn.Sequential(*self.children_list)
        
    def forward(self,x):
        x = self.net(x)
        x = torch.flatten(x, 1)
        return x

class VGGNet(nn.Module):
    def __init__(self, pretrain):
        super(VGGNet, self).__init__()
        self.vgg = tms.vgg19_bn(pretrained=pretrain)
        self.vgg_features = self.vgg.features
        self.fc_features = nn.Sequential(*list(self.vgg.classifier.children())[:-2])

    def forward(self, x):
        features = self.vgg_features(x).view(x.shape[0], -1)
        features = self.fc_features(features)
        return features

class ImageModule(nn.Module):
    def __init__(self, code_length):
        super(ImageModule, self).__init__()

        self.vgg19 = torchvision.models.vgg19(pretrained=True)
        #pretrained_weights = torch.load('vgg19-dcbb9e9d.pth')
        #self.vgg19.load_state_dict(pretrained_weights)
        self.vgg19.classifier = nn.Sequential(*list(self.vgg19.classifier.children())[:-1])
        
        self.fc_encode = nn.Linear(4096, code_length)
        self.fc_encode.weight.data = torch.randn(code_length, 4096) * 0.01
        self.fc_encode.bias.data = torch.randn(code_length) * 0.01
        self.act = nn.Tanh()
        self.classify = nn.Linear(4096, 21)

    def forward(self, x):
        feat = self.vgg19(x)
        x = self.fc_encode(feat)
        x = self.act(x)
        pred = self.classify(feat)
        return x, pred

class ImgNet(nn.Module):
    def __init__(self, arch='vggnet', bit=16, norm=True, mid_num1=1024*8, mid_num2=1024*8, num_classes=21, hiden_layer=3, pretrain=True):
        super(ImgNet, self).__init__()
        self.module_name = "image_model"
        if arch == 'resnet':
            self.feat_dim = 2048
            self.features = ResNet(pretrain)
        elif arch == 'vggnet':
            self.feat_dim = 4096
            self.features = VGGNet(pretrain)
        mid_num1 = mid_num1 if hiden_layer > 1 else bit
        modules = [nn.Linear(self.feat_dim, mid_num1)]
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
        self.alpha = 1.0
        self.classify = nn.Linear(bit, num_classes)
    def forward(self, x):
        feat = self.features(x)
        out = self.fc(feat)
        code = out.tanh()
        pred = self.classify(code)
        if self.norm:
            norm_x = torch.norm(code, dim=1, keepdim=True)
            code = code / norm_x
        return code, pred
    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)
