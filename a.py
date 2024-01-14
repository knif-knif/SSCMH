from cmdataset import get_trans_data, get_data
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as tms
a = [[1, 2, 3], [1, 2, 3]]
a = torch.tensor(a)
b = [a, a, a]
b = torch.cat(b)
c = [b]
x = 10 * len(c)
print(torch.cat(x))