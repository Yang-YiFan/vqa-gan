import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import platform, os

a = torch.Tensor([1,2,3])

print(platform.linux_distribution())
print(platform.system())
print(platform.release())
print(os.name)
print('hello tensor!')
print(a)
