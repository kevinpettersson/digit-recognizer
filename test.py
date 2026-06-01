import torch
import numpy as np

tensor = torch.ones(4, 4)
tensor[:2:3] = 0
print(tensor)