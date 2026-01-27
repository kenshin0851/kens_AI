# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 17:18:19 2026

@author: kimke
"""

import torch
import numpy
print("cuda available:{}".format(torch.cuda.is_available()))
device =  'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
a=torch.rand((1, 2),device=device)
print(a.device)
print(a)
gpu2cpu = a.cpu()
print(gpu2cpu)
tensor2 = torch.tensor([1,2,3], dtype=float, device='cuda')
tensor = torch.cuda.FloatTensor([1, 2, 3])
ndarray = tensor.detach().cpu().numpy()
ndarray2 = tensor2.detach().cpu().numpy()
print(tensor2)
print(tensor)
a=torch.tensor(([1,2],[3,4]),dtype=float, device="cuda")
print(a)
b=torch.tensor(([9,8],[7,6]),dtype=float, device="cuda")
print(b)
#@:내적 **:제곱 %:나머지 