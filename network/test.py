import torch
import torch.nn as nn
from torch.nn import functional as F

gt = torch.zeros(10)
# gt[3]=1
pred = torch.rand(10)
print(pred)
print(gt)
loss = F.binary_cross_entropy_with_logits(pred, gt)
loss2 = torch.zeros(10)
for i in range(len(gt)):
    loss2[i] = -gt[i] * torch.log(torch.sigmoid(pred[i])) - (1 - gt[i]) * torch.log(1 - torch.sigmoid(pred[i]))
max_val = (-pred).clamp(min=0)
loss3 = pred - pred * gt + max_val + ((-max_val).exp() + (-pred - max_val).exp()).log()
print('bce loss', loss)
print('cal loss1', loss2.mean(), loss2.sum())
print('cal loss2', loss3.mean(), loss3.sum())
