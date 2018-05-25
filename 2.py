import torch
import torch.nn.functional as F
loss_fn = torch.nn.BCELoss(reduce=False, size_average=False)
input = torch.randn(1, 1)
target = torch.FloatTensor(1, 1).random_(2)
loss = loss_fn(F.sigmoid(input), target)
print(input)
print(target)
print(loss)