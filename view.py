import torch

a = torch.randn([1, 3, 2])
b = a.view([3, 1, 2])
c = b.view([1, 3, 2])

print(torch.equal(a, b))
print(torch.equal(b, c))
print(torch.equal(a, c))