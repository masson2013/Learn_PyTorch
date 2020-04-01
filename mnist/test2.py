import torch
from   torch      import nn
from   torch.nn   import functional as F
from   torch      import optim

import torchvision
from   matplotlib import pyplot as pyplot

from   utils      import plot_image, plot_curve, one_hot

# Step1  Load Dataset
batch_size = 1
train_loader = torch.utils.data.DataLoader (
  torchvision.datasets.MNIST  ('mnist_data', train = True, download = True,
                                transform = torchvision.transforms.Compose ([
                                  torchvision.transforms.ToTensor(),
                                  # torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                ])
                              ),
  batch_size = batch_size, shuffle = True
)

test_loader = torch.utils.data.DataLoader (
  torchvision.datasets.MNIST  ('mnist_data/', train = False, download = True,
                                transform = torchvision.transforms.Compose ([
                                  torchvision.transforms.ToTensor(),
                                  # torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                ])
                              ),
  batch_size = batch_size, shuffle = False
)

x = torch.rand(1, 2, 2)
print(x.shape)
print(x)
# x = torch.clamp(x, -0.1, 0.1)
# print(x)
fc1 = torch.nn.Linear(2, 2)
out = fc1(x)
print(out.shape)
print(out)

# class Net(nn.Module):
#   def __init__(self):
#     super(Net, self).__init__()
#     # xw + b
#     self.fc1 = nn.Linear(9, 9) # 256 is a random number
#   def forward(self, x):
#     x = self.fc1(x)
#     return x

# net = Net()

# out = net(x)


# x, y = next(iter(train_loader))
# print(x.shape, y.shape, x.min(), x.max())
# print(x.shape)
# print(x)

# plot_image(x, y, 'Image Sample')

# class Net(nn.Module):

#   def __init__(self):
#     super(Net, self).__init__()

#     # xw + b
#     self.fc1 = nn.Linear(28 * 28, 256) # 256 is a random number

#   def forward(self, x):
#     x = self.fc1(x)
#     return x

# net = Net()

# # x, y = next(iter(train_loader))


# for batch_idx, (x, y) in enumerate(train_loader):
#   print(batch_idx, x.shape)
#   # x = x.view(x.size(0), 28 * 28)
#   # print(batch_idx, x.shape)
#   out = net(x.view(x.size(0), 28 * 28))
#   print(out.shape)
#   break

# print(out)

#     # print(x.shape, y.shape)
#     # x: [b, 1, 28, 28] y: [512]
#     # [b, 1, 28, 28] -> [b, 28 * 28]
#     x = x.view(x.size(0), 28 * 28)
#     # => [b, 10]
#     out = net(x)


# out = Net(x.view(x.size(0), 28 * 28))
# print(out.shape)




# class Net(nn.Module):
#   def __init__(self):
#     super(Net, self).__init__()

#     # xw + b
#     self.fc1 = nn.Linear(28 * 28, 256) # 256 is a random number
#     self.fc2 = nn.Linear(256, 64) # 64 is a random number
#     self.fc3 = nn.Linear(64, 10) # 10 is 0~9

#   def forward(self, x):
#     # x: [b, 1, 28, 28]
#     # h1 = relu(xw+b)
#     x = F.relu(self.fc1(x))
#     # h2 = relu(h1w2+b2)
#     x = F.relu(self.fc2(x))
#     # h3 = h2w3+b3
#     x = self.fc3(x)

#     return x


# net = Net()
# # [w1, b1, w2, b2, w3, b3]
# optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum = 0.9)

# train_loss = []

# for epoch in range(3):
#   for batch_idx, (x, y) in enumerate(train_loader):

#     # print(x.shape, y.shape)
#     # x: [b, 1, 28, 28] y: [512]
#     # [b, 1, 28, 28] -> [b, 28 * 28]
#     x = x.view(x.size(0), 28 * 28)
#     # => [b, 10]
#     out = net(x)
#     # [b, 10]
#     y_onehot = one_hot(y, 10)
#     # loss = mean_square_error(out, y_onehot)
#     loss = F.mse_loss(out, y_onehot)

#     optimizer.zero_grad()
#     loss.backward() # grad
#     # w' = w - lr * grad
#     optimizer.step()

#     train_loss.append(loss.item())

#     if batch_idx % 10 == 0:
#       print(epoch, batch_idx, loss.item())

# total_correct = 0
# for x, y in test_loader:
#   x = x.view(x.size(0), 28 * 28)
#   out = net(x)
#   # out [b, 10] => predict: [b]
#   predict = out.argmax(dim = 1)
#   correct = predict.eq(y).sum().float().item()
#   # print(correct)
#   total_correct += correct

# # print(total_correct)
# total_num = len(test_loader.dataset)
# # print(total_num)
# acc = total_correct / total_num
# print('test_acc:', acc)


# # we get optimal [w1, b1, w2, b2, w3, b3]
# plot_curve(train_loss)

# # Predict Label
# x, y = next(iter(test_loader))
# out = net(x.view(x.size(0), 28 * 28))
# predict = out.argmax(dim = 1)
# plot_image(x, predict, 'Test')