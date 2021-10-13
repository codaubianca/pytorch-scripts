# typical feed-forward neural network for digits classification (LeNet)

# A typical training procedure for a neural network is as follows:

#    Define the neural network that has some learnable parameters (or weights)
#    Iterate over a dataset of inputs
#    Process input through the network
#    Compute the loss (how far is the output from being correct)
#    Propagate gradients back into the network’s parameters
#    Update the weights of the network, typically using a simple update rule: weight -= learning_rate * gradient

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    #backward function is automatically defined using autograd


net = LeNet()
print(net)

#learnable parameters
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

#try net on random input. size must be 32x32!!
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
# torch.nn only support mini batches. if single sample --> create fake batch dim with input.unsqueeze(0)

#backprop
net.zero_grad()
out.backward(torch.randn(1, 10))

#calculate loss
output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss() # mean squared error loss

loss = criterion(output, target)
print(loss)

#backprop loss
net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward() #updates bias gradients of each layer

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# update weights
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# use optimizers like SGD and Adam as update rules instead
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.01) # create your optimizer in the begining

# in training loop do:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update

