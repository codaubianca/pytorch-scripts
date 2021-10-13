# torch.autograd == automatic differentiaton engine for neural network training

#A training step
import torch, torchvision
model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

prediction = model(data) # forward pass

loss = (prediction - labels).sum()
loss.backward() # backward pass --> torch.autograd calculates and stores
                #                   gradients for each model parameter in .grad

optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step() #gradient descent --> optimizer adjusts each parameter by its .grad

#TODO read more on how differentiation is used in autograd + how it records data (DAG)