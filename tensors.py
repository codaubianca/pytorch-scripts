import torch 
import numpy as np

#Initialization
data = [[1, 2], [3, 4]]

x_data = torch.tensor(data)
print(x_data)

x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


#Attributes

tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

#Operations: https://pytorch.org/docs/stable/torch.html#tensors

#Math op with tensors https://pytorch.org/docs/stable/torch.html#math-operations

# mul == * == element-wise multiplication
# matmul = @ == matrix multiplication
# in-place operations have a suffix "_":
print(tensor, "\n")
tensor.add_(5)
print(tensor)

# We can move our tensor to the GPU if available
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
  print(f"Device tensor is stored on: {tensor.device}")

# numpy -> pytorch
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

#numpy <- pytorch
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")


