import matplotlib.pyplot as plt
import numpy as np
import torch
import math as m

#A common method to create a tensor is from a numpy array:
an_array = np.array([[1, 2], [3, 4]])
a_tensor = torch.tensor(an_array)  # creates tensor of same size as an_array

print(a_tensor)

#------------------Let us see tensor multiplication in both ways: numpy and torch----------------
another_array = np.array([[5, 6, 7], [8, 9, 0]])  # create 2x3 array
another_tensor = torch.tensor(another_array)  # create another tensor of same size as above array

# numpy array multiplication
prod_array = np.matmul(an_array, another_array)

# torch tensor multiplication
prod_tensor = torch.matmul(a_tensor, another_tensor)

print(f"Numpy array multiplication result: {prod_array}")
print(f"Torch tensor multiplication result: {prod_tensor}")

#-------------Calculating gradients------------------
x = torch.tensor(3.0, requires_grad=True)
print(x)
y = x ** 2 
y.backward() # populates gradient (.grad) attributes of y with respect to all of its independent variables
print(x.grad)  # returns the grad attribute (the gradient) of y with respect to x

#---------------------Lets see now y=x^2 
def y(x):
    return torch.cos(x)
print('check details:')
x = torch.linspace(-5, 5, 100)
plt.plot(x, y(x))

x_start = torch.tensor(-1.0, requires_grad=True)  # tensor with x coordinate of starting point
y_start = y(x_start)  # tensor with y coordinate of starting point

plt.scatter(x_start.item(), y_start.item())  # plot starting point

# we can calculate the derivative of y = x ** 2 evaluated at x_start
y_start.backward()  # populate x_start.grad
slope_start = x_start.grad

# and use this to evaluate the tangent line
tangent_line = slope_start * (x - x_start) + y_start
print(slope_start)
print(x)
print(y(x))

plt.plot(x, tangent_line.detach().numpy())
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.xlim(xmin=-5, xmax=5)
plt.ylim(ymin=-10, ymax=10)
plt.show()