import numpy as np

def add(atuple , btuple):
	(a, adot) = atuple
	(b, bdot) = btuple
	return ( a + b, adot + bdot)

def subtract(atuple , btuple):
	(a, adot) = atuple
	(b, bdot) = btuple
	return (a - b, adot - bdot)

def multiply(atuple , btuple):
	(a, adot) = atuple
	(b, bdot) = btuple
	return (a * b, adot * b + bdot * a)

def divide(atuple , btuple):
	(a, adot) = atuple
	(b, bdot) = btuple
	return (a / b, (adot * b - bdot * a) / (b*b))

def exp(atuple):
	(a, adot) = atuple
	return (np.exp(a), np.exp(a)*adot)

def sin(atuple):
	(a, adot) = atuple
	return (np.sin(a), np.cos(a)*adot)

def myfunc(x1, x2):
	a = divide(x1, x2)
	b = exp(x2)
	return multiply(subtract(add(sin(a), a), b), subtract(a, b))

f1=myfunc((1.5, 1.), (0.5, 0.))
f2=myfunc((1.5, 0.), (0.5, 1.))

print(f1)
print(f2)