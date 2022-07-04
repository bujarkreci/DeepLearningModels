import symengine as sym
import numpy as nu

varsi = sym.symbols('x1 x2') # Define x and y variables

f = sym.sympify(['x2*x1**2', '5*x1 + sin(x2)']) # Define function it works good
#f = sym.sympify(['Exp(x1*x2**2)']) # Define function it works good
#f = sym.sympify(['Pow(E,x1*x2**2)']) # Define function

functions = len(f)
variables = len(varsi)

#size = (functions,variables)

#J = nu.zeros(size, dtype=float)

J = sym.zeros(functions,variables) # Initialise Jacobian matrix


# Fill Jacobian matrix with entries
for i, fi in enumerate(f):
    for j, s in enumerate(varsi):
        J[i,j] = sym.diff(fi, s)

#look the jacobian how it looks
print (J)

#print the matrix
#print (sym.Matrix.det(J))