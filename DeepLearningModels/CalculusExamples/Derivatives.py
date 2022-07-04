import sympy as sym
import math as m

# initialize symbol to take derivatives with respect to
x = sym.symbols('x')

r = sym.diff(sym.sin(x)*((2*x**2)+2))
r1 = sym.diff(sym.exp(2*x+5))

print(r,' ',r1)

#partial derivatives
x1, x2 = sym.symbols('x1 x2')

f=x1**2 - x2**3
print(sym.diff(f,x1))
print(sym.diff(f,x2))