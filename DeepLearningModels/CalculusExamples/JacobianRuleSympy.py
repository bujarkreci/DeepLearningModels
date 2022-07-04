import sympy as sp
from sympy import *
import numpy as np
import time as t
x1,x2=sp.symbols('x1,x2', real=True)
print(x1)

J = Function('J')(x1,x2)
f1=(sp.sin(x1/x2)+x1/x2-sp.exp(x2))*((x1/x2)-sp.exp(x2))
#f1=sp.cos(x2)-sp.sin(x1)
f2=x1 - 3*x2*(1-x1**2)

print('Now lets see with the simple method solution 1:')
start = t.time()
f1x1=diff(f1,x1)
f1x2=diff(f1,x2)
f2x1=diff(f2,x1)
f2x2=diff(f2,x2)
J = sp.Matrix([[f1x1,f1x2],[f2x1,f2x2]])
res = J.subs([(x1,0), (x2,0)])
end = t.time()
print(J)
print (res)
print('Execution time of the first method: ', end-start)
print()
#------------Another Method the same result faster----------------------
print('Now lets see with the Jacobian method solution 2:')
start = t.time()
F = sp.Matrix([f1,f2])
print(type(F) )
jacob=F.jacobian([x1,x2])
sub = F.jacobian([x1,x2]).subs([(x1,1.5), (x2,0.5)])
end = t.time()
print(F)
print(jacob)
print(sub)
print('Execution time of the second method: ', end-start)
#J=np.array([[f1x1,f1x2],[f2x1,f2x2]])
#J1=J(0,0)

