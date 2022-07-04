import numpy as np

a = np.array([[2,3,4],[2,1],[3,2,1,0]])

for i, seq in enumerate(a):
    for j in seq:
        print('i = ', i)
        print('j=', j)
        #print('sequence= ', seq)