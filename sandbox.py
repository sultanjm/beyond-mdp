import mdps
import algos
import numpy as np

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

g = 0.9999
num_s = 2
num_a = 2

# #rQ = np.random.random_sample([num_a, num_s]) 
# #rQ = np.array([[1,g],[1.5*g,1.5]])
# # rQ = np.array([[0.3/(1-g),0.5/(1-g)],[1/(1-g),0.75/(1-g)]])
# rQ = np.array([[1,-2000],[3.4,4]])
# #T,R = mdps.random_mdp(num_a,num_s)
# T,R,rmin,rmax = mdps.random_mdp(num_a,num_s,rQ,g)
# Q = (rQ - rmin/(1-g))/(rmax-rmin)
# rQsim = algos.tabular_vi(T, (rmax-rmin)*R + rmin, g)
# Qsim = algos.tabular_vi(T, R, g)
# print("|| Q - Qsim || = ", np.linalg.norm(Q-Qsim))
# print("|| rQ - rQsim || = ", np.linalg.norm(rQ-rQsim))


T = np.random.random_sample([2,2,2])
T = T/T.sum(axis=2,keepdims=True)
Pi = np.array([[0.01,0.99],[0.99,0.01]])

d = np.array([0.335, 0.665])
d = mdps.stationary_dist(T,Pi,d)
print(d)