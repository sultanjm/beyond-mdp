import mdps
import algos
import numpy as np

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

g = 0.9999
num_s = 2
num_a = 2

#rQ = np.random.random_sample([num_a, num_s]) 
#rQ = np.array([[1,g],[1.5*g,1.5]])
# rQ = np.array([[0.3/(1-g),0.5/(1-g)],[1/(1-g),0.75/(1-g)]])
rQ = np.array([[1,2000],[3.4,4]])
#T,R = mdps.random_mdp(num_a,num_s)
T,R,rmin,rmax = mdps.random_mdp(num_a,num_s,rQ,g)
nQ = (rQ - rmin/(1-g))/(rmax-rmin)
Q,_ = algos.tabular_vi(T, R, g)
print("|| Q - nQ || = ", np.linalg.norm(Q-nQ))
# print("R=",R)
# print("T=",T)
print("Value Iter. Q=", Q)


# T = np.array([[[1,0],[1,0]],[[0,1],[0,1]]])
# R = np.array([[[1,0],[0,0]],[[0,0],[0,1]]])

# Q,_ = algos.tabular_vi(T,R,g)
# Qprint(Q)