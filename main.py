import mdps
import numpy as np

g = 0.999999
num_s = 2
num_a = 2

rel_Q = np.random.random_sample([num_s, num_a]) 
Pi = rel_Q.argmax(axis=1)
Q = rel_Q/(1-g)
T,R = mdps.random_mdp(num_s,num_a, Q, g)

Pi_R = R.argmax(axis=1)

# print("rel_Q=",rel_Q)
# print("Q=",Q)
# print("R=",R)
print("Pi - Pi_R = ", Pi - Pi_R)
