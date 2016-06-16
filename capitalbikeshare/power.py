import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


G = np.matrix('1 3 5; 4 2 1; 4 1 0')

v = np.array([1,1,1])/3
v.shape = (3,1)
print("G: {}".format(G))
print("v: {}".format(v))
for i in range(10):
    v = (G*v)/sum(G*v)
    print("v: {}".format(v))

w, vec = np.linalg.eig(G)

print(w)
print(vec/sum(vec))
