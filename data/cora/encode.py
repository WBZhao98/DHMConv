import collections
import numpy as np
import pandas as pd

edges = pd.read_csv('cora.cites', sep='\t', header=None)
encodeset = set(edges[1]) | set(edges[0])
num = len(encodeset)
a = list(range(num))
b = list(encodeset)
c = zip(b, a)
map = dict(c)
print(map)
print(num)

# citeseer.txt=pd.read_csv('citeseer.txt.txt',sep='\t',header=None)
# for i,j in zip(citeseer.txt[0],citeseer.txt[1]):
#     x=map[i]
#     y=map[j]
