
import pandas as pd
import collections
import numpy as np
for i in range(10):
    split='split'+str(i)

    edges = pd.read_csv(split+'/test.txt', sep='\t', header=None)
    (edges-1).to_csv(split+'/test.txt',sep='\t',header=None,index=None)
    edges = pd.read_csv(split + '/train.txt', sep='\t', header=None)
    (edges - 1).to_csv(split + '/train.txt', sep='\t', header=None, index=None)

# edges = pd.read_csv('val_pos_edges.txt',sep='\t',header=None)
# (edges-1).to_csv('val.txt',sep='\t',header=None,index=None)
num = max(max(edges[0]),max(edges[1]))
num+=1

edge_dict = collections.defaultdict(list)

for i,j in zip(edges[0],edges[1]):
    edge_dict[i].append(j)

print(edge_dict)

# 采样负样本。与正样本数相同
def sample_neg(edge_dict,num):
    neg_dict = collections.defaultdict(list)
    for u in edge_dict.keys():
        n = len(edge_dict[u])
        neg_items=[]
        while True:
            if len(neg_items) == n:
                break
            neg_id = np.random.randint(low=0, high=num, size=1)[0]

            if neg_id not in edge_dict[u] and neg_id not in neg_items:
                neg_items.append(neg_id)
        neg_dict[u]=neg_items
    print(neg_dict)

sample_neg(edge_dict,num)