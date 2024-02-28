test = []
train = []
import random
with open('citeseer.txt') as f:
    for line in f:
        #items= line.strip().split()
        if random.random()>0.2:
            train.append(line)
        else:
            test.append(line)
import os
split='split9'
a=os.getcwd()
os.mkdir(a+'/'+split)
with open(split+'/test.txt', 'w') as f:
    f.writelines(test)
with open(split+'/train.txt', 'w') as f:
    f.writelines(train)