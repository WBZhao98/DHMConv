test = []
train = []
import random
with open('cora.txt') as f:
    for line in f:
        #items= line.strip().split()
        if random.random()>0.2:
            train.append(line)
        else:
            test.append(line)

with open('split8/test.txt', 'w') as f:
    f.writelines(test)
with open('split8/train.txt', 'w') as f:
    f.writelines(train)