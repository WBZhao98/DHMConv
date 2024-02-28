test = []
train = []
import random
with open('../amazon0302/amazon0302.txt') as f:
    for line in f:
        #items= line.strip().split()
        if random.random()>0.2:
            train.append(line)
        else:
            test.append(line)

with open('../amazon0302/test.txt', 'w') as f:
    f.writelines(test)
with open('../amazon0302/train.txt', 'w') as f:
    f.writelines(train)