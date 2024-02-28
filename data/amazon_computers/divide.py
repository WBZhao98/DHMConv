test = []
train = []
import random
with open('amazon_computers.txt') as f:
    for line in f:
        #items= line.strip().split()
        if random.random()>0.2:
            train.append(line)
        else:
            test.append(line)

with open('split9/test.txt', 'w') as f:
    f.writelines(test)
with open('split9/train.txt', 'w') as f:
    f.writelines(train)