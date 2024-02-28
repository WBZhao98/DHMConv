test = []
train = []
import random
with open('soc-Epinions1.txt') as f:
    for line in f:
        #items= line.strip().split()
        if random.random()>0.2:
            train.append(line)
        else:
            test.append(line)

with open('split0/test.txt', 'w') as f:
    f.writelines(test)
with open('split0/train.txt', 'w') as f:
    f.writelines(train)