test = []
train = []
import random
with open('test.txt') as f:
    for line in f:
        #items= line.strip().split()
        if random.random()>0.25:
            train.append(line)
        else:
            test.append(line)

with open('val.txt', 'w') as f:
    f.writelines(test)
with open('test.txt', 'w') as f:
    f.writelines(train)