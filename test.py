from collections import defaultdict, deque
lis = [[0 for i in range(11)] for l in range(7)]
lis[6][1] = 1
for i in lis:
    print(i)

deq = deque()
deq.append((10, 1))
print(deq.pop())