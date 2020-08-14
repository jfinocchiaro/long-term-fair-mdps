import itertools
import time

t1 = time.time()
print([(a,b) for (a,b) in itertools.product(range(20), range(20)) if a+b<=20])
print("Time elapsed is "+str(time.time()-t1))

t2 = time.time()
print([(a,b) for a in range(20) for b in range(20-a+1)])
print("Time elapsed is "+str(time.time()-t2))
