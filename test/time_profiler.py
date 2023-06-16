import time_profiler_py
import numpy as np
import time

t = time_profiler_py.TimeProfiler()
t.tic("test1")
time.sleep(0.1)
out = t.toc("test1")
print(out)

t.tic("test2")
time.sleep(0.2)
out = t.toc("test2")
print(out)

t.tic("test3")
time.sleep(0.3)
out = t.toc("test3")
print(out)

t.tic("test1")
time.sleep(0.15)
out = t.toc("test1")
print(out)

print('test complete')
