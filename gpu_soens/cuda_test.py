import numpy as np
import random
from numba import vectorize, jit, cuda
import cupy as cp
import cupyx.scipy.sparse
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import time


def s_of_phi(phi,s,n):
    """
    Function to get rate array 
    """
    A=1
    B=.466
    ib=1.8
    phi_th = 0.1675

    r_fq = A*((phi-phi_th)*(((B*ib)-s))) #s vector, and phi 


    indices = np.where(phi<phi_th)[0]
    for i in indices:
        r_fq[i] = 0  #if phi from incoming node is below threshold then it passes on nothing
    return r_fq

@cuda.jit
def add(x, y, out):
        start = cuda.grid(1)
        stride = cuda.gridsize(1)
        for i in range(start, x.shape[0], stride):
                out[i] = x[i] + y[i]

#a = cp.arange(10)
#b = a * 2
#out = cp.zeros_like(a)

#print(out)  # => [0 0 0 0 0 0 0 0 0 0]

#add[2, 32](a, b, out)

#print(out)  # => [ 0  3  6  9 12 15 18 21 24 27]

n=1000
@cuda.jit
def graph(n, out):
    #start = cuda.grid(1)
    #stride = cuda.gridsize(1)
    for i in range(n):
        #for j in range(i):
            #if (cp.all(out[j]==0) == True):
            #value = numba.cuda.random.randint(j+1,n-1) 
            #print(value)
        out[i][i+1] = 0.6

#out = cp.zeros( (n,n))
#print(out)
#graph[32,64](n,out)
#print(out)
#print(cp.all(out[1]==0))
@cuda.jit
def compute_pi(rng_states, k, out):
    thread_id = cuda.grid(1)

    for i in range(k-1):
        x = xoroshiro128p_uniform_float32(rng_states, thread_id)
        out[i] = x-0.4

#2 outgoing per dend???
#nonzero??
#bluh

k=12
#test = cp.zeros((k,k),dtype=cp.float16)
#for i in range(k):
#    for j in range(k):
#            test[j][i]=0.5
test = cp.full((k,k), 0, dtype=cp.float16)

mini = [
    [0,   0.5, 0,   0.5 ],
    [0.5, 0,   0.5, 0 ],
    [0.5, 0,   0,   0.5 ],
    [0,   0,   0,   0 ]
]
#mini = np.array(mini)
mini = cp.asarray(mini)
print(test)

for i in range(0,k,4):
    test[i:i+4, i:i+4]=mini
    test[(i-1)%k][(i+1)%k]=0.3
#test[0:4,0:4]=mini
#test[4:8,4:8]=mini
#test[8:12,8:12]=mini

print(test)

'''
t1 = time.perf_counter()
rng = np.random.default_rng()
S = cupyx.scipy.sparse.random(30000, 30000, density=0.01, dtype=np.float32)#, random_state=0)
#S = (S.toarray())
#S_gpu = cp.asarray(S)
t2 = time.perf_counter()
print(t2-t1)
print(S.toarray())
'''
'''
k=10
threads_per_block = 64
blocks = 32
rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=1)
out = cp.zeros(k)

compute_pi[blocks, threads_per_block](rng_states, 10, out)
out = cp.clip(out,0,1)
#print('pi:', out.mean())
print(out)

@cuda.jit
def gpu_get_leaf(rng_states, k, leaf_nodes):
    thread_id = cuda.grid(1)
    for i in range(k-1):
        x = xoroshiro128p_uniform_float32(rng_states, thread_id)
        leaf_nodes[i] = x-0.2
            

@cuda.jit
def graph(rng_states, k, out):
    thread_id = cuda.grid(1)
    for i in range(k):
        for j in range(k):
            x = xoroshiro128p_uniform_float32(rng_states, thread_id)
            if x>0.5:
                out[i][j] = x
'''



