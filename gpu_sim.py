import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import loadtxt
import networkx as nx
from numba import jit, cuda
import time
import cupyx.scipy.sparse
#import scipy
import cupy as cp
#from cupy import random
from cupy import cuda as cua
from math_sim import generate_graph

#phi_spd
data = loadtxt('phi_signal.csv', delimiter=',')
data = cp.float16(data)
#plt.plot(np.arange(0,10001), data)
#plt.savefig('input_sig.png', dpi=400, bbox_inches='tight')

flux_spike = loadtxt('spike.csv')
flux_spike=cp.float16(flux_spike)
#time=np.arange(403)
#plt.plot(time,flux_spike)
#plt.savefig('test_plot1.png')

#physical constants
phi_th=0.1675 #flux threshold
d_tau = cp.float32(1e-9/1.2827820602389245e-12)
beta = cp.float32(2000*np.pi)
alpha = cp.float32(0.053733049288045114)
A=cp.float32(1)
B=cp.float32(.466)
ib=cp.float32(1.8)
s_th = cp.float32(0.694) #signal threshold for somas

#time and size parameters
t=2000
k=7000 #same as n
neuron_size = 7

@cuda.jit
def s_of_phi(phi,s,n, r_fq):
    """
    Function to approximate rate array 
    """
    start=cuda.grid(1)
    stride=cuda.gridsize(1)
    for i in range(start,n,stride):
        if phi[i]<phi_th: 
            r_fq[i] = 0
        else:
            r_fq[i] = A*(phi[i]-phi_th)*((B*ib)-s[i])
        
@cuda.jit
def spike_check(signal_vector, somas,spike_check_arr):
    """
    Iterates through all the soma's to check if their signal is above threshold
    """
    start=cuda.grid(1)
    stride=cuda.gridsize(1)
    for j in range(start,int(k/neuron_size),stride):
        if signal_vector[somas[j]]>=s_th:
            signal_vector[somas[j]]=0
            spike_check_arr[somas[j]]=1
        
@cuda.jit
def spike_time(s_array, flux_vector, t_spike,spike_check_arr):
    """
    Add flux spikes recieved from other neurons
    """
    start=cuda.grid(1)
    stride=cuda.gridsize(1)
    for j in range(start,s_array.size, stride):
        x = s_array[j]
        val = t_spike[x]
        if  val<=402:
            flux_vector[(x+3)%(k-1)] +=flux_spike[val]
            t_spike[x] +=1
        else:
            spike_check_arr[x] = 0
            t_spike[x]=0
       
@cuda.jit
def sig_update(signal_vector, d_tau,alpha,beta,r_fq):
    start=cuda.grid(1)
    stride=cuda.gridsize(1)
    for i in range(start, k, stride):
        signal_vector[i] = signal_vector[i]*(1 - d_tau*alpha/beta) +  (d_tau/beta )*r_fq[i]

mini = [[0, 0.6, 0, 0,   0,   0,   0],
        [0, 0,   0, 0.5, 0,   0,   0],
        [0, 0,   0, 0.7, 0,   0,   0],
        [0, 0,   0, 0,   0.7, 0,   0],
        [0, 0,   0, 0,   0,   0.5, 0],
        [0, 0,   0, 0,   0,   0,   0.7],
        [0, 0,   0, 0,   0,   0,   0,]]
mini = cp.asarray(mini, dtype=cp.float16)

def generate_adj_matrix(mini):
    adj_matrix = cp.zeros((k,k), dtype=cp.float16)
    for i in range(0,k,neuron_size):
        adj_matrix[i:i+neuron_size, i:i+neuron_size]=mini
    return adj_matrix

adj_matrix = generate_adj_matrix(mini)

som = cp.zeros(int(k/neuron_size), dtype=int)
som[0] = neuron_size-1
for i in range(1,int(k/neuron_size)):
    som[i]=(som[i-1]+neuron_size)

#neuron_matrix = cp.asarray(generate_graph(k))
#neuron_matrix[k-1][1]=0.5 #don't let them go into soma directly???
#print(neuron_matrix)

def neuron_step(t,n, data):
    '''
    Iterates through time and updates flux and signal using the equation (signal_vector@weight_matrix) + leaf_nodes*data[i%10000]
    and signal is updated using the update equation (4) from phenom paper
    '''
    spike_check_arr = cp.zeros(n, dtype=int)
    t_spike = cp.zeros(n, dtype=int)
    somas = som

    #start_gpu = cp.cuda.Event()
    #end_gpu= cp.cuda.Event() 
    
    plot_signals = cp.zeros((t,n), dtype=cp.float16)
    plot_fluxes = cp.zeros((t,n), dtype=cp.float16)

    #weight_matrix = ((cp.random.rand(n,n, dtype=cp.float32))) / (n * 0.5/0.72)
    weight_matrix = adj_matrix

    #leaf_nodes = ((cp.random.rand(n, dtype=cp.float32))-0.95) *(0.5/0.72)
    leaf_nodes = cp.zeros(k, dtype=cp.float16)
    leaf_nodes[0:k:neuron_size]= 0.7
    leaf_nodes[2:k:neuron_size]= 0.7
    leaf_nodes[somas] = 0

    signal_vector = leaf_nodes*data[0]
    r_fq = cp.zeros(n, dtype=cp.float16)

    #print(weight_matrix)
    #print(leaf_nodes)
    
    for i in range(t):
        #print(f"Timestep = {i}", end="\r") 

        flux_vector=(cp.matmul(signal_vector,weight_matrix))+(leaf_nodes * data[i%10000])
        #start_gpu.record()

        if(cp.max(spike_check_arr)==1):
            s_array = cp.where(spike_check_arr==1)[0]
            spike_time[256,256](s_array, flux_vector, t_spike, spike_check_arr)
            #for x in cp.where(spike_check_arr==1)[0]:
                #if t_spike[x]==i:
                #   spike_check_arr[x]=0
                #   t_spike[x]=0
                #elif t_spike[x]==0:
                #    t_spike[x]=i+403 #only add once
                #else:
                #    flux_vector[(x+1)%k] +=flux_spike[int(t_spike[x])]

                #if  t_spike[x]<=402:
                #    flux_vector[(x+2)%k] +=flux_spike[int(t_spike[x])]
                #    t_spike[x] +=1
                #else:
                #    spike_check_arr[x] = 0
                #    t_spike[x]=0

        #end_gpu.record()
        #end_gpu.synchronize()
        #t_gpu1 = cua.get_elapsed_time(start_gpu, end_gpu)
        #print(t_gpu1)   

        r_fq[:]=0
        #r_fq = cp.zeros(n, dtype=cp.float16)
        s_of_phi[512,1024](flux_vector, signal_vector,n, r_fq)
        signal_vector = signal_vector*(1 - d_tau*alpha/beta) + (d_tau/beta )*r_fq
        #sig_update[512,512](signal_vector, d_tau,alpha,beta,r_fq)
        if (cp.max(signal_vector[somas])>=s_th):
            spike_check[512,1024](signal_vector, somas, spike_check_arr)
       
        plot_signals[i] = signal_vector
        plot_fluxes[i] = flux_vector 
    return plot_signals, plot_fluxes, weight_matrix



'''
mode=''
if(mode=='size'):
    print(mode)
    total_time = time_measure( data,t, mode="size")
    #np.savetxt("time_perf1.csv", total_time, delimiter=",")

    #print(np.shape(np.arange(2,1020,11)))
    plt.plot(np.arange(2,6020,1000), total_time)
    plt.savefig('test_plot4.png', dpi=400, bbox_inches='tight')
    print(total_time)
    #plt.plot(np.arange(2,1020,30), total_time_1, "r")
    
    plt.show()
elif(mode=='length'):
    total_time = time_measure( data,t, mode="length")
    #np.savetxt("time_perf2.csv", total_time, delimiter=",")
    plt.plot(np.arange(0,t,1000), total_time)
    plt.savefig('test_plot3.png', dpi=400, bbox_inches='tight')
    print(total_time)
    #plt.plot(np.arange(0,t,1000), total_time_2, "r")

    #plt.show()
'''


#print('used bytes',mempool.used_bytes())
#print('total bytes',mempool.total_bytes())
#print('cpu mem?',pinned_mempool.n_free_blocks())

start_gpu1 = cp.cuda.Event()
end_gpu1= cp.cuda.Event() 

start_gpu1.record()

plot_signals,plot_fluxes, weight_matrix = neuron_step(t, k , data)

end_gpu1.record()
end_gpu1.synchronize()
t_gpu = cua.get_elapsed_time(start_gpu1, end_gpu1)


print('time',t_gpu/1000)
print('count',cp.count_nonzero(weight_matrix)/(k**2))
#print(weight_matrix[:,1500])
print('sum',cp.sum(weight_matrix[:,1]))

time_axis = np.arange(t)
#plt.plot(time_axis, cp.asnumpy(plot_signals)[:,0], label='fluxes')
#plt.savefig('test_plot4.png', dpi=400, bbox_inches='tight')
#print(cp.asnumpy(plot_fluxes[:,k-1]))

fig, axs = plt.subplots(14)
axs[0].plot(time_axis, cp.asnumpy(plot_signals)[:,0])
axs[1].plot(time_axis, cp.asnumpy(plot_signals)[:,1])
axs[2].plot(time_axis, cp.asnumpy(plot_signals)[:,2])
axs[3].plot(time_axis, cp.asnumpy(plot_signals)[:,3])
axs[4].plot(time_axis, cp.asnumpy(plot_signals)[:,4])
axs[5].plot(time_axis, cp.asnumpy(plot_signals)[:,5])
axs[6].plot(time_axis, cp.asnumpy(plot_signals)[:,6])

axs[7].plot(time_axis, cp.asnumpy(plot_signals)[:,k-7])
axs[8].plot(time_axis, cp.asnumpy(plot_signals)[:,k-6])
axs[9].plot(time_axis, cp.asnumpy(plot_signals)[:,k-5])
axs[10].plot(time_axis, cp.asnumpy(plot_signals)[:,k-4])
axs[11].plot(time_axis, cp.asnumpy(plot_signals)[:,k-3])
axs[12].plot(time_axis, cp.asnumpy(plot_signals)[:,k-2])
axs[13].plot(time_axis, cp.asnumpy(plot_signals)[:,k-1])
'''
'''
axs[0].plot(time_axis, cp.asnumpy(plot_fluxes)[:,0])
axs[1].plot(time_axis, cp.asnumpy(plot_fluxes)[:,1])
axs[2].plot(time_axis, cp.asnumpy(plot_fluxes)[:,2])
axs[3].plot(time_axis, cp.asnumpy(plot_fluxes)[:,3])
axs[4].plot(time_axis, cp.asnumpy(plot_fluxes)[:,4])
axs[5].plot(time_axis, cp.asnumpy(plot_fluxes)[:,5])
axs[6].plot(time_axis, cp.asnumpy(plot_fluxes)[:,6])

axs[7].plot(time_axis, cp.asnumpy(plot_fluxes)[:,k-7])
axs[8].plot(time_axis, cp.asnumpy(plot_fluxes)[:,k-6])
axs[9].plot(time_axis, cp.asnumpy(plot_fluxes)[:,k-5])
axs[10].plot(time_axis, cp.asnumpy(plot_fluxes)[:,k-4])
axs[11].plot(time_axis, cp.asnumpy(plot_fluxes)[:,k-3])
axs[12].plot(time_axis, cp.asnumpy(plot_fluxes)[:,k-2])
axs[13].plot(time_axis, cp.asnumpy(plot_fluxes)[:,k-1])
'''
'''
axs[0].set_title('node ' + str(1))
for i in range(14):
    axs[i].set_ylim(0,1.2)
    axs[i].set_xlim(0,2000)

plt.savefig('flux_sig_spike_plot2.png', dpi=400, bbox_inches='tight')
#'''