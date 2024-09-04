import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import loadtxt
#import networkx as nx
from numba import jit, cuda
import time
#import cupyx.scipy.sparse
#import scipy
import cupy as cp
#from cupy import random
from cupy import cuda as cua
import nvidia_smi
import psutil
import sys
import gc

nvidia_smi.nvmlInit()
handle0 = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
#handle1 = nvidia_smi.nvmlDeviceGetHandleByIndex(1)
# card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

#phi_spd
data = loadtxt('phi_signal.csv', delimiter=',')
data = cp.asarray(data)
data=data.astype(cp.float32)
#plt.plot(np.arange(0,10001), data)
#plt.savefig('input_sig.png', dpi=400, bbox_inches='tight')

flux_spike = loadtxt('spike.csv')
flux_spike=cp.float32(flux_spike)
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
s_th = cp.float32(0.7) #signal threshold for somas

#time and size parameters
t=2000
k=35#35770 #same as n
neuron_size = 7
#gc.set_threshold(450, 50, 50)
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
def spike_check(signal_vector, somas,spike_check_arr, spike_counter):
    """
    Iterates through all the soma's to check if their signal is above threshold
    """
    start=cuda.grid(1)
    stride=cuda.gridsize(1)
    for j in range(start,int(k/neuron_size),stride):
        if signal_vector[somas[j]]>=s_th:
            signal_vector[somas[j]]=0
            spike_check_arr[somas[j]]=1
            spike_counter[somas[j]]+=1
        

@cuda.jit
def spike_time(s_array, flux_vector, t_spike,spike_check_arr, network_adj):
    """
    Add flux spikes recieved from other neurons
    """
    start=cuda.grid(1)
    stride=cuda.gridsize(1)
    for j in range(start,s_array.size, stride):
        x = s_array[j]
        val = t_spike[x]
        if  val<=402:
            #flux_vector[(x+3)%(k-1)] +=flux_spike[val]
            flux_vector[network_adj[x]] += flux_spike[val]
            flux_vector[(network_adj[x]+1)%k] += flux_spike[val]  
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

mini = [[0, 0, 0.5, 0,  0,   0,   0],
        [0, 0, 0.4, 0,  0,   0,   0],
        [0, 0, 0,    0,  0,   0,   0.5],
        [0, 0, 0,    0,  0,   0.5,0],
        [0, 0, 0,    0,  0,   0.4,0],
        [0, 0, 0,    0,  0,   0,   0.4],
        [0, 0, 0,    0,  0,   0,   0,]]
mini = cp.asarray(mini, dtype=cp.float32)

def generate_adj_matrix(mini):
    adj_matrix = cp.zeros((k,k))
    for i in range(0,k,neuron_size):
        adj_matrix[i:i+neuron_size, i:i+neuron_size]=mini
    adj_matrix=adj_matrix.astype(cp.float32)

    return adj_matrix

adj_matrix = generate_adj_matrix(mini)

som = cp.zeros(int(k/neuron_size))
som[0] = neuron_size-1
for i in range(1,int(k/neuron_size)):
    som[i]=(som[i-1]+neuron_size)
som=som.astype(cp.int32)

print('somas',som)
network_adj = cp.zeros(k,dtype=cp.int32)

for soma in som:
    network_adj[soma]=random.randint(0,k)#((soma+neuron_size)%(k))-1

#network_adj[6]=20
#network_adj[13]=20#20
#network_adj[20]=6
print('net',network_adj)

#mempool = cp.get_default_memory_pool()
#pinned_mempool = cp.get_default_pinned_memory_pool()

def neuron_step(t,n, data, flux_offset=0):
    '''
    Iterates through time and updates flux and signal using the equation (signal_vector@weight_matrix) + leaf_nodes*data[i%10000]
    and signal is updated using the update equation (4) from phenom paper
    '''

    spike_check_arr = cp.zeros(n)
    spike_check_arr=spike_check_arr.astype(cp.int32)
    t_spike = cp.zeros(n)
    t_spike = t_spike.astype(cp.int32)
    somas = som

    start_gpu = cp.cuda.Event()
    end_gpu= cp.cuda.Event() 
    plot_signals = cp.zeros((t,n), dtype=cp.float32)
    plot_fluxes = cp.zeros((t,n), dtype=cp.float32)

    #weight_matrix = ((cp.random.rand(n,n, dtype=cp.float32))) / (n * 0.5/0.72)
    weight_matrix = adj_matrix

    #leaf_nodes = ((cp.random.rand(n, dtype=cp.float32))-0.95) *(0.5/0.72)
    leaf_nodes = cp.zeros(k,dtype='float32')
    leaf_nodes[0:k:neuron_size]= 0.5
    leaf_nodes[1:k:neuron_size]= 0.5
    leaf_nodes[3:k:neuron_size]= 0.6
    leaf_nodes[4:k:neuron_size]= 0.5

    leaf_nodes[somas] = 0
    leaf_nodes = leaf_nodes.astype(cp.float32)

    signal_vector = leaf_nodes*data[0]
    signal_vector = signal_vector.astype(cp.float32)
    r_fq = cp.zeros(n, dtype=cp.float32)
    
    gpu_array=cp.zeros(t,dtype=cp.float32)
    spike_counter=cp.zeros(n,dtype=cp.float32)
    #print('cpu pre loop',psutil.cpu_percent())

    #psutil.virtual_memory().available *100/psutil.virtual_memory().total
    #psutil.cpu_percent()
    #psutil.virtual_memory().percent

    #mem_array=cp.zeros(t)
    #sum1=0
    #sum0=0
    #sum2=0
    #print('used bytes',mempool.used_bytes())
    #print('total bytes',mempool.total_bytes())
    #print('cpu mem?',pinned_mempool.n_free_blocks())
    for i in range(t):
        #print(f"Timestep = {i}", end="\r") 
        
        signal_vector=plot_signals[i-1]
            
        flux_vector=(cp.matmul(signal_vector,weight_matrix))+(leaf_nodes * data[i%10000]) + flux_offset
        #print('cpu in loop1',psutil.cpu_percent())
        #gpu_array[i]= psutil.cpu_percent()
       
        if(cp.max(spike_check_arr)==1):
            s_array = cp.where(spike_check_arr==1)[0]
            spike_time[256,256](s_array, flux_vector, t_spike, spike_check_arr, network_adj)
        
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
        
        #start_gpu.record()
        r_fq[:]=0

        s_of_phi[512,1024](flux_vector, signal_vector,n, r_fq)
        #print('cpu s_of_phi',psutil.cpu_percent())

        #end_gpu.record()
        #end_gpu.synchronize()
        #t_gpu1 = cua.get_elapsed_time(start_gpu, end_gpu)
        #gpu_array[i]=t_gpu1
        #print(t_gpu1)
        signal_vector = signal_vector*(1 - d_tau*alpha/beta) + (d_tau/beta )*r_fq
        #print('cpu in loop s',psutil.cpu_percent())
        #print(i)
        if (cp.max(signal_vector[somas])>=s_th):
            spike_check[512,1024](signal_vector, somas, spike_check_arr, spike_counter)
        
        plot_signals[i] = signal_vector
        plot_fluxes[i] = flux_vector
        #print('gc count', gc.get_count())
        gpu_array[i]=psutil.cpu_percent()#gc.get_count()[0]
        #print('flux ref',sys.getrefcount(flux_vector))

        #res0 = nvidia_smi.nvmlDeviceGetUtilizationRates(handle0)
        #gpu_array[i] = res0.gpu
        #mem_array[i] = res0.memory
        #print('used bytes',mempool.used_bytes())
        #print('total bytes',mempool.total_bytes())
        #print('cpu mem?',pinned_mempool.n_free_blocks())
        
        '''
        if i>300 and i<500:
            sum0+=t_gpu1
        if i>500 and i<700:
            sum1+=t_gpu1
        if i>800 and i<1000:
            sum2+=t_gpu1
        '''
    #print('s0',sum0)
    #print('s1',sum1)
    #print('s2',sum2)

    '''    
    print('spk_chk',(spike_check_arr.device))
    print('t_spk',(t_spike.device))
    print('som',(somas.device))
    print('w',(weight_matrix.device))
    print('l',(leaf_nodes.device))
    print('s', (signal_vector.device))
    print('f',(flux_vector.device))
    print('r_fq', (r_fq.device)) 
    print('plot_s', (plot_signals.device)) 
    print('plot_f', (plot_fluxes.device)) 
    '''
    
    return plot_signals, plot_fluxes, weight_matrix, gpu_array, spike_counter


start_gpu1 = cp.cuda.Event()
end_gpu1= cp.cuda.Event() 
num_iter=1
convergence=False

flux_offset=cp.zeros(k,dtype=cp.float32)
expected_spikes=cp.zeros(k,dtype=cp.float32)
ones = cp.ones(k,dtype=cp.float32)

expected_spikes[:]=5
ones[som]=0
averages = cp.zeros(k, dtype=cp.float32)

for iterations in range(num_iter):
#while convergence==False:
    print('cpu pre call',psutil.cpu_percent())

    start_gpu1.record()
    plot_signals,plot_fluxes, weight_matrix, gpu_array, spike_counter = neuron_step(t, k , data, flux_offset)
    print('cpu post call',psutil.cpu_percent())

    end_gpu1.record()
    end_gpu1.synchronize()
    t_gpu = cp.cuda.get_elapsed_time(start_gpu1, end_gpu1)
    print('time',t_gpu/(1000))

    #count=0
    for i in range(6,k,7):
        spike_counter[(i-6):i]=spike_counter[i]
        #count+7
    print(spike_counter)

    error = expected_spikes-spike_counter

    if error.any()==0:
        print('done')
        convergence=True
        break

    for nodes in range((plot_signals[0].size)):
        averages[nodes] = (cp.average(plot_signals[:,nodes]))

    flux_offset += 0.01*(cp.multiply(averages,error))
    #print('offset',flux_offset)

#print('dev',np.std(gpu_array))
print('count',cp.count_nonzero(weight_matrix)/(k**2))
#print(weight_matrix[:,1500])
print('sum',cp.sum(weight_matrix[:,1]))


time_axis = np.arange(t)

fig, axs = plt.subplots(1)
#axs.plot(time_axis, cp.asnumpy(plot_signals)[:,k-1])
#axs.plot(time_axis, cp.asnumpy(plot_fluxes)[:,k-1])
axs.plot(time_axis,cp.asnumpy(gpu_array))
#axs.plot(time_axis,cp.asnumpy(mem_array)/50)
axs.set_ylim(0,100)

plt.savefig('mem_cpu.png', dpi=400, bbox_inches='tight')




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
axs[13].vlines(300,0,1.2,'r')
#axs[13].vlines(400,0,1.2,'r')
axs[13].vlines(500,0,1.2,'r')
#axs[13].vlines(600,0,1.2,'r')
axs[13].vlines(700,0,1.2,'r')
axs[13].vlines(800,0,1.2,'g')
axs[13].vlines(1000,0,1.2,'g')

axs[0].set_title('node ' + str(1))
for i in range(14):
    axs[i].set_ylim(0,1.2)
    axs[i].set_xlim(0,2000)

plt.savefig('flux_sig_spike_plot3.png', dpi=400, bbox_inches='tight')
#'''
