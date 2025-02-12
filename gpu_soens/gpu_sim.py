import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import loadtxt
from numba import jit, cuda
import time
import cupy as cp
from cupy import cuda as cua
import nvidia_smi
import psutil
import sys
import gc
import networkx as nx

nvidia_smi.nvmlInit()
handle0 = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
#handle1 = nvidia_smi.nvmlDeviceGetHandleByIndex(1)
# card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

#phi_spd
data = loadtxt('phi_signal.csv', delimiter=',')
# 1000 nanoseconds
#plt.plot(np.arange(0,975), data[1101:2076])
#plt.xlabel("Time (ns)")
#plt.ylabel("Flux")
#plt.savefig('input_spike.png', dpi=400, bbox_inches='tight')
every_10th = data[::10]
#plt.plot(np.arange(every_10th.size),every_10th)
#plt.savefig('input_spike???.png', dpi=400, bbox_inches='tight')

data = cp.asarray(every_10th)
data = data.astype(cp.float32)

#403
flux_spike = (loadtxt('spike.csv', dtype=cp.float32))
flux_spike=flux_spike[1:404]
flux_spike_cupy=cp.array(flux_spike)
flux_spike_cupy = flux_spike_cupy.reshape(402,1)

#physical constants
phi_th=0.1675 #flux threshold
d_tau = cp.float32(1e-9/1.2827820602389245e-12)
beta = cp.float32(2000*cp.pi)
alpha = cp.float32(0.053733049288045114)
A=cp.float32(1)
B=cp.float32(.466)
ib=cp.float32(1.8)
s_th = cp.float32(0.7) #signal threshold for somas

t=10000 #time steps
k= 7000#35770 #same as n, number of neurons
neuron_size = 7
connect = 1
#gc.set_threshold(450, 50, 50)

def s_of_array_phi(phi, s, n, r_fq):
    phi_zero = cp.where(phi<phi_th)[0]
    phi_nonzero = cp.where(phi>=phi_th)[0]
    phi[phi_zero]=0
    try: 
        phi[phi_nonzero]=A*(phi-phi_th)*((B*ib)-s)
    except:
        phi[phi_zero]=0

@cuda.jit
def s_of_phi(phi,s,n, r_fq):
    """
    Function to approximate rate array 
    """
    start=cuda.grid(1)
    stride=cuda.gridsize(1)
    for i in range(start,n,stride):
        if phi[i]<phi_th: 
            r_fq[i] = 0.0
        else:
            r_fq[i] = A*(phi[i]-phi_th)*((B*ib)-s[i])
        
def spike_good_check(signal_vector, somas, spike_check_arr, spike_counter, spike_wait):
    spiking = signal_vector>=s_th
    #spiking =spiking.astype(cp.int32)
    spiking = somas[spiking[somas]]

    signal_vector[spiking] = 0
    spike_check_arr[spiking] = 1
    spike_counter[spiking]+=1
    spike_wait[spiking]=35

@cuda.jit
def spike_check(signal_vector, somas,spike_check_arr, spike_counter, spike_wait):
    """
    Iterates through all the soma's to check if their signal is above threshold
    """
    start=cuda.grid(1)
    stride=cuda.gridsize(1)
    for j in range(start,int(k/neuron_size),stride):
        #sig = signal_vector[somas[j]]
        if signal_vector[somas[j]]>=s_th:
            signal_vector[somas[j]]=0
            spike_check_arr[somas[j]]=1 #change
            spike_counter[somas[j]]+=1
            spike_wait[somas[j]]=35
        #else:
        #    spike_check_arr[somas[j]]=0

def spike_good_time(s_array, t_spike, network_adj1, flux_vector, flux_spike_cupy, spike_check_arr):
    #can_spike = cp.where(t_spike<401)[0][s_array]
    can_spike = t_spike<401
    can_spike = s_array[can_spike[s_array]]

    multiarray = network_adj1[[can_spike]]
    flux_vector[multiarray] = (flux_spike_cupy[t_spike[can_spike]])
    t_spike[can_spike]+=1

    cannot_spike = t_spike>=401
    cannot_spike = s_array[cannot_spike[s_array]]
    t_spike[cannot_spike]=0
    spike_check_arr[[cannot_spike]] = 0
    
    #cannot_spike = cp.where(t_spike>=401)[0]
    #t_spike[cannot_spike]=0
    #spike_check_arr[[cannot_spike]] = 0

@cuda.jit
def spike_time(s_array, flux_vector, t_spike,spike_check_arr, connect, network_adj1):
    """
    Add flux spikes recieved from other neurons
    """

    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    for j in range(start, s_array.size, stride):#array check if s_array<402
        #x = s_array[j]
        if t_spike[s_array[j]] < 402:#array
            multiarray = network_adj1[s_array[j]]
            for l in range(connect + 1): #arry flux[multiarray] += b
                idx = multiarray[l]
                contribution = flux_spike[t_spike[s_array[j]]] / (connect + 1)
                cuda.atomic.add(flux_vector, idx, contribution)
            t_spike[s_array[j]] += 1
        else:
            spike_check_arr[s_array[j]] = 0
            t_spike[s_array[j]] = 0
 
mini = [[0, 0, 0.5, 0,  0,   0,   0],
        [0, 0, 0.4, 0,  0,   0,   0],
        [0, 0, 0,    0,  0,   0,   0.5],
        [0, 0, 0,    0,  0,   0.5,0],
        [0, 0, 0,    0,  0,   0.4,0],
        [0, 0, 0,    0,  0,   0,   0.4],
        [0, 0, 0,    0,  0,   0,   0,]]
mini = cp.asarray(mini, dtype=cp.float32)

def generate_adj_matrix(k, mini):
    adj_matrix = cp.zeros((k,k),dtype=cp.float32)
    for i in range(0,k,neuron_size):
        adj_matrix[i:i+neuron_size, i:i+neuron_size]=mini
    #adj_matrix=adj_matrix.astype(cp.float32)
    return adj_matrix

def create_somas(k, neuron_size):
    #somas = cp.zeros(int(k/neuron_size))
    #somas[0] = neuron_size-1
    #for i in range(1,int(k/neuron_size)):
    #    somas[i]=(somas[i-1]+neuron_size)
    somas = cp.arange(1,int(k/neuron_size)+1,dtype=cp.int32)
    somas *=7
    somas = somas-1
    #somas=somas.astype(cp.int32)
    return somas

#do list comprehension (faster, shorter)
def create_connections(k, connect, somas, neuron_size):
    network_adj = cp.zeros((k,connect+1), dtype=cp.int32)

    t = (cp.random.randint(0,k-1,size=(len(somas),connect),dtype=cp.int32))
    offset = (somas+2)%k
    offset = offset[:, cp.newaxis]
    t = cp.concatenate((t, offset),axis=1)

    #start1 = cp.maximum(0, somas - neuron_size + 1)
    #end1 = cp.minimum(k, somas)
    #forbidden_mask = ((t >= start1[:,cp.newaxis])  & (t<end1[:,cp.newaxis]))
    #while(cp.any(forbidden_mask)):
    #    new_values = (neuron_size-1+t)%k
    #    t[forbidden_mask] = new_values[forbidden_mask]

    for i in range(len(somas)):
        network_adj[somas[i]] = t[i]

    #for soma in somas:
    #    adj_array = cp.random.randint(0,k-1,size=connect,dtype=cp.int32)#+cp.array([(soma+9)%k])#cp.random.randint(0,k-1,size=connect,dtype=cp.int32)
    #    adj_array = cp.concatenate((adj_array,cp.array([(soma+2)%k]) ))
        #adj_array = cp.array([(soma+9)%k]) 
        #print('soma', soma, adj_array)
    #    print('b',adj_array)
    #    start = max(0, int(soma) - neuron_size + 1)
    #    end = min(k, int(soma))
    #    forbidden_array = cp.arange(start, end)
    #    for j in range(connect):
    #        while adj_array[j] in forbidden_array:
    #                adj_array[j] = (neuron_size - 1 + adj_array[j]) % k

    #    network_adj[soma]= adj_array 
    #print("final product ",network_adj)#how to ensure whole network is connected?

    return network_adj
    
#mempool = cp.get_default_memory_pool()
#pinned_mempool = cp.get_default_pinned_memory_pool()

#time complexity O(n)
def neuron_step(t,n, connect,data, flux_offset=0, track=True):
    '''
    Iterates through time and updates flux and signal using the equation (signal_vector@weight_matrix) + leaf_nodes*data[i%10000]
    and signal is updated using the update equation (4) from phenom paper
    '''
    start_gpu = cp.cuda.Event()
    end_gpu= cp.cuda.Event() 

    adj_matrix = generate_adj_matrix(n, mini)
    
    start_gpu.record()

    somas = create_somas(n,neuron_size)
    network_adj1 = create_connections(n, connect, somas, neuron_size)
    
    end_gpu.record()
    end_gpu.synchronize()
    print('init time',cp.cuda.get_elapsed_time(start_gpu, end_gpu)/1000)

    if(track):
        print("TRACKING")
        plot_signals =  cp.zeros((t,n), dtype=cp.float32) #O(t*n)
        plot_fluxes = cp.zeros((t,n), dtype=cp.float32)
    if(not track):
        print("NOT TRACKING")
        plot_signals =  0
        plot_fluxes = 0

    spike_check_arr = cp.zeros(n,dtype=cp.int32) #O(n)
    #spike_check_arr=spike_check_arr.astype(cp.int32) #O(n)
    t_spike = cp.zeros(n,dtype=cp.int32) #O(n)
    #t_spike = t_spike.astype(cp.int32)

    weight_matrix = adj_matrix

    leaf_nodes = cp.zeros(n,dtype=cp.float32) 
    leaf_nodes[0:k:neuron_size]= 1#0.6
    leaf_nodes[1:k:neuron_size]= 1#0.6
    leaf_nodes[3:k:neuron_size]= 1#0.6
    leaf_nodes[4:k:neuron_size]= 1#0.6
    leaf_nodes[0:k:2*neuron_size]=0.2
    leaf_nodes[3:k:2*neuron_size]=0.2
    leaf_nodes[somas] = 0
    #leaf_nodes = leaf_nodes.astype(cp.float32)

    signal_vector = leaf_nodes*data[0]
    signal_vector = signal_vector.astype(cp.float32)
    r_fq = cp.zeros(n, dtype=cp.float32)
    
    gpu_array=0#cp.zeros(t,dtype=cp.float32)
    spike_counter=cp.zeros(n,dtype=cp.float32)
    spike_wait = cp.zeros(n, dtype=cp.int32)

    #print('cpu pre loop',psutil.cpu_percent())

    #psutil.virtual_memory().available *100/psutil.virtual_memory().total
    #psutil.cpu_percent()
    #psutil.virtual_memory().percent

    for i in range(t): #O(t* )
        #print(f"Timestep = {i}", end="\r") 
            
        flux_vector=(cp.matmul(signal_vector,weight_matrix))+flux_offset+(leaf_nodes * data[i%1000])# + flux_offset
        #O(n * n) + O(n)

        #print('cpu in loop1',psutil.cpu_percent())
        #gpu_array[i]= psutil.cpu_percent()
        
        

        if(cp.max(spike_check_arr)==1):
            s_array = cp.where(spike_check_arr==1)[0]
            #print('s_array',s_array)            
            spike_good_time(s_array, t_spike, network_adj1, flux_vector, flux_spike_cupy, spike_check_arr)
            #O(n)
            #spike_time[256,256](s_array, flux_vector, t_spike, spike_check_arr, connect, network_adj1)
                    
        r_fq[:]=0

        #s_of_array_phi(flux_vector, signal_vector,n, r_fq)
        s_of_phi[512,1024](flux_vector, signal_vector,n, r_fq) #O(n)
        #print('cpu s_of_phi',psutil.cpu_percent())

        signal_vector = signal_vector*(1 - d_tau*alpha/beta) + (d_tau/beta )*r_fq #O(n)
        #print('cpu in loop s',psutil.cpu_percent())

        if (cp.max(signal_vector[somas])>=s_th):
            #spike_good_check(signal_vector, somas, spike_check_arr, spike_counter, spike_wait)
            spiking = signal_vector>=s_th
            #spiking =spiking.astype(cp.int32)
            spiking = somas[spiking[somas]]

            signal_vector[spiking] = 0
            spike_check_arr[spiking] = 1
            spike_counter[spiking]+=1
            spike_wait[spiking]=35
            #spike_check[512,1024](signal_vector, somas, spike_check_arr, spike_counter, spike_wait)

        spike_wait[somas] -= 1
        wait_index = cp.argwhere(spike_wait > 0).flatten() #n n log n?
        #print('timer',spike_wait)
        #print('wait index', wait_index)
        signal_vector[wait_index]=cp.float32(0)

        if(track):
            plot_signals[i] = signal_vector
            plot_fluxes[i] = flux_vector
        #print('gc count', gc.get_count())
        #gpu_array[i]=psutil.cpu_percent()#gc.get_count()[0]
        #print('flux ref',sys.getrefcount(flux_vector))

        #res0 = nvidia_smi.nvmlDeviceGetUtilizationRates(handle0)
        #gpu_array[i] = res0.gpu
        #mem_array[i] = res0.memory
     
    return plot_signals, plot_fluxes, weight_matrix, gpu_array, spike_counter

start_gpu1 = cp.cuda.Event()
end_gpu1= cp.cuda.Event() 
num_iter=1
convergence=False
time_array = []

for iterations in range(num_iter):
    flux_offset=cp.zeros(k,dtype=cp.float32)
    expected_spikes=cp.zeros(k,dtype=cp.float32)
    ones = cp.ones(k,dtype=cp.float32)
    expected_spikes[:]=5
    #ones[som]=0
    averages = cp.zeros(k, dtype=cp.float32)

    #while convergence==False:
    print('cpu pre call',psutil.cpu_percent())

    start_gpu1.record()
    plot_signals,plot_fluxes, weight_matrix, gpu_array, spike_counter = neuron_step(t, k ,connect, data, flux_offset, track=True)
    print('cpu post call',psutil.cpu_percent())

    end_gpu1.record()
    end_gpu1.synchronize()
    t_gpu = cp.cuda.get_elapsed_time(start_gpu1, end_gpu1)
    k += 2100
    #connect += 200
    #t += 500
    print('gpu time',t_gpu/(1000))
    time_array.append(t_gpu/(1000))
    #k = k+7
    '''
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
    '''
#print('dev',np.std(gpu_array))
#print('count',cp.count_nonzero(weight_matrix)/(k**2))
#print(weight_matrix[:,1500])
#print('sum',cp.sum(weight_matrix[:,1]))

print('timesteps',t)
print('dend',k)
print('connect', connect)
print('not recording')
print('spikes')
#recorded_timestep = loadtxt('recorded-timestep.csv')
recorded_dend = loadtxt('recorded-dend.csv')
#recorded_connect = loadtxt('recorded-connect.csv')

line1, = plt.plot(np.arange(10)*2100,recorded_dend, color='blue', label="Tracking")
line2, = plt.plot(np.arange(num_iter)*2100,time_array, color ='red', label='No tracking')
plt.legend(handles=[line1,line2])
#plt.plot(cp.arange(num_iter)*2100,time_array)
#np.savetxt("recorded-dend.csv",time_array)
plt.title("Simulation Runtime vs Dendrites")
plt.xlabel('Dendrites')
plt.ylabel('Runtime (s)')
plt.ylim(0,max(time_array)+1)
plt.savefig('a_paper_4.png', dpi=400, bbox_inches='tight')

time_axis = np.arange(t)

#fig, axs = plt.subplots(1)
#axs.plot(time_axis, cp.asnumpy(plot_signals)[:,13])
#axs.plot(time_axis, cp.asnumpy(plot_fluxes)[:,13])
#axs.plot(time_axis,cp.asnumpy(gpu_array))
#axs.plot(time_axis,cp.asnumpy(mem_array)/50)
#axs.set_ylim(0,100)
#plt.savefig('a_paper_1.png')
#plt.savefig('mem_cpu.png', dpi=400, bbox_inches='tight')
plot_stuff=False
if(plot_stuff):
    binary_tree = nx.DiGraph()
    edges = [(2,0),(2,1),(6,2),(5,3),(5,4),(6,5)]

    binary_tree.add_edges_from(edges)
    pos = nx.nx_agraph.graphviz_layout(binary_tree, prog="dot", args="-Grankdir=RL")
    
    fig1, axs1 = plt.subplots(figsize=(20, 20))
    nx.draw(binary_tree, pos, with_labels=True, node_color='lightblue', node_size=500, ax=axs1)
    print(binary_tree.nodes)
    #index = 0
    for node in binary_tree.nodes:
        x_fig, y_fig = axs1.transData.transform(pos[node])
        x_norm = x_fig / fig1.bbox.width
        y_norm = y_fig / fig1.bbox.height

        time_axis
        s1 = cp.asnumpy(plot_signals)[:,node+7]
        f1 = cp.asnumpy(plot_fluxes)[:,node+7]
        print(s1.size)
        #index+=1
        inset_size = 0.21  # Increase size for better visibility
        inset_ax = fig1.add_axes([x_norm - inset_size / 2, y_norm - inset_size / 2, inset_size, inset_size])

        inset_ax.plot(time_axis, s1, label='Signal')
        inset_ax.plot(time_axis, f1, label='Flux')
        inset_ax.set_xlim(0,1000)
        inset_ax.set_ylim(0.0,1.2)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])


    plt.savefig('a_paper_6.png')

"""
somas1 = create_somas(70,neuron_size)
network_adj = create_connections(70, connect, somas1, neuron_size)
print(network_adj)
edges = []
for node in range(network_adj.shape[0]):
    if node in somas1:  
        connections = network_adj[node]
        for target in cp.asnumpy(connections): 
            edges.append((node, target))

G = nx.DiGraph()
G.add_edges_from(edges)

plt.figure(figsize=(8, 8))
pos = nx.circular_layout(G) 
nx.draw(G, pos, with_labels=False, node_color='black', node_size=10, edge_color='gray')  # No labels, black nodes, smaller size
plt.savefig('a_paper_6.png')
"""

#fig1, axs1 = plt.subplots(14)
#plt.subplots_adjust(left=0.125,bottom=0.1,right=0.9,top=0.9,wspace=0.2,hspace=0.7)
#for i in range(14):
    #axs1[i].plot(time_axis, cp.asnumpy(plot_signals)[:,i])
    #axs1[i].plot(time_axis, cp.asnumpy(plot_fluxes)[:,i])
    #axs1[i].x_label("Time (ns)")
    #axs1[i].y_label("Flux (Orange), Signal (Blue)")
    #axs1[i].title("Soma of Neuron 0")

    #axs1.set_ylim(0,1.2)
    #axs1.set_xlim(0,1000)
#fig1.tight_layout()
#plt.savefig('a_paper_1.png', dpi=400, bbox_inches='tight')
    
