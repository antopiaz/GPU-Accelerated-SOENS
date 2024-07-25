import numpy as np
import cupy as cp
import random
import matplotlib.pyplot as plt
from numpy import loadtxt
from numba import jit, cuda
import time
from cupy import cuda as cua

#external signal coming in to spd
data = loadtxt('phi_signal.csv', delimiter=',')
data = cp.float16(data)
plt.plot(np.arange(0,10001), data)
plt.savefig('input_sig.png', dpi=400, bbox_inches='tight')

flux_spike = loadtxt('spike.csv')
flux_spike=cp.float16(flux_spike)
#plt.plot(np.arange(403),flux_spike)
#plt.savefig('input_spike.png')

#physical constants
phi_th=0.1675
d_tau = cp.float32(1e-9/1.2827820602389245e-12)
beta = cp.float32(2000*np.pi)
alpha = cp.float32(0.053733049288045114)
A=cp.float32(1)
B=cp.float32(.466)
ib=cp.float32(1.8)
s_th = cp.float32(0.694) #signal threshold for somas

#time and size parameters
t=3000
k=7000 #overall number of dend and somas, must be multiple of neuron_size
neuron_size = 7 #number of dend and soma in a neuron

@cuda.jit
def s_of_phi(phi,s,n, r_fq):
    """
    Function to approximate rate array 
    """
    start=cuda.grid(1)
    stride=cuda.gridsize(1)
    for i in range(start,n,stride):
        if phi[i]<phi_th: 
            r_fq[i] = 0 #if flux below threshold, pass on nothing
        else:
            r_fq[i] = A*(phi[i]-phi_th)*((B*ib)-s[i]) #else pass on this
        
@cuda.jit
def spike_check(signal_vector, somas,spike_check_arr):
    """
    Iterates through all the somas to check if their signal is above threshold
    """
    start=cuda.grid(1)
    stride=cuda.gridsize(1)

    for j in range(start,int(k/neuron_size),stride): #only k/neuron_size number of somas
        if signal_vector[somas[j]]>s_th: #flux threshold for spiking is 0.7
            signal_vector[somas[j]]=0 #reset signal to 0
            spike_check_arr[somas[j]]=1 #set spike check to 1
        
@cuda.jit
def spike_time(s_array, flux_vector, t_spike,spike_check_arr):
    """
    Add flux spikes recieved from other neurons
    """
    start=cuda.grid(1)
    stride=cuda.gridsize(1)

    for j in range(start,s_array.size, stride): #go through all somas that are currently sending spikes
        x = s_array[j] #get index of the activated soma
        if  t_spike[x]<=402: #check if the spike is still coming through since it takes 403 time steps
            flux_vector[(x+3)%(k-1)] +=flux_spike[t_spike[x]] #send the spike to the next neuron's
                                                              # third dendrite
            t_spike[x] +=1
        else: 
            spike_check_arr[x] = 0  #reset the soma and the timer
            t_spike[x]=0
       

mini = [[0, 0.6, 0, 0,   0,   0,   0], #dend0 -> dend1 
        [0, 0,   0, 0.5, 0,   0,   0], #dend1 -> dend3
        [0, 0,   0, 0.7, 0,   0,   0], #dend2 -> dend3
        [0, 0,   0, 0,   0.7, 0,   0], #dend3 -> dend4
        [0, 0,   0, 0,   0,   0.5, 0], #dend4 -> dend5
        [0, 0,   0, 0,   0,   0,   0.7], #dend5 -> soma
        [0, 0,   0, 0,   0,   0,   0,]] #soma passes on spikes, so in adj_mat it's all 0

mini = cp.asarray(mini) #mini is the adj matrix of the dend and soma for one neuron

def generate_adj_matrix(mini):
    adj_matrix = cp.zeros((k,k), dtype=cp.float16) #empty adj matrix
    for i in range(0,k,neuron_size):
        adj_matrix[i:i+neuron_size, i:i+neuron_size]=mini
    return adj_matrix
# ^^^ fill the total adj matrix with all of the adj matrices of the neurons
#so it's a bunch of identical neurons
#YOU CAN MAKE YOUR OWN ADJACENCY MATRIX FOR DENDRITES WHERE THE ROWS FOR SOMAS ARE ALL ZEROS
#e.g. fully connected matrix with no self connections? random generated matrix? random sparse generated matrix?
#e.g. or change mini to different size and values

adj_matrix = generate_adj_matrix(mini)


som = cp.zeros(int(k/neuron_size), dtype=int)
som[0] = neuron_size-1 
for i in range(1,int(k/neuron_size)):
    som[i]=(som[i-1]+neuron_size)
#create an array of all the indices of the somas
#since each neuron is size 7 that means the first soma is at 6, then 13,20,27,34, ...
#you can check by printing out the adj_matrix and som and see if the empty rows match up to the soma indices


def neuron_step(t,n, data):
    '''
    Iterates through time and updates flux and signal using the equation (signal_vector@weight_matrix) + leaf_nodes*data[i%10000]
    and signal is updated using the update equation (4) from phenom paper
    '''
    #initialize arrays for everything
    spike_check_arr = cp.zeros(n, dtype=int) #array of all dendrites and somas
    t_spike = cp.zeros(n, dtype=int) #array of all 
    somas = som
    
    plot_signals =  cp.zeros((t,n), dtype=cp.float16)
    plot_fluxes = cp.zeros((t,n), dtype=cp.float16)

    weight_matrix = adj_matrix

    leaf_nodes = cp.zeros(k,dtype=cp.float16)
    leaf_nodes[0:k:neuron_size]= 0.7
    leaf_nodes[2:k:neuron_size]= 0.7
    leaf_nodes[somas] = 0
    # ^^^ so each first and third dendrite in each neuron gets external signal
    #and soma do not get external signal

    signal_vector = leaf_nodes*data[0] #init signal vector
    r_fq = cp.zeros(n, dtype=cp.float16) #init vector for flux that will be passed on

    for i in range(t):
        #calculate flux for this step based on incoming internal signal and external signal
        flux_vector=(cp.matmul(signal_vector,weight_matrix))+(leaf_nodes * data[i%10000])

        if(cp.max(spike_check_arr)==1): #if at least one of the somas is activated then 
            s_array = cp.where(spike_check_arr==1)[0] #get indices of all activated somas
            spike_time[256,256](s_array, flux_vector, t_spike, spike_check_arr) #send spikes from activated somas
           
        r_fq[:]=0
        s_of_phi[512,1024](flux_vector, signal_vector,n, r_fq) #do rate array
        signal_vector = signal_vector*(1 - d_tau*alpha/beta) + (d_tau/beta )*r_fq #update signal

        if (cp.max(signal_vector[somas])>s_th): #if at least one soma is above threshold
            spike_check[512,1024](signal_vector, somas, spike_check_arr) #check which somas are above threshold and activate them
       

        plot_signals[i] = signal_vector #save current flux and signal for later
        plot_fluxes[i] = flux_vector

    return plot_signals, plot_fluxes, weight_matrix



start_gpu1 = cp.cuda.Event()
end_gpu1= cp.cuda.Event() 

start_gpu1.record()

plot_signals,plot_fluxes, weight_matrix = neuron_step(t, k , data) #run simulation

end_gpu1.record()
end_gpu1.synchronize()
t_gpu = cua.get_elapsed_time(start_gpu1, end_gpu1)

print('time',t_gpu/1000)

time_axis = np.arange(t)
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

for i in range(14):
    axs[i].set_ylim(0,1.2)
    axs[i].set_xlim(0,2000)

plt.savefig('flux_sig_spike_plot3.png', dpi=400, bbox_inches='tight')
