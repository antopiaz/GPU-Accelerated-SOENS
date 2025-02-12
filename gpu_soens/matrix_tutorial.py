from gpu_sim import neuron_step, data
import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
from numba import jit, cuda
import time
import cupy as cp
import gc
import networkx as nx

t=1000 #time steps
k= 14 #same as n, number of neurons, must be multiple of neuron size
neuron_size = 7
connect = 1

start_gpu1 = cp.cuda.Event() #timing
end_gpu1= cp.cuda.Event() 

num_iter=10
time_array = []

for iterations in range(num_iter):
    flux_offset=cp.zeros(k,dtype=cp.float32)

    start_gpu1.record()
    plot_signals,plot_fluxes, weight_matrix, gpu_array, spike_counter = neuron_step(t, k ,connect, data, flux_offset, track=True)

    end_gpu1.record()
    end_gpu1.synchronize()
    t_gpu = cp.cuda.get_elapsed_time(start_gpu1, end_gpu1)
    
    k += 2100
    #connect += 200
    #t += 500
    print('gpu time',t_gpu/(1000))
    time_array.append(t_gpu/(1000))
    

print('timesteps',t)
print('dend',k)
print('connect', connect)
print('not recording')
print('spikes')
#recorded_timestep = loadtxt('recorded-timestep.csv')
#recorded_dend = loadtxt('recorded-dend.csv')
#recorded_connect = loadtxt('recorded-connect.csv')

#line1, = plt.plot(np.arange(10)*2100,recorded_dend, color='blue', label="Tracking")
#line2, = plt.plot(np.arange(num_iter)*2100,time_array, color ='red', label='No tracking')
#plt.legend(handles=[line1,line2])
plt.plot(np.arange(num_iter)*2100,time_array)
#np.savetxt("recorded-dend.csv",time_array)
plt.title("Simulation Runtime vs Dendrites")
plt.xlabel('Dendrites')
plt.ylabel('Runtime (s)')
plt.ylim(0,max(time_array)+1)
plt.savefig('a_test_4.png', dpi=400, bbox_inches='tight')

time_axis = np.arange(t)

plot_stuff=True
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


    plt.savefig('a_test_6.png')
