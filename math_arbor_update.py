import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import loadtxt
import networkx as nx
from math_sim import generate_graph, get_leaves, s_of_phi, plot_signal_flux
from numba import jit
import time


t=500
n=13
letters = [np.array([1,1,0,0,1,0,0,1,1,  0,0,0,0]), np.array([1,0,1,1,0,1,0,1,0  ,0,0,0,0]), np.array([0,1,0,1,0,1,1,0,1  ,0,0,0,0]) ] #adding a bunch of zeros
#maybe extend or shorten one of the arrays later on but could be very expensive
letters = np.array(letters)

weight_matrix=np.zeros((13,13))

for i in range(13):
    if i<3:
        weight_matrix[i][9]+=0.5
    if 2<i<6:
        weight_matrix[i][10]+=0.5
    if 5<i<9:
        weight_matrix[i][11]+=0.5
    if 8<i<12:
        weight_matrix[i][12]+=0.5
print(weight_matrix)


@jit(nopython=True)
def arbor_step(t,n,phi_spd, flux_offset):
    '''
    Iterates through time and updates flux and signal using the equation (signal_vector@weight_matrix) + leaf_nodes*data[i%10000]
    and signal is updated using the update equation (4) from phenom paper
    '''
    plot_signals = np.zeros((t,n))
    plot_fluxes = np.zeros((t,n))
    count = 0
    t_refractory =0


    leaf_nodes = get_leaves(weight_matrix,n,random_val=False)

    signal_vector = leaf_nodes*phi_spd#leaf_nodes*data[0]

    #print('plot ', np.shape(plot_signals))
    #print('matrix ',np.shape(weight_matrix))
    #print('leaf ', (leaf_nodes))
    #print('sig ', np.shape(signal_vector))
    average = np.zeros(n)
    for i in range(t):
        #print('slice ',plot_signals[:])

        #print('avg ',flux_offset)
        flux_vector = (signal_vector@weight_matrix) + leaf_nodes*phi_spd + flux_offset#leaf_nodes*data[i%10000]

        #if(i==162):
        #    print(data[i])
        #    print('leaf checl ', leaf_nodes*data[i])
        #    print('check ', flux_vector)
        #    print('check ',signal_vector@weight_matrix)
        
        signal_vector = signal_vector*(1- (1e-9/1.2827820602389245e-12)*(.053733049288045114/(2*np.pi*1e3))) + ((1e-9/1.2827820602389245e-12)/(2*np.pi*1e3))*s_of_phi(flux_vector, signal_vector,n)
        if signal_vector[-1]>0.7:
            count += 1
            signal_vector[-1]=0
            t_refractory = i+5 #unclear???
        if i<t_refractory:
            signal_vector[-1]=0
            #flux_vector[-1]=0
        #dend.s[t_idx+1] = dend.s[t_idx]*(1 - d_tau*dend.alpha/dend.beta) + (d_tau/dend.beta)*r_fq

        #new average = old average + (next data - old average) / next count
        #average = average + ((signal_vector[-1] - average)/(i+1))
        average = average + signal_vector

        plot_signals[i] = signal_vector
        plot_fluxes[i] = flux_vector

    average  = average/t
    return plot_signals, plot_fluxes, weight_matrix, count, average

@jit(nopython=True)
def arbor_update_rule(letters, i, learning_rate=.01, single_classifier=True):
    convergence = False
    flux_offset = np.zeros(n)
    spikes=[[4,0,0],
            [0,4,0],
            [0,0,4]]
    
    if single_classifier:
        expected_spikes =spikes[i]# [1,2,4]
    else:
        expected_spikes=[1,2,4]

    while convergence != True:
        total_error = np.zeros(3)
        
        for k in range(np.shape(letters)[0]):

            plot_signals, plot_fluxes, weight_matrix, count, average = arbor_step(t,n,letters[k], flux_offset)

            error=expected_spikes[k]-count #is error same as delta y? how to calculate
            
            total_error[k] = total_error[k]-error
            #print('total ',total_error)

            for j in range(n):

                #print(np.average(plot_signals[:,j]) - average[j])
                #print(average)

                #flux_offset[j] +=  learning_rate*np.average(plot_signals[:,j])*error 
                flux_offset[j] +=  learning_rate*average[j]*error 

                #* expected_signal[i]-plot_signals[i] #use a running average or an exact average? using plot signals?
                #spikes from soma threshold

        if total_error.any()==0:
            convergence=True
            #print('flux ',flux_offset)
            #print(count)
            #plot_signal_flux(plot_signals, plot_fluxes,weight_matrix, t, n)
            return flux_offset

         
node_num=2
run_time=0
for i in range(50):
    t1 = time.perf_counter()

    flux = arbor_update_rule(letters,node_num, single_classifier=True)
    t2 = time.perf_counter()
    run_time = run_time + t2 - t1

print('run_time', run_time/50)

#with calculated avg 0.165, 0.122 with numba 0.027
#with numpy 0.172, 0.122 with numba 0.025

plot_signals1, plot_fluxes1, weight_matrix, count1, average1 = arbor_step(t,n,letters[0], flux)
'''
plot_signals2, plot_fluxes2, weight_matrix, count2 = arbor_step(t,n,letters[1], flux)

plot_signals3, plot_fluxes3, weight_matrix, count3 = arbor_step(t,n,letters[2], flux)
print(count1, count2, count3)
#plot_signal_flux(plot_signals, plot_fluxes,weight_matrix, t, n)

title_letter=['Z','V','N']   
truncate = 0
time_axis = np.arange(truncate,t)

fig, axs = plt.subplots(3, figsize=(7,7))
plt.suptitle(title_letter[node_num] + ' node')

axs[0].plot(time_axis, plot_signals1[:,i][truncate:t], label='signal')
axs[0].plot(time_axis, plot_fluxes1[:,i][truncate:t], label='fluxes')
axs[0].set_title('z')

axs[1].plot(time_axis, plot_signals2[:,i][truncate:t], label='signal')
axs[1].plot(time_axis, plot_fluxes2[:,i][truncate:t], label='fluxes')
axs[1].set_title('v')

axs[2].plot(time_axis, plot_signals3[:,i][truncate:t], label='signal')
axs[2].plot(time_axis, plot_fluxes3[:,i][truncate:t], label='fluxes')
axs[2].set_title('n')

plt.show()
'''
