from math_sim import *


n=10
plot_signals,plot_fluxes,weight_matrix = neuron_step(t,n, data)

#print('weights \n',weight_matrix)
#print('fluxes ',signal_vector@weight_matrix)
#print('signals ', signal_vector)
#print('leaf ', leaf_nodes)
#print('runtime ', run_time)
#print('plot ', plot_signals[:,0])
plot_signal_flux(plot_signals, plot_fluxes,weight_matrix, t, n)


#iterate through time
@jit(nopython=True)
def neuron_step(t,n, data, random_weights=True):
    '''
    Iterates through time and updates flux and signal using the equation (signal_vector@weight_matrix) + leaf_nodes*data[i%10000]
    and signal is updated using the update equation (4) from phenom paper
    '''
    #phi_spd=0.5
    #learning_rate=.01

    plot_signals = 0 #np.zeros((t,n))
    plot_fluxes = 0#np.zeros((t,n))

    weight_matrix = generate_graph(n)
    t_refractory=0
    


    leaf_nodes = get_leaves_classic(weight_matrix,n)
    signal_vector = leaf_nodes*data[0]

    #print('plot ', np.shape(plot_signals))
    #print('matrix ',np.shape(weight_matrix))
    #print('leaf ', (leaf_nodes))
    #print('sig ', np.shape(signal_vector))

    for i in range(t):
        #flux_offset =  learning_rate*np.average(plot_signals[:,i]) #* expected_signal[:][i]-plot_signals[:][i] #use a running average or an exact average? using plot signals?
        #print('slice ',plot_signals[:,i])
        #print('avg ',flux_offset)
        flux_vector = (signal_vector@weight_matrix) + leaf_nodes*data[i%10000]

        #if(i==162):
        #    print(data[i])
        #    print('leaf checl ', leaf_nodes*data[i])
        #    print('check ', flux_vector)
        #    print('check ',signal_vector@weight_matrix)
        
        signal_vector = signal_vector*(1- (1e-9/1.2827820602389245e-12)*(.053733049288045114/(2*np.pi*1e3))) + ((1e-9/1.2827820602389245e-12)/(2*np.pi*1e3))*s_of_phi(flux_vector, signal_vector,n)
        if signal_vector[-1]>0.7:
            signal_vector[-1]=0
            #flux_vector[-1]=0
            #print(t)
            #t_refractory = i+10
            #print(i<t_refractory)
        #if i<t_refractory:
        #    signal_vector[-1]=0
            #flux_vector[-1]=0
        #dend.s[t_idx+1] = dend.s[t_idx]*(1 - d_tau*dend.alpha/dend.beta) + (d_tau/dend.beta)*r_fq
    
        #plot_signals[i] = signal_vector
        #plot_fluxes[i] = flux_vector
    
    return plot_signals, plot_fluxes, weight_matrix