import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import loadtxt
import networkx as nx
from numba import jit
import time


#phi_spd
data = loadtxt('phi_signal.csv', delimiter=',')
#plt.plot(np.arange(0,10001), data)
#plt.show()

#total_time_2 = loadtxt('time_perf2.csv', delimiter=',')
#plt.plot(np.arange(0,50000,1000), total_time_2)

#plt.plot(np.arange(2,1020,30), total_time_1, "r")
#plt.show()

phi_th=0.1675
t=1000

#letters = [np.array([1,1,0,0,1,0,0,1,1]), np.array([1,0,1,1,0,1,0,1,0]), np.array([0,1,0,1,0,1,1,0,1]) ]
#letters = np.array(letters)

@jit(nopython=True)
def s_of_phi(phi,s,n,A=1,B=.466,ib=1.8):
    """
    Function to get rate array?
    """
    phi_th = 0.1675
    r_fq = A*(phi-phi_th)*((B*ib)-s) #s vector, and phi     
    for i in range(n):
        if phi[i]<phi_th: 
            r_fq[i] = 0  #if phi from incoming node is below threshold then it passes on nothing
    return r_fq

@jit(nopython=True)
def generate_graph(n):
    '''
    Generate arbitrary tree graph using adjacency graph
    '''
    weight_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(i):  #graph connectivity problems
            if weight_matrix[j].any() == 0: #why does this work?
                value = random.randint(j+1,n-1)   # random.choice(np.arange(j+1,n,1))
                weight_matrix[j][value]=0.5#(round(random.random(),2))*0.7+0.3
                #weight_matrix[j][value]=0.5
    
    return weight_matrix

@jit(nopython=True)
def get_leaves(weight_matrix, n, random_val=True):
    '''
    to find columns that are empty implying a leaf node
    '''

    leaf_nodes = np.zeros(n)
    if(random_val):            
        for i in range(n-1):
            if(weight_matrix[:,i].any()==0):
                    leaf_nodes[i]+=(round(random.random(),2))*0.7+0.3
    else:
        for i in range(n-1):
            if(weight_matrix[:,i].any()==0):
                    leaf_nodes[i]+=0.5
            
            
    return(leaf_nodes)

@jit(nopython=True)
def get_leaves_classic(weight_matrix, n):
    '''
    to find columns that are empty implying a leaf node
    '''

    leaf_nodes = np.zeros(n)
               
    for i in range(n-1):
        if(weight_matrix[:,i].any()==0):
                leaf_nodes[i]+=(round(random.random(),2))*0.7+0.3
            
    return(leaf_nodes)


#iterate through time
@jit(nopython=True)
def neuron_step(t,n, data, random_weights=True):
    '''
    Iterates through time and updates flux and signal using the equation (signal_vector@weight_matrix) + leaf_nodes*data[i%10000]
    and signal is updated using the update equation (4) from phenom paper
    '''
    #phi_spd=0.5
    #learning_rate=.01

    plot_signals = np.zeros((t,n))
    plot_fluxes = np.zeros((t,n))

    weight_matrix = generate_graph(n)
    t_refractory=0
    
    leaf_nodes = get_leaves_classic(weight_matrix,n)
    signal_vector = leaf_nodes*data[0]

   

    for i in range(t):
        #flux_offset =  learning_rate*np.average(plot_signals[:,i]) #* expected_signal[:][i]-plot_signals[:][i] #use a running average or an exact average? using plot signals?
        #print(i)
        flux_vector = (signal_vector@weight_matrix) + leaf_nodes*data[i%10000]

        
        
        #dend.s[t_idx+1] = dend.s[t_idx]*(1 - d_tau*dend.alpha/dend.beta) + (d_tau/dend.beta)*r_fq
        signal_vector = signal_vector*(1- (1e-9/1.2827820602389245e-12)*(.053733049288045114/(2*np.pi*1e3))) + ((1e-9/1.2827820602389245e-12)/(2*np.pi*1e3))*s_of_phi(flux_vector, signal_vector,n)
        #if signal_vector[-1]>0.7:
        #    signal_vector[-1]=0

            #flux_vector[-1]=0
            #print(t)
            #t_refractory = i+10
            #print(i<t_refractory)
        #if i<t_refractory:
        #    signal_vector[-1]=0
            #flux_vector[-1]=0

        plot_signals[i] = signal_vector
        plot_fluxes[i] = flux_vector
    
    return plot_signals, plot_fluxes, weight_matrix


def time_measure(data, t, mode="length"):
    '''
    Used to measure the performance of neuron step function for increasing time length and increasing dendrites
    '''
    time_array = np.zeros(0)
    count=0
    
    if(mode=="length"):
        #n = random.randint(2,20)
        n=100

        for j in range(0,t,1000):
            t1 = time.perf_counter()
            plot_signals,plot_fluxes, weight_matrix = neuron_step(j,n, data)
            t2 = time.perf_counter()
            run_time = t2-t1
            #print(run_time)
            if(j%1000==0):
                print(count)
                count +=1

            time_array=np.append(time_array, run_time)

            #check to make sure it is running correctly
            if(j==50000):

                truncate = 0
                time_axis = np.arange(truncate,49000)

                fig, axs = plt.subplots(n)

                for i in range(n):
                    axs[i].plot(time_axis, plot_signals[:,i][truncate:49000], label='signal')
                    axs[i].plot(time_axis, plot_fluxes[:,i][truncate:49000], label='fluxes')
                    #axs.legend()
                    #axs[i].set_title('node ' + str(i))
                plt.show()

    
    elif (mode=="size"):
        for k in range(2,6020,1000):

            print(f"Run = {k}", end="\r")

            t1 = time.perf_counter()
            plot_signals,plot_fluxes, weight_matrix = neuron_step(int(t), k , data)
            t2 = time.perf_counter()
            run_time = t2-t1
            print(run_time)

            time_array=np.append(time_array, run_time)
            #print(time_array)

    return time_array



mode=''#'length'#'length'
if(mode=='size'):
    print(mode)
    total_time = time_measure( data,t, mode="size")
    #np.savetxt("time_perf1.csv", total_time, delimiter=",")
    #print(np.shape(np.arange(2,1020,11)))
    #plt.plot(np.arange(2,6020,1000), total_time)
    #plt.plot(np.arange(2,1020,30), total_time_1, "r")
    
    #plt.show()
elif(mode=='length'):
    total_time = time_measure( data,t, mode="length")
    
    #np.savetxt("time_perf2.csv", total_time, delimiter=",")
    plt.plot(np.arange(0,t,1000), total_time)
    #plt.plot(np.arange(0,t,1000), total_time_2, "r")

    #plt.show()


#plot
def plot_signal_flux(plot_signals, plot_fluxes,weight_matrix, t, n):
    '''
    Plots the signal flux and dendrite graph
    '''
    truncate = 0
    time_axis = np.arange(truncate,t)

    fig, axs = plt.subplots(n)
    #plt.set_ylim(bottom=0)
    #plt.ylim(top=1)
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.8)
    
    for i in range(n):
        axs[i].plot(time_axis, plot_signals[:,i][truncate:t], label='signal')
        axs[i].plot(time_axis, plot_fluxes[:,i][truncate:t], label='fluxes')
        axs[i].set_ylim(0,1.2)
        #axs.legend()
        axs[i].set_title('node ' + str(i))
    plt.show()

    #fig, axs = plt.subplots(n)

    #for i in range(n):
    #    axs[i].plot(time_axis, plot_signals[:,i][truncate:t], label='signal')
    #plt.show()


    edges = []
    for i in range(n):
        for j in range(n):
            if weight_matrix[:,i][j] !=0:
                edges.append((j, i))

    #print(edges)
    G = nx.DiGraph(directed=True)
    G.add_edges_from(
        edges)
    
    edge_labels = dict([((n1, n2), np.around(weight_matrix[n1][n2], decimals=2))
                    for n1, n2 in G.edges])
    pos = nx.planar_layout(G)

    nx.draw_networkx(G,pos)#, node_color=values)

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    #values = np.ones(n)
    #values[n-1]+=1
    #print(values)
    #pos = nx.planar_layout(G)



    plt.show()




#t1 = time.perf_counter()
#plot_signals,plot_fluxes, weight_matrix = neuron_step(int(t), 3000 , data)
#t2 = time.perf_counter()
#print(t2-t1)
#print(get_leaves(weight_matrix,13, random_val=False))

    


    