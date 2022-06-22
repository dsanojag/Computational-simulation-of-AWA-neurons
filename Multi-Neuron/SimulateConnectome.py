
import neuroml
import neuroml.loaders as loaders
import neuroml.writers as writers
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
from scipy.integrate import odeint
import numpy as np
import random
import matplotlib.animation as animation
import time
import pickle
import os

usePickle = False

'''Connectome parameters.'''

connectome_spreadsheet = "OpenWorm/connectome.csv"
#Olfaction neurons
#all_the_cell_names = ['ASHR',"AWAR", "AWAL","AWBL","AWBR","AWCL","AWCR"]

#just the core sensory + interneuron circuit from Wicks
all_the_cell_names = ["AWAR", "AWAL","ALML","ALMR",'ASHL', 'ASHR', "AVM", "AVDL","AVDR", "AVAL","AVAR","AVBL","AVBR","AWBL","AWBR","PLML","PLMR","PVCL","PVCR","PVDL","PVDR","DVA"] 

#all_the_cell_names = ['ASHR']

l = len(all_the_cell_names)
indices = range(l)

coords = [] # list of 3D coordinates of all the neurons
cells_connections = {} # synaptic
cells_gaps = {} # gap junctional

connection_nums = {} # synaptic connection nums dict
gap_nums = {} # gap junction connection nums dict

GABA_adjacency = {} # synaptic connection nums dict specific to GABA
Chol_adjacency = {} # synaptic connection nums dict specific to Acetylcholine
Glut_adjacency = {} # synaptic connection nums dict specific to Glutamate
Other_adjacency = {} # synaptic connection nums dict specific to Other
GABA_Chol_Glut_adjacency = {} # synaptic connection nums dict including everything except "Other"

GABA_expressing = [] # list of pre-synaptic cells expressing GABA
Cholinergic = [] # list of pre-synaptic cells expressing Acetylcholine
Glutamatergic = [] # list of pre-synaptic cells expressing Glutamate
labeled_neuron = "PLML" # give a particular neuron a different color
coords_of_labeled_neuron = (None, None, None)

'''Simulation parameters'''
# params of the simulation
I_ext_magnitude = 18 * 1e-12 # external current in units of Amperes
stim_dur = 0.5
stim_start = 0.5
start_time = 0.0
end_time = 1.5
I_ext_mask = [1.0 if all_the_cell_names[i] in ["PLML", "PLMR"] else 0.0 for i in indices] # which neurons get external current input

'''Dynamical parameters.'''
enhanced_synaptic_steepness = 8.0 # Important -- set this to > 1 to increase the nonlinearity: it appears to be effectively set to 1.0, however, in the Kunert and Wicks papers
K = -4.39 * enhanced_synaptic_steepness # pre-factor in the exponential synapse activation
G_c = 100.0 * 1e-12 # cell membrane conductance (S)
C = 10.0 * 1e-12 # cell membrane capacitance (F)
g_syn = 10.0 * 1e-12 # synaptic conductance (S)
g_gap = 5.0 * 1e-12 # gap junctional conductance (S)
G_c_over_C = G_c/C
g_syn_over_C = g_syn/C
g_gap_over_C = g_gap/C
E_c = [-35.0 * 1e-3 for i in indices] # -35 mV leakage potentials
V_range = 35.0 * 1e-3 # 35 mV range
relative_GABA = 0.5 # relative strength of GABA compared to Glut and Chol synapses

def main():          
    # get each cell's soma coords
    importData()
    
    # read connectivity data
    readSynapticConnectome()
    readGapConnectome()
    
    # plot the connectome in 3D
    plotConnectomeIn3D()
    
    # plot adjacency matrices
    buildAdjacency()
    plotAdjacency()

    
    # integrate the ODEs
    soln = integrateODEs(1)
    
    # plot the states of the synaptic parameters
    V_eq = findEquilibriumPotentials()
    V_vals_to_plot = np.linspace(-50 * 1e-3, 0.0 * 1e-3, num = 100, endpoint = True)
    for k in indices:
        plt.plot(V_vals_to_plot, [1.0/(1+np.exp(K*(V_vals_to_plot[i] - V_eq[k])/V_range)) for i in range(len(V_vals_to_plot))], color='r')
        
    for i in indices:
        plt.plot([float(p) for p in soln[:,i]], [1*float(p)/len(soln[:,i]) for p in range(len(soln[:,i]))], color = 'k')
    
    # calculate Jacobian
    calculateJacobian(soln)
    
    # run SVD
    if len(all_the_cell_names) > 18: # this IF statement just checks to be sure there are motor neurons in the simulation to run SVD on!
        runSVD(soln)
    
    # show the plots
    plt.show()
        
def integrateODEs(ext_on):    
    print ("\nIntegrating ODEs...")
    
    # inputs to the simulation
    t  = np.linspace(start_time, end_time, num = 500, endpoint = True)
    V_eq = findEquilibriumPotentials()
    F_ext = [ext_on * (I_ext_magnitude / C) * 1 for i in indices]
    
    if os.path.isfile("ode_soln.p") and usePickle: # delete this file if you want it to run from scratch
        soln = pickle.load(open("ode_soln.p", "rb" ))
    else:
        soln = odeint(deriv, np.array(V_eq), t, args = (F_ext,V_eq, stim_dur, stim_start),  mxstep = 5000) # system starts in equilibrium at time zero
        pickle.dump(soln, open("ode_soln.p", "wb" ))
    
    # plot the ODE solutions
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title("V_m (V)")
    plt.xlabel("Time (s)")
    plt.ylabel("Cell")
    for i in indices[::-1]:
        plt.plot(t, [i for q in range(len(t))], [float(p) for p in soln[:,i]], lw=1.5)
        
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title("V_m - V_eq (V)")
    plt.xlabel("Time (s)")
    plt.ylabel("Cell")
    for i in indices[::-1]:
        plt.plot(t, [i for q in range(len(t))], [float(p) - float(V_eq[i]) for p in soln[:,i]], lw=1.5)
        
    fig = plt.figure()
    plt.xlabel("Time (s)")
    plt.ylabel("Cell")
    plt.ylim(0,len(indices))
    ax = fig.gca()
    ax.pcolor(soln.T)
    ax.set_xticks([0, soln.T.shape[1]])
    ax.set_xticklabels([0, t[-1]])
        
    return soln
    
def deriv(vect,t, F_ext, V_eq, stim_dur, stim_start):
    '''This function defines the particular set of ODEs used in the simulation. See the documentation for odeint.'''
    
    # -45 mV reversal potential for GABA inhibitory synapses, 0 mV for excitatory
    reversal = [-45 * 1e-3 if all_the_cell_names[i] in GABA_expressing else 0.0 for i in indices] 
    V = list(vect) # prefer working with lists rather than numpy arrays internally
            
    # compute the gap junction input current        
    I_gap = [sum([g_gap_over_C * gap_nums[all_the_cell_names[j]][all_the_cell_names[i]] \
    * (V[i]-V[j]) for j in indices]) for i in indices]
    
    # compute the synaptic input current
    I_syn = [sum([g_syn_over_C * GABA_Chol_Glut_adjacency[all_the_cell_names[j]][all_the_cell_names[i]] \
    * (V[i]-reversal[j])/(1 + np.exp(K*(V[j] - V_eq[j])/V_range)) for j in indices]) for i in indices]
    
    # compute the temporal derivatives
    combined = [G_c_over_C*(E_c[i]-V[i]) - I_gap[i] - I_syn[i] + F_ext[i] \
    * (1.0 if t < (stim_start + stim_dur) and t > stim_start else 0.0) for i in indices]
    
    return np.array(combined)
    
        
def findEquilibriumPotentials():
    # -48 mV reversal potential for GABA inhibitory synapses, 0 mV for excitatory
    reversal = [-45 * 0.001 if all_the_cell_names[i] in GABA_expressing else 0.0 for i in indices] 
    conn = GABA_Chol_Glut_adjacency
    gap = gap_nums
    names = all_the_cell_names
    AList = [[None for j in indices] for i in indices]
    bList = [E_c[i] + sum([(g_syn_over_C/G_c_over_C) * conn[names[j]][names[i]] * reversal[j]/2.0 for j in indices]) for i in indices]

    for i in indices:
        AList[i][i] = 1 + sum([(g_gap_over_C/G_c_over_C) * gap[names[j]][names[i]] \
        + (g_syn_over_C/G_c_over_C) * conn[names[j]][names[i]]/2.0 for j in indices])

    for i in indices:
        for j in indices:
            if j != i:
                AList[i][j] = -1 * (g_gap_over_C/G_c_over_C) * gap[names[j]][names[i]]

    AMatrix = np.matrix(AList)
    bVector = np.matrix(bList).T # column vector
    return [float(k) for k in np.dot(AMatrix.I, bVector)] # convert to list

def calculateJacobian(soln):
    print ("Calculating Jacobian...")
    # -45 mV reversal potential for GABA inhibitory synapses, 0 mV for excitatory
    reversal = [-45 * 0.001 if all_the_cell_names[i] in GABA_expressing else 0.0 for i in indices] 
    conn = GABA_Chol_Glut_adjacency
    gap = gap_nums
    names = all_the_cell_names
    V = [np.mean(soln.T[i][len(soln.T[i])/2:]) for i in indices]
    V_eq = findEquilibriumPotentials()
    
    J = [[0 for j in indices] for i in indices]
    
    for i in indices: # here we hand-calculate the Jacobian matrix
        J[i][i] = -1 * ((sum([g_gap_over_C * gap[names[j]][names[i]] \
        + g_syn_over_C * conn[names[j]][names[i]]/(1 + np.exp(K*(V[j] - V_eq[j])/V_range)) for j in indices])) + G_c_over_C)
        
        for j in indices:
            if j != i:
                J[i][j] = g_gap_over_C * gap[names[j]][names[i]] \
                + g_syn_over_C * conn[names[j]][names[i]] * (reversal[j] - V[i]) * (K/V_range) \
                * ((1/(1+np.exp(K*(V[j] - V_eq[j])/V_range)))**2) * np.exp(K*(V[j] - V_eq[j])/V_range) 
    
    # plot the complex eigenvalues of the Jacobian matrix
    fig = plt.figure()
    J_matrix = np.matrix(J)
    eigenvalues = np.linalg.eig(J_matrix)
    reals = eigenvalues[0].real
    imags = eigenvalues[0].imag
    
    plt.title("Eigenvalues of Jacobian at the externally-forced equilbrium")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.vlines([0], -80, 80, linewidth = 5)
    plt.scatter(reals, imags, s = 85)
    
def readGapConnectome():
    for k1 in all_the_cell_names:
        gap_nums[k1] = {}
        for k2 in all_the_cell_names:
            gap_nums[k1][k2] = 0.0
    
    with open(connectome_spreadsheet, 'rU') as f:
        r = csv.reader(f)
        for row in r:
            if row[2] != "Send":
                if row[0] in all_the_cell_names and row[1] in all_the_cell_names:
                    cells_gaps[row[0]].append(row[1])
                    gap_nums[row[0]][row[1]] = float(row[3])
    
def readSynapticConnectome():
    for k1 in all_the_cell_names:
        connection_nums[k1] = {}
        for k2 in all_the_cell_names:
            connection_nums[k1][k2] = 0.0
    
    with open(connectome_spreadsheet, 'rU') as f:
        r = csv.reader(f)
        for row in r:
            if row[2] == "Send":
                if row[0] in all_the_cell_names and row[1] in all_the_cell_names:
                    cells_connections[row[0]].append(row[1])
                    connection_nums[row[0]][row[1]] = float(row[3])
                    if "GABA" in row[4] and not row[0] in GABA_expressing:
                        GABA_expressing.append(row[0])
                    if "Acetylcholine" in row[4] and not row[0] in Cholinergic:
                        Cholinergic.append(row[0])
                    if "Glutamate"  in row[4] and not row[0] in Glutamatergic:
                        Glutamatergic.append(row[0])

def getSegmentIds(cell): # from OpenWorm code
    seg_ids = []
    for segment in cell.morphology.segments:
        seg_ids.append(segment.id)

    return seg_ids

def get3DPosition(cell, segment_index, fraction_along): # from OpenWorm code
    seg = cell.morphology.segments[segment_index]
 
    end = seg.distal

    start = seg.proximal
    if start is None:
        segs = getSegmentIds(cell)
        seg_index_parent = segs.index(seg.parent.segments)
        start = cell.morphology.segments[seg_index_parent].distal

    fx = fract(start.x, end.x, fraction_along)
    fy = fract(start.y, end.y, fraction_along)
    fz = fract(start.z, end.z, fraction_along)

    return fx, fy, fz

def fract(a, b, f): # from OpenWorm code
    return a+(b-a)*f
    
def importData():
    for cellName in all_the_cell_names:
        inFile = "OpenWorm/straightenedNeuroML2/" + cellName + ".nml"

        # open the cell
        doc = loaders.NeuroMLLoader.load(inFile)
        if not doc.cells:
            sys.exit(1)
        cell = doc.cells[0]

        # find the soma coordinates
        c = get3DPosition(cell, 0, 0.5)
        if cellName == labeled_neuron:
            coords_of_labeled_neuron = c
        coords.append(c)
        cells_connections[cellName] = [c]
        cells_gaps[cellName] = [c]
    
def plotConnectomeIn3D():
    # set up the plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim3d(-30,30)
    ax.set_ylim3d(-400,400)
    ax.set_zlim3d(-30,30)
    plt.xlabel("x ($\mu$m)")
    plt.ylabel("y ($\mu$m)")
    
    # plot the synaptic connections in 3D
    k = cells_connections.keys()
    for key in k:
        s = cells_connections[key][0]    
        end_coords_list = [cells_connections[p][0] for p in cells_connections[key][1:]]
        for i in range(len(end_coords_list)):
            e = end_coords_list[i]
            p = cells_connections[key][1:][i]
            num = float(connection_nums[key][p])
            if key in GABA_expressing:
                ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color='r', linestyle='dashed', linewidth = num)
            elif key in Cholinergic:
                pass
                ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color='g', linestyle='dashed', linewidth = num/3)
            elif key in Glutamatergic:
                pass
                ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color='b', linestyle='dashed', linewidth = num/3)
            else:
                pass
                ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color='m', linestyle='dashed', linewidth = num/3)
            
    # plot the gap junctions in 3D
    k = cells_gaps.keys()
    for key in k:
        s = cells_gaps[key][0]
        end_coords_list = [cells_gaps[p][0] for p in cells_gaps[key][1:]]
        for i in range(len(end_coords_list)):
            e = end_coords_list[i]
            p = cells_gaps[key][1:][i]
            num = float(gap_nums[key][p])
            ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color='k', linestyle='dotted', linewidth = num/3)
            
    # plot the soma coordinates in 3D
    ax.plot([c[0] for c in coords], [c[1] for c in coords], [c[2] for c in coords], 'o', color='k', alpha=1)
            
    # plot the labeled neuron in 3D
    ax.plot([coords_of_labeled_neuron[0]], [coords_of_labeled_neuron[1]], [coords_of_labeled_neuron[2]], 'o', color='y',ms=10.0, alpha=1)
    
def buildAdjacency():
    # initialize the adjacency matrices
       for k1 in all_the_cell_names:
            GABA_adjacency[k1] = {}
            Chol_adjacency[k1] = {}
            Glut_adjacency[k1] = {}
            Other_adjacency[k1] = {}
            GABA_Chol_Glut_adjacency[k1] = {}
            for k2 in all_the_cell_names:
                GABA_adjacency[k1][k2] = 0
                Chol_adjacency[k1][k2] = 0
                Glut_adjacency[k1][k2] = 0
                Other_adjacency[k1][k2] = 0
                
       # populate adjacency matrices
       for k1 in all_the_cell_names:
           for k2 in all_the_cell_names:
               if k1 in GABA_expressing:
                   GABA_adjacency[k1][k2] = relative_GABA * connection_nums[k1][k2] # setting the relative strength of inhibition here
               elif k1 in Cholinergic:
                   Chol_adjacency[k1][k2] = connection_nums[k1][k2]
               elif k1 in Glutamatergic:
                   Glut_adjacency[k1][k2] = connection_nums[k1][k2]
               else:
                   Other_adjacency[k1][k2] = connection_nums[k1][k2]
               # this matrix just includes GABAergic, Cholinergic and Glutamatergic synapses
               GABA_Chol_Glut_adjacency[k1][k2] = GABA_adjacency[k1][k2] + Chol_adjacency[k1][k2] + Glut_adjacency[k1][k2] + Other_adjacency[k1][k2] 
               
def plotAdjacency():
       # plot the transmitter-specific adjacency matrices
       # note: this treats e.g. "Serotonin_Acetylcholine" in the spreadsheet as "Acetylcholine"
       
       pointSizeScaleFactor = 10
       
       # Adjacency matrix for GABA
       fig = plt.figure()
       plt.xlim(0 - pointSizeScaleFactor, len(all_the_cell_names) + pointSizeScaleFactor) 
       plt.ylim(0 - pointSizeScaleFactor, len(all_the_cell_names) + pointSizeScaleFactor)
       plt.xlabel("Input Cell")
       plt.ylabel("Ouput Cell")
       ax = fig.gca()
       ax.set_title("GABA")
       xvals = []
       yvals = []
       sizes = []
       for i in range(len(all_the_cell_names)):
           for j in range(len(all_the_cell_names)):
               if GABA_adjacency[all_the_cell_names[i]][all_the_cell_names[j]] > 0:
                   xvals.append(i)
                   yvals.append(j)
                   sizes.append(pointSizeScaleFactor*float(GABA_adjacency[all_the_cell_names[i]][all_the_cell_names[j]]))
       plt.scatter(xvals, yvals, s = sizes, c = 'r')

       # Adjacency matrix for Acetylcholine
       fig = plt.figure()
       plt.xlim(0 - pointSizeScaleFactor, len(all_the_cell_names) + pointSizeScaleFactor) 
       plt.ylim(0 - pointSizeScaleFactor, len(all_the_cell_names) + pointSizeScaleFactor)
       plt.xlabel("Input Cell")
       plt.ylabel("Ouput Cell")
       ax = fig.gca()
       ax.set_title("Acetylcholine")
       xvals = []
       yvals = []
       sizes = []
       for i in range(len(all_the_cell_names)):
           for j in range(len(all_the_cell_names)):
               if Chol_adjacency[all_the_cell_names[i]][all_the_cell_names[j]] > 0:
                   xvals.append(i)
                   yvals.append(j)
                   sizes.append(pointSizeScaleFactor*float(Chol_adjacency[all_the_cell_names[i]][all_the_cell_names[j]]))
       plt.scatter(xvals, yvals, s = sizes, c = 'g')

       # Adjacency matrix for Glutamate
       fig = plt.figure()
       plt.xlim(0 - pointSizeScaleFactor, len(all_the_cell_names) + pointSizeScaleFactor) 
       plt.ylim(0 - pointSizeScaleFactor, len(all_the_cell_names) + pointSizeScaleFactor)
       plt.xlabel("Input Cell")
       plt.ylabel("Ouput Cell")
       ax = fig.gca()
       ax.set_title("Glutamate")
       xvals = []
       yvals = []
       sizes = []
       for i in range(len(all_the_cell_names)):
           for j in range(len(all_the_cell_names)):
               if Glut_adjacency[all_the_cell_names[i]][all_the_cell_names[j]] > 0:
                   xvals.append(i)
                   yvals.append(j)
                   sizes.append(pointSizeScaleFactor*float(Glut_adjacency[all_the_cell_names[i]][all_the_cell_names[j]]))
       plt.scatter(xvals, yvals, s = sizes, c = 'b')    

       # Adjacency matrix for other neurotransmitters
       fig = plt.figure()
       plt.xlim(0 - pointSizeScaleFactor, len(all_the_cell_names) + pointSizeScaleFactor) 
       plt.ylim(0 - pointSizeScaleFactor, len(all_the_cell_names) + pointSizeScaleFactor)
       plt.xlabel("Input Cell")
       plt.ylabel("Ouput Cell")
       ax = fig.gca()
       ax.set_title("Other Transmitter (Serotonin, Dopamine, etc.)")
       xvals = []
       yvals = []
       sizes = []
       for i in range(len(all_the_cell_names)):
           for j in range(len(all_the_cell_names)):
               if Other_adjacency[all_the_cell_names[i]][all_the_cell_names[j]] > 0:
                   xvals.append(i)
                   yvals.append(j)
                   sizes.append(pointSizeScaleFactor*float(Other_adjacency[all_the_cell_names[i]][all_the_cell_names[j]]))
       plt.scatter(xvals, yvals, s = sizes, c = 'm')

       # Adjacency matrix for gap junctions
       fig = plt.figure()
       plt.xlim(0 - pointSizeScaleFactor, len(all_the_cell_names) + pointSizeScaleFactor) 
       plt.ylim(0 - pointSizeScaleFactor, len(all_the_cell_names) + pointSizeScaleFactor)
       plt.xlabel("Input Cell")
       plt.ylabel("Ouput Cell")
       ax = fig.gca()
       ax.set_title("Gap Junctions")
       xvals = []
       yvals = []
       sizes = []                
       for i in range(len(all_the_cell_names)):
           for j in range(len(all_the_cell_names)):
               if gap_nums[all_the_cell_names[i]][all_the_cell_names[j]] > 0:
                   xvals.append(i)
                   yvals.append(j)
                   sizes.append(pointSizeScaleFactor*float(gap_nums[all_the_cell_names[i]][all_the_cell_names[j]]))
       plt.scatter(xvals, yvals, s = sizes, c = 'k')
       
def runSVD(soln):
    print ("SVD analysis...")
    motor_neuron_indices = indices[18:] # the first 18 neurons are the sensory neurons and interneurons
    # Time-slices of motor neurons, starting halfway through the simulation to allow the initial transients to settle
    M = np.matrix([[k - np.mean(soln.T[i][len(soln.T[i])/2:]) for k in soln.T[i][len(soln.T[i])/2:]] for i in motor_neuron_indices]) 
    print (M)
    U, S, V = np.linalg.svd(M)
    fig = plt.figure()
    plt.xlabel("Mode number")
    plt.ylabel("Fractional contribution")
    plt.title("Singular values")
    plt.xlim(0, len(S)+1)
    plt.ylim(0,1)
    plt.scatter(range(1,len(S)+1),[p for p in S]/sum([q for q in S]), s=50)
    
    # plot the dynamics of the modes
    firstmode_vec = U[:,0]/np.linalg.norm(U[:,0])
    secondmode_vec = U[:,1]/np.linalg.norm(U[:,1])
    thirdmode_vec = U[:,2]/np.linalg.norm(U[:,2])
    
    # plot the modes as vectors
    fig = plt.figure()
    plt.title("Mode vectors")
    plt.xlabel("Motor neuron number")
    plt.ylabel("Mode weight on neuron")
    plt.plot([p[(0,0)] for p in firstmode_vec], linewidth = 3, marker = 'o', markersize = 15)
    plt.plot([p[(0,0)] for p in secondmode_vec], linewidth = 3, marker = 'o', markersize = 15)
    plt.plot([p[(0,0)] for p in thirdmode_vec], linewidth = 3, marker = 'o', markersize = 15)
    
    a1 = []
    a2 = []
    a3 = []
    for i in range(np.shape(M)[1]):
        timeslice = M[:,i]
        a1.append((timeslice.T * firstmode_vec)[(0,0)])
        a2.append((timeslice.T * secondmode_vec)[(0,0)])
        a3.append((timeslice.T * thirdmode_vec)[(0,0)])
        
    fig = plt.figure()
    plt.title("Shapes of the first three modes")
    plt.xlabel("Time (au)")
    plt.ylabel("Mode amplitude (au)")
    plt.plot(a1, linewidth = 2)
    plt.plot(a2, linewidth = 2)
    plt.plot(a3, linewidth = 2)
    plt.figure()
    plt.title("Mode phase space evolution")
    plt.xlabel("Mode")
    plt.ylabel("Mode")
    plt.plot(a1, a2, linewidth = 1)

def update_plot(i, data, scat):
    scat.set_array(data[i])
    return scat
    
if __name__ == '__main__':
    main()