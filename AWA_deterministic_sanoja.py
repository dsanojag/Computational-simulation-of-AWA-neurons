from scipy import *
from scipy.integrate import *
from scipy.optimize import *
from scipy import signal
import time
from numpy.random import seed,poisson

from pylab import *
ion()

factor1 = 1.0 #1.3 for stochastic simulation
factor2 = 1.0 #1.5 for stochastic simulation

################# Main Code: you call upon run_deterministic for it to work ###########
def run_deterministic(current):

    ### make sure to set factor1 and factor2 to 1 at the top of the code
    ### and set full_model to have constant current input

    init1 = [-70,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0,0]  ### random initial condition with zero current
    times = linspace(0,10000,100000)
    vs = odeint(full_model,init1,times) ### integrate for ten seconds to get proper zero current steady-state
    init2 = vs[-1,:] 
    init2[-1] = current ### set input current
    times2 = linspace(0,5000,50000) 
    vs2 = odeint(full_model,init2,times2)  ### integrate 5 second current pulse
    
    return vs2
#####################################################################

def minf(V,Va,Vb):

    return 0.5*(1 + tanh((V-Va)/Vb))

def tau(V,Va,Vb):
    
    return 1.0/cosh((V-Va)/(2*Vb))

def CaV(V,x):
    
    ### calcium channel fit to voltage clamp with potassium blockers and 0K+ buffer
    ### current has fast activation (Vm1/Vm2) and slower partial inactivation (Vm3/Vm4)
    ### Inactivation time scale also fit from data (T2/23/24)
    
    Vm1 = -21.6
    Vm2 = 9.17
    Vm3 = 16.2
    Vm4 = -16.1

    T1 = 1.0
    #T2 = 1.0/50.0
    T2 = 80*tau(V,23,24)

    mx = 3.612/4
    
    m1 = minf(V,Vm1,Vm2)**1
    m2 = minf(V,Vm3,Vm4)**1
    
    y = ones([3,])-1
    y[0] = (m1*m2/mx-x[0])/T1-m1*m2*x[1]/(mx*T1)-x[0]/(2*T2)+x[1]/(2*T2)
    y[1] = (x[0]-x[1])/(2*T2)
    
    return y

def CaV2(V,x):
    
    ### copy of calcium channel function without quick self-inactivaton. for testing.
    
    Vm1 = -19.6
    Vm2 = 9.17
    
    Vm3 = 19.2
    Vm4 = -16.1

    T1 = 1.0
    T2 = 80*tau(V,23,24)
    
    mx = 3.612
    
    m1 = minf(V,Vm1,Vm2)**1
    m2 = minf(V,Vm3,Vm4)**1

    y = (m1*m2/mx-x[0])/T1
    
    return y

def SHK1(V,x):
    
    ### SHK-1 function, fit to subtractino of shk-1 current from wt at 0 Ca
    
    T = 30.0
    m = minf(V,2,10)
    
    return (m-x)/T

def BK1(V,x):
    
    ### inactivation of slow potassium current. fit to get delay to bursting and baseline
    
    T = 1200.0*factor2 #*1.5 # 1200
    m = minf(V,-42,5) #m = minf(V,-40,5)
    
    return (m-x)/T

def SLO12(V,x):
    
    #T = 2350.0 ## rough estimate from data
    TKL = 18000*factor2 #*1.5
    TKH = 2000*factor2 #*1.5
    vtk1 = -52. #-50
    vtk2 = 20.
    #T = 2500.0 ## rough estimate from data
    T = TKL+(TKH-TKL)*0.5*(1+tanh(V-vtk1)/vtk2)
    m = minf(V,-32.,2.) #minf(V,-30,4)
    
    return (m-x)/T

def SLO3(V,x):
    
    m = minf(V,13,20)
    T = 1000 # 1200
    
    return (m-x)/T

def SLO4(V,x):
    
    m = minf(V,-25,5)
    T = 1000.0 # 600
    
    return (m-x)/T
    
def full_model(x,t):
    
    cap = 1.5e-3 ### from qiang's measurement
    gCa = 0.1 #0.3 #0.3 ### fit to calcium channel voltage clamp
    gK = 1.5 #1.5 #0.5 ### fit to SHK-1 voltage clamp
    #gK = 0.0
    gK2 = 0.8 #0.5 #0.6 #16.5 #33.0 ### tuned to give resting potential around -75 mV
    #gK2 = 0.0
    gK3 = 0.3*factor1 #*1.3#1.1 #0.95 #0.7 #0.1 #10.0 ### tuned to make voltage jump after current step the right height
    gK4 = 1.0 #1.0 #0.6 #10.0 #6.4 ### fit from combined slo-1 and slo-2 currents
    gK5 = 0.7*factor1 #*1.3 #1.4 #1.2 #0.3 #0.3?
    gK6 = 0.0 #0.9 #0.3?
    gK7 = 0.1
    gL = 0.25 #0.4 ### fit to leak slope from calcium channel voltage clamp
    #gL = 1.0
    vL = -65.0 ### fit to leak slope from calcium channel voltage clamp
    vCa = 120.0 ### canonical
    vK = -84.0 ### canonical
    gKI = 5.0 ### fit to get spiking baseline
    fac = 0.4  ## inactivation of calcium
    

    V = x[0]
    Ca1 = x[1]
    Ca2 = x[2]
    Ca3 = x[3]
    SHK = x[4]
    BK = x[5]
    SLO = x[6]
    KB = x[7] #KB
    Ca = x[8]
    SLO2 = x[9]
    
    
    ICa = gCa*(Ca1+fac*Ca2)*(V-vCa)
    IL = gL*(V-vL) #### Linear leak current
    
    Kir = -log(1+exp(-0.2*(V-vK-gKI)))/0.2+gKI
    
    z = Ca/(Ca+10.0)  ### calicum-sensitive potassium current
    
    
    I = x[-1]  #### constant current input, use this for determinsitic sim
    
    y = ones([12,])-1.0
    y[0] = (I-ICa-(gK*SHK+gK3*minf(V,-42,5)*(1-BK)+gK4*SLO+gK5*KB+gK6+gK7*SLO2)*(V-vK)-IL-gK2*Kir)/cap
    yCa = CaV(V,[Ca1,Ca2,Ca3])
    y[1] = yCa[0]
    y[2] = yCa[1]
    y[3] = yCa[2]
    y[4] = SHK1(V,SHK)
    y[5] = BK1(V,BK)
    y[6] = SLO3(V,SLO)
    y[7] = SLO12(V,KB)
    y[8] = (-1.25*ICa-194.75*x[8]/5)/10000.0  ### calcium ion concentration in nM, setting Caeq to 5nm
    y[9] = SLO4(V,SLO2)
           
    
    return y

###TO ACTIVATE CODE input

outs18=run_deterministic(24) 
plot(outs18[:,0],label = 'I=18pamp')
ylabel('vM (mV)')
xlabel('Time (ms)')
legend(loc='lower right')
