from __future__ import division
from builtins import str
from builtins import range
from past.utils import old_div
from scipy import *
from scipy.integrate import *
from scipy.optimize import *
from scipy import signal
import time
from numpy.random import seed,poisson

from pylab import *
ion()

################# Main Code: you call upon main_model for it to work ###########
def main_model(x): #x[] is an input vector of the form [input current, Cl-K channels, Period of sine signal]
    
    seed()
    
    start=time.time() #Checks time of simulation
    
    I = 32  #INPUT CURRENT x[0] is user-defined
    #main_x turns into x[] in full_model function
    main_x= [-75.03,1.13e-5,1.13e-5,1.0,2.04e-7,0.0,3.48e-5,0.0,0.312,1.24e-9,x[2],I] 
    nCa = 1000 #x[1] calcium channels
    nBK = x[1] #potassium channels
    nKB = x[1] #potassium channels
    tf = 7000 #100000 original duration of simulation
    ts = [0]
    t = 0
    tau = 1.0
    vals = [main_x]
    
    
    while t<tf:
        rates = full_model(main_x,0) #Call of fulL_model
        dt = min([0.05/max(abs(rates[1:])),0.25]) #time step is defined
        Vs = odeint(full_model,main_x,linspace(t,t+dt,2)) #differential equation solved
        t = t+dt #time 
        Ca1 = main_x[1]
        Ca2 = main_x[2]
        BK = main_x[5]
        KB = main_x[7]
        V = main_x[0]
        BKrate = dt*nBK*tau*BK1(V,BK)
        Carate = dt*nCa*tau*CaV(V,[Ca1,Ca2,0.5])
        Ca1flip = poisson(abs(Carate[0]))*sign(Carate[0])
        Ca2flip = poisson(abs(Carate[1]))*sign(Carate[1])
        KBrate = dt*nKB*tau*SLO12(V,KB)
        BKflip = poisson(abs(BKrate))*sign(BKrate)
        KBflip = poisson(abs(KBrate))*sign(KBrate)
        main_x = Vs[-1,:]
        main_x[1] = old_div(Ca1flip,nCa)+Ca1
        main_x[2] = old_div(Ca2flip,nCa)+Ca2
        main_x[5] = old_div(BKflip,nBK)+BK
        main_x[7] = old_div(KBflip,nKB)+KB
        ts.append(t)
        vals.append(main_x)
        
    vals = array(vals)
    ts2,xs2 = downsample(ts,vals[:,0],0.25)
    vals2 = ones([len(ts2),len(main_x)])-1
    vals2[:,0] = xs2
    
    for i in range(1,len(vals[0,:])):
        ts2,xs2 = downsample(ts,vals[:,i],0.25)
        vals2[:,i] = array(xs2)
        
    end = time.time()
    print('Elapsed time for code to run (seconds):')
    print (end-start)
    
    
    return ts2,vals2

#Sub function where the real model is defined
def full_model(x,t):
    
    cap = 1.5e-3 ### from qiang's measurement
    
    ####CHANGE WT SHK1###
    gCa = 0.13 #0.1 for wt model #0.13 for shk-1 model ### fit to calcium channel voltage clamp
   
    ####CHANGE WT SHK1###
    gK = 0.0 #1.5 for wt #0.0 for shk-1 ### fit to SHK-1 voltage clamp
   
   
    gK2 = 0.8 #gKIR ### tuned to give resting potential around -75 mV
  
    
    ####CHANGE WT SHK1###
    gK3 = 0.77 #gka #0.3*1.3 for wt #0.77 for shk1 ### tuned to make voltage jump after current step the right height
   
    ####CHANGE WT SHK1###
    gK4 = 2.0 #1.0 for wt #2.0 for shk1 ### fit from combined slo-1 and slo-2 currents
   
    ####CHANGE WT SHK1###
    gK5 = 1.98 #0.7*1.3 wt #1.98 shk1
   
    gK6 = 0.0 #0.9 #0.3?
    gK7 = 0.1
    gL = 0.25 #0.4 ### fit to leak slope from calcium channel voltage clamp
    
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
    
    z = old_div(Ca,(Ca+10.0))  ### calicum-sensitive potassium current
    
    #CURRENT INJECTED TO CLAMP
    # I = x[-1]*sin(old_div(2*pi*t,x[10]))
    # I = x[-1]*(1+signal.square(2*pi*(t-1000.)/10000.,0.5))/2.0
    I = x[-1]
    # I = 0.5*(sign(t-25000+x[10])+1)*x[-1]*(t-25000+x[10])/x[10]
    
    y = ones([12,])-1.0
    y[0] = old_div((I-ICa-(gK*SHK+gK3*minf(V,-42,5)*(1-BK)+gK4*SLO+gK5*KB+gK6+gK7*SLO2)*(V-vK)-IL-gK2*Kir),cap)
    yCa = CaV(V,[Ca1,Ca2,Ca3])
    y[1] = yCa[0]
    y[2] = yCa[1]
    y[3] = yCa[2]
    y[4] = SHK1(V,SHK)
    y[5] = BK1(V,BK)
    y[6] = SLO3(V,SLO)
    y[7] = SLO12(V,KB)
    y[8] = (-1.25*ICa-old_div(194.75*x[8],5))/10000.0  ### calcium ion concentration in nM, setting Caeq to 5nm
    y[9] = SLO4(V,SLO2)
           
    
    return y

###################Sub-functions for the full_model###########################

def minf(V,Va,Vb):

    return 0.5*(1 + tanh(old_div((V-Va),Vb)))

def tau(V,Va,Vb):
    
    return 1.0/cosh(old_div((V-Va),(2*Vb)))

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
    y[0] = old_div((old_div(m1*m2,mx)-x[0]),T1)-old_div(m1*m2*x[1],(mx*T1))-old_div(x[0],(2*T2))+old_div(x[1],(2*T2))
    y[1] = old_div((x[0]-x[1]),(2*T2))
    
    return y

def SHK1(V,x):
    
    ### SHK-1 function, fit to subtractino of shk-1 current from wt at 0 Ca
    
    T = 30.0
    m = minf(V,2,10)
    
    return old_div((m-x),T)

def BK1(V,x):
    
    ### inactivation of slow potassium current. fit to get delay to bursting and baseline
    
    T = 1200.0*1.5 # 1200
    m = minf(V,-42,5) #m = minf(V,-40,5)
    
    return old_div((m-x),T)

def SLO12(V,x):
    
    TKL = 18000*1.5
    TKH = 2000*1.5
    vtk1 = -52. #-50
    vtk2 = 20.
    T = TKL+(TKH-TKL)*0.5*(1+old_div(tanh(V-vtk1),vtk2))
    m = minf(V,-32.,2.) #minf(V,-30,4)
    
    return old_div((m-x),T)

def SLO3(V,x):
    
    m = minf(V,13,20)
    T = 1000 # 1200
    
    return old_div((m-x),T)

def SLO4(V,x):
    
    m = minf(V,-25,5)
    T = 1000.0 # 600
    
    return old_div((m-x),T)
    
################### Sub functions for the run_model ##########################

def downsample(ts,xs,step):
    
    dts = array(ts[1:])-array(ts[0:-1])
    maxL = int(old_div(step,min(dts)))+1
    first = 0
    num = int(old_div(ts[-1],step))
    ts2 = linspace(0,ts[-1],num)
    ts3 = []
    xs2 = []
    
    for t in ts2:
        ind = findInd(ts[first:first+maxL],t)
        first = first+ind
        ts3.append(ts[first])
        xs2.append(xs[first])
    
    return ts3,xs2

def findInd(vec,val):
    
    c1 = array(vec)-val
    
    return argmin(abs(c1))


x=[4, 1000, 0.10]
[result1,result2]= main_model(x) 
out32=result2[:,0]

plot(result1,result2[:,0],label = 'I=32pamp')
ylabel('vM (mV)')
xlabel('Time (ms)')
legend(loc='lower right')
show()
plot(result1,result2[:,2])
show()
plot(result1,result2[:,3])
show()
plot(result1,result2[:,4])
show()
plot(result1,result2[:,5])
show()
plot(result1,result2[:,6])
show()
plot(result1,result2[:,7])
show()
plot(result1,result2[:,8])
show()
plot(result1,result2[:,9])
show()