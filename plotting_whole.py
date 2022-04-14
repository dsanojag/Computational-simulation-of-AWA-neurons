#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 20:31:05 2022

@author: diegosanoja
"""
from scipy import *
import matplotlib.pyplot as plt
from pylab import *

plot(result1,out24,'-r', label = 'I=24pamp')
plot(result1,out28,'-y',label = 'I=28pamp')
plot(result1,out32,'-g',label = 'I=32pamp')
ylabel('vM (mV)')
xlabel('Time (ms)')
legend(loc='lower right')
show()
