#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 20:16:05 2025

@author: usuario
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

#armo un array con las muestras
m = 8 #cantidad de muestras
X= np.zeros(m)
X[3] =1
X[4] =1

print(X)
plt.plot(X)
plt.show()

plt.stem(X)
plt.show()


