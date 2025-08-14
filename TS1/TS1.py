#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 18:49:40 2025

@author: Angelina Fantauzzo Fabio

"""

import numpy as np
import matplotlib.pyplot as plt

n= np.arange(0,2*np.pi,0.01) #secuencia de puntos parte de la funcion evaluados en la funcion seno


#paso a radianes
N = len(n)
print("esto es N:",N)
r=n*(2*np.pi) #la formula es r=n.(2pi/N)
print("esto son los radianes:",r)
f = 2000

fs = 2000
w = f*2*np.pi #esta es la frecuencia angular 

#fs = 1/Ts
#Nos queda definir el fs, por el teorema de muestreo nos dice que tiene que ser ms grande que la frecuencia del ancho de banda 
#La frecuencia del ancho de banda es 
#el delta f nos vincula N y fs. 
#Si elijo fs=N , delta f =1 y N.Ts =1s 
#los ciclos, ajustandolo as, me temrinan diciendo cuantos hertz tengo. Si tengo dos ciclos 2hz, un ciclo 1hz (ciclos son las ondas, los picos)



fun = np.sin(r)




plt.plot(n, fun, color = 'red', marker='*')
plt.title("Un periodo del seno")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

    
#tt = np.arange(0,N*(1/fs,fs/N) #el segundo parametro es 1 segundo y el tercero tmbien es 1 xq fs=N=1000

tt= np.arange(0,1,0.001)

fun2= np.sin(tt)

plt.plot(1000, fun2, color = 'red', marker='*')
plt.title("Un periodo del seno")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

    
    
    





