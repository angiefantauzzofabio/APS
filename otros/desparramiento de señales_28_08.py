#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 19:21:38 2025

@author: usuario
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft

N = 1000
fs = N
delta_f = fs/N #resolucion espectral
Ts = 1/fs

def sen(ff, nn, amp = 1, ph =0, dc = 0, fs = 2):
    N = np.arange(nn)
    t = N/fs
    x = dc + amp*np.sin(2*np.pi*ff*t + ph)
    return t,x 

t1,x1 = sen(ff=(N/4)*delta_f, nn=N, fs=fs)
t2,x2 = sen(ff=((N/4)+1)*delta_f, nn=N, fs=fs)

#FFTs
X1= fft(x1)
X1_abs= np.abs(X1)
X1_ang= np.angle(X1)

X2= fft(x2)
X2_abs = np.abs(X2)
X2_ang = np.angle(X2)

Ff = np.arange(N)*delta_f #eje de frecuencias

frecuencia_medio= ((N/4)*delta_f ) + (((N/4)*delta_f)+1) #N/4 + 0.5 (si grafico los tres hay desparamiento)
_, X_medio = sen(frecuencia_medio, nn=N, fs=fs)
X_medio_abs = np.abs(X_medio)
X_medio_ang = np.angle(X_medio)


plt.figure(1)

"""
------asi grafico no en db, en frecuencia y amplitud---
plt.plot(Ff,X1_abs, "x", label = "X1 abs")
plt.plot(Ff,X2_abs, "o", label = "X2 abs")
plt.plot(Ff,X_medio_abs , "*", label = "X medio abs")
plt.title("FFT")
plt.xlabel("Frecuencia")
plt.ylabel("Amplitud")
"""

#graficado en DB
plt.plot(Ff,np.log10(X1_abs)*20, "x", label = "X1 abs en dB")
plt.plot(Ff,np.log10(X2_abs)*20, "o", label = "X2 abs en dB")
plt.plot(Ff,np.log10(X_medio_abs)*20 , "*", label = "X medio abs en dB")
plt.title("FFT")
plt.xlabel("Frecuencia")
plt.ylabel("dB")


plt.xlim([0,fs/2]) #con esta linea lo limito


plt.legend()
plt.show()


"""
Porque vemos este grafico?
lo tengo anotado en mi tablet


"""










