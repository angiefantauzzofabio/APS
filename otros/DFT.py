#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 20:39:09 2025

@author: usuario
"""
import numpy as np
import matplotlib.pyplot as plt

def generador_de_señales(vmax, dc, f, ph, N, fs):
 '''
  PARAMETROS:
  vmax:amplitud max de la senoidal [Volts]
  dc:valor medio [Volts]
  f:frecuencia [Hz]
  ph:fase en [rad]
  N:cantidad de muestras
  fs:frecuencia de muestreo [Hz]
 '''

 Ts = 1/fs
 tiempo = np.arange(0,N*Ts,Ts)
 w0 = 2*np.pi*f
 x = vmax*np.sin(w0* tiempo + ph)+dc
 return tiempo,x



tt, yy = generador_de_señales(1, 0, 1, 0, 10, 10)


def DFT(x):
    N = len(x)
    res = np.zeros(100, dtype=np.complex128())
    for k in range (N):
        for n in range (N):
            res[k] += (x[n] * np.exp(-1j * k * 2 * np.pi * n / N))
    
    print(res)
    return res

DFT(yy)


plt.figure(1)
plt.stem(np.abs(yy)) #aca vemos el modulo de x 
#la fase la calculamos con np.angle
plt.show()

#esto lo podes graficae en muestras, en frecuencias o en 2pi. Si quiero pasar de muestras a frecuencia multiplico por delta f que es (k*2pi*n)/n

#esto esto es para chequear, fft.fft calcula la transformada discreta de fourier. x
res = np.fft.fft(yy)
print("CHEQUEAR",res)


