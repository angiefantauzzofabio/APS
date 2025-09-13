#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 19:29:05 2025

@author: usuario
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
from scipy.signal import windows
from numpy.fft import fftfreq


def eje_temporal(N,fs):
    Ts = 1/fs
    t_final = N*Ts
    tt= np.arange(0,t_final,Ts)
    return tt

def seno(tt,frec,amp,fase = 0, v_medio=0):
    xx = amp*np.sin(2*np.pi*frec*tt + fase) + v_medio
    return xx


SNR = 50 #SNR es db
amp_0= np.sqrt(2) #en Volts
N = 1000
fs = N # en Hertz
deltaf = fs/N # En Hertz, resolucion espectral.
mu = 0
realizaciones = 200  
fr = np.random.uniform(2,-2,size=realizaciones) #Estas son nuestras frecuencias aleatornias, es uns distribucion normal

nn = np.arange(N)
ff = (np.arange(N)*deltaf).flatten()
tt = eje_temporal(N = N, fs=fs).flatten()

frecuencia = ((N/4) +fr)*deltaf
#Quiero armar una matriz de de señales porque tengo una lista de frecuencias randoms. 
s_1 = amp_0*np.sin(2*np.pi*fs*tt)


# Convertimos tt en columna y repetimos R veces
T = np.tile(tt.reshape(N, 1), (1, realizaciones))  # shape: (N, R)
print("Matriz T:", T)
print("Shape:", T.shape)  # (1000, 200)


# Convertimos fr en fila y repetimos N veces
FR = np.tile(fr.reshape(1, realizaciones), (N, 1))  # shape: (N, R)
print("Matriz FR:", FR)
print("Shape:", FR.shape)  # (1000, 200)


S = amp_0 * np.sin(2 * np.pi * FR * T)  # shape: (1000, 200)
print("Matriz S:", S)
print("Shape:", S.shape)




  


potencia_ruido = amp_0**2/(2*10**(SNR/10)) #esta seria la varianza tmb
print("Potencia/Varianza de ruido:", potencia_ruido)
desvio_estandar = np.sqrt(potencia_ruido)

ruido = np.random.normal(mu, desvio_estandar, N) #este es el ruido 
varianza_ruido = np.var(ruido)
print("Potencia/Varianza de ruido:", varianza_ruido)


#Aca estamos instrumentando, deberia dar casi lo mismo como nos pasa. esta correcto


x_1 = s_1 + ruido  
X_1 = (fft(x_1))*(1/N)  #Ese 1/N lo pongo para ESCALARLO, notar que ahi me quedo a cero dB en la linea como deberia ser porque si use amp raiz cuadrada de 2 va a dar eso.
S_1 = fft(s_1)*(1/N)
RUIDO = fft(ruido)*(1/N)

plt.figure()
plt.plot(ff,20*np.log10(np.abs(X_1)), label = 'Señal con ruido', color = 'blue')
plt.plot(ff, 20*np.log10(np.abs(S_1)),label = 'Señal limpia', color = 'black')
plt.plot(ff, 20*np.log10(np.abs(RUIDO)),label = 'Ruido', color = 'red')
plt.grid(True)
plt.legend()
plt.show()

#Notar que el ruido es insignificante


plt.figure()
plt.plot(ff, (10*np.log10(2*np.abs(X_1)**2)),label = 'DESNIDAD ESPECTRAL', color = 'orange') #Aca multiplioque por dos para caliblrarlo, para escalarlo, lo subo 3db y eso lo logro multiplicando por dos el modulo cuadrado
plt.grid(True)
plt.xlim(0, fs/2)
plt.legend()
plt.show()

#Subir 3dB es duplicar en potencia, porque 3dB es raiz de 3
















