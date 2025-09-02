#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 22:29:58 2025

@author: usuario
"""

#Actividad de identidad de parseval


#Calcular varianza y comparar con amplitud
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft

N = 1000
fs = N
delta_f = fs/N #resolucion espectral
Ts = 1/fs

def sen(ff, nn, amp = np.sqrt(2), ph =0, dc = 0, fs = 2):
    N = np.arange(nn)
    t = N/fs
    x = dc + amp*np.sin(2*np.pi*ff*t + ph)
    return t,x 


t,x = sen(ff=(N/4)*delta_f, nn=N, fs=fs)


var = np.var(x)

print("varianza de la funcion:", var)


#punto 2: Calcular en db el modulo cuadrado del especto y graficar. Ver densidad espectral de potencia

X= fft(x)
X_abs= np.abs(X)
X_abs_cuadrado = X_abs**2

Ff = np.arange(N)*delta_f

plt.figure(1)
plt.plot(Ff,np.log10(X_abs_cuadrado)*20, "x", label = "X abs en dB")

plt.legend()
plt.show()



#punto 3: verificar parseval
suma_modulo_cuadrado = np.sum(X_abs_cuadrado)
print(suma_modulo_cuadrado)
suma_cuadrados = np.sum(abs(X**2))
print(suma_cuadrados)

if suma_modulo_cuadrado == suma_cuadrados:
    print("se cumple parseval")
else:
    print("no se cumple")
