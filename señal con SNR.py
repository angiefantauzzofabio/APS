#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Armar la señal cion el SRN 

"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft


fs = 1000
N = fs
Ts = 1/fs
deltaf = (2*np.pi)/N


a0 = np.sqrt(2)
omega0 = fs/4
fr = 0 #aca es cero, lo fijamos ahi pero sdino es una distribucion uniforme entre dos numeros
N = 1000
omega1 = omega0 + fr * ((2*np.pi)/N)

snr = np.random.uniform(3, 10) #el SNR va entre 3 y 10 que me dice la consignas

mu = 0 #la media
sigma = 10**(-snr/10)#varianza, se lo paso con los valores del SNR


n = np.arange(N) #este es el vector tiempo, para el seno
s = a0*np.sin(omega0*n) # esta es la señal
sigma_s = np.var(s)
print("Varianza de s:",sigma_s)

R = np.random.normal(mu, sigma, N) #este es el ruido 
sigma_ruido = np.var(R)
print("Varianza de ruido:", sigma_ruido)

x = s + R  #esta es mi x que es mi señal mas el ruido 

sigma_x =  np.var(x)
print("Varianza de x:", sigma_x)

#falta la ventana, las 4 ventanas del tp. despues transformamos 

X= fft(x)
X_abs= np.abs(X)
X_abs_cuadrado = X_abs**2


frecuencias = np.arange(N)*deltaf #este es el eje x


plt.figure()
plt.plot(frecuencias,np.log10(X_abs_cuadrado)*10) #espectro en potencia
plt.show()






