#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Armar la señal cion el SRN 

"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
from scipy.signal import windows
from numpy.fft import fftfreq



fs = 1000
N = fs
Ts = 1/fs
deltaf = (2*np.pi)/N


a0 = np.sqrt(2)
omega0 = 2 * np.pi *fs/4
fr = 0 #aca es cero, lo fijamos ahi, pero sino es una distribucion uniforme entre dos numeros 
N = 1000
omega1 = omega0 + fr * ((2*np.pi)/N)

snr = np.random.uniform(3, 10) #el SNR va entre 3 y 10 que me dice la consignas

potencia_snr = a0**2/(2*10**(snr/10))
print(f'La potencia del SNR: {potencia_snr}')

mu = 0 #la media
sigma = 10**(-snr/10)#varianza, se lo paso con los valores del SNR

desvio_estandar = np.sqrt(sigma)


n = np.arange(N) #este es el vector tiempo, para el seno
s = a0*np.sin(omega0*n) # esta es la señal

sigma_s = np.var(s)
print("Varianza de s:",sigma_s)

R = np.random.normal(mu, desvio_estandar, N) #este es el ruido 
sigma_ruido = np.var(R)
sigma2_ruido = sigma_s / (10**(snr/10))   # varianza del ruido, esta es la formula general
print("Varianza de ruido:", sigma_ruido, sigma2_ruido)

x = s + R  #esta es mi x que es mi señal mas el ruido 

sigma_x =  np.var(x)
print("Varianza de x:", sigma_x)




freqs = fftfreq(N, Ts) * 2*np.pi   # eje de frecuencias en rad/muestra




#Primer ventana: Rectangular
ventana1 = windows.triang(N)   
xw1 = x * ventana1
Xw1 = fft(xw1)
Xw1_mag = np.abs(Xw1)

a1_hat_1 = np.max(Xw1_mag)   
print("Estimador de amplitud a1:", a1_hat_1)


Omega1_hat_1 = freqs[np.argmax(Xw1_mag)] 
print("Estimador de frecuencia Ω1:", Omega1_hat_1)

plt.figure(figsize=(10,5))
plt.plot(freqs, Xw1_mag)
plt.title("Espectro con ventana rectangular")
plt.xlabel("Frecuencia [rad/muestra]")
plt.ylabel("|Xw(Ω)|")
plt.grid(True)
plt.show()



#Sergunda ventana: Flopp
ventana2 = windows.flattop(N)   # elegí flattop como ejemplo
xw2 = x * ventana2
Xw2 = fft(xw2)
Xw2_mag = np.abs(Xw2)

a1_hat_2 = np.max(Xw2_mag)   # la amplitud es el máximo del espectro
print("Estimador de amplitud a1:", a1_hat_2)


Omega1_hat_2 = freqs[np.argmax(Xw2_mag)]  # frecuencia donde ocurre el máximo
print("Estimador de frecuencia Ω1:", Omega1_hat_2)

plt.figure(figsize=(10,5))
plt.plot(freqs, Xw2_mag)
plt.title("Espectro con ventana flattop")
plt.xlabel("Frecuencia [rad/muestra]")
plt.ylabel("|Xw(Ω)|")
plt.grid(True)
plt.show()



#Tercer ventana: blackmanharris
ventana3 = windows.blackmanharris(N)   # elegí flattop como ejemplo
xw3 = x * ventana3
Xw3 = fft(xw3)
Xw3_mag = np.abs(Xw3)

a1_hat_3 = np.max(Xw3_mag)   # la amplitud es el máximo del espectro
print("Estimador de amplitud a1:", a1_hat_3)


Omega1_hat_3 = freqs[np.argmax(Xw3_mag)]  # frecuencia donde ocurre el máximo
print("Estimador de frecuencia Ω1:", Omega1_hat_3)

plt.figure(figsize=(10,5))
plt.plot(freqs, Xw3_mag)
plt.title("Espectro con ventana Blackmanharris")
plt.xlabel("Frecuencia [rad/muestra]")
plt.ylabel("|Xw(Ω)|")
plt.grid(True)
plt.show()


#Cuarta ventana: blackmanharris
ventana4 = windows.bohman(N)  # elegí flattop como ejemplo
xw4 = x * ventana4
Xw4 = fft(xw4)
Xw4_mag = np.abs(Xw4)

a1_hat_4 = np.max(Xw4_mag)   # la amplitud es el máximo del espectro
print("Estimador de amplitud a1:", a1_hat_4)


Omega1_hat_4 = freqs[np.argmax(Xw4_mag)]  # frecuencia donde ocurre el máximo
print("Estimador de frecuencia Ω1:", Omega1_hat_4)

plt.figure(figsize=(10,5))
plt.plot(freqs, Xw4_mag)
plt.title("Espectro con ventana Bohman")
plt.xlabel("Frecuencia [rad/muestra]")
plt.ylabel("|Xw(Ω)|")
plt.grid(True)
plt.show()



#En el TS3 el Na es el ruido



