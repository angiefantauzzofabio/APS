#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 13:50:54 2025

@author: usuario
"""

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


t1,x1 = sen(ff=(N/4 )*delta_f, nn=N, fs=fs)
t2,x2 = sen(ff=((N/4) + 0.25)*delta_f, nn=N, fs=fs)
t3,x3 = sen(ff=((N/4) + 0.5)*delta_f, nn=N, fs=fs)



#punto 2: Calcular en db el modulo cuadrado del especto y graficar. Ver densidad espectral de potencia


X1= fft(x1)*(1/N)
X1_abs= np.abs(X1)
X1_abs_cuadrado = X1_abs**2

X2= fft(x2)*(1/N)
X2_abs= np.abs(X2)
X2_abs_cuadrado = X2_abs**2

X3= fft(x3)*(1/N)
X3_abs= np.abs(X3)
X3_abs_cuadrado = X3_abs**2

Ff = np.arange(N)*delta_f 


plt.figure()

# X1
plt.subplot(3,1,1)
plt.plot(Ff, 10*np.log10(X1_abs_cuadrado), color="blue", label="Densidad espectral de X1")
plt.xlim([0,fs/2])
plt.ylabel("Potencia [dB]")
plt.legend()
plt.grid(True)

# X2
plt.subplot(3,1,2)
plt.plot(Ff, 10*np.log10(X2_abs_cuadrado), color="red", label="Densidad espectral de X2")
plt.xlim([0,fs/2])
plt.ylabel("Potencia [dB]")
plt.legend()
plt.grid(True)

# X3
plt.subplot(3,1,3)
plt.plot(Ff, 10*np.log10(X3_abs_cuadrado), color="green", label="Densidad espectral de X3")
plt.xlim([0,fs/2])
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Potencia [dB]")
plt.legend()
plt.grid(True)

plt.suptitle("Densidad espectral de potencia de X1, X2 y X3", fontsize=14)
plt.tight_layout(rect=[0,0,1,0.96])
plt.show()

#Graficos individuales

plt.figure() 
plt.plot(Ff,10*np.log10(X1_abs_cuadrado), label = "Densidad espectral de X1", color = "blue") #espectro en potencia 
plt.xlim([0,fs/2]) #fs/2 es la frecuencia de nyquist 
plt.legend() 
plt.grid(True) 
plt.show() 

plt.figure() 
plt.plot(Ff,10*np.log10(X2_abs_cuadrado), label = "Densidad espectral de X2", color = "red") #espectro en potencia 
plt.xlim([0,fs/2]) #fs/2 es la frecuencia de nyquist 
plt.legend() 
plt.grid(True) 
plt.show() 

plt.figure() 
plt.plot(Ff,10*np.log10(X3_abs_cuadrado), label = "Densidad espectral de X3", color = "green") #espectro en potencia 
plt.xlim([0,fs/2]) #fs/2 es la frecuencia de nyquist 
plt.legend() 
plt.grid(True) 
plt.show()



#Punto 2

energia_tiempo_1 = np.sum(np.abs(x1)**2) / N
energia_frec_1   = np.sum(np.abs(X1)**2)
if energia_tiempo_1 == energia_frec_1:
    print("se cumple parseval")
else:
    print("no se cumple")
    
energia_tiempo_2 = np.sum(np.abs(x2)**2) / N
energia_frec_2   = np.sum(np.abs(X2)**2)
if energia_tiempo_2 == energia_frec_2:
    print("se cumple parseval")
else:
    print("no se cumple")

energia_tiempo_3 = np.sum(np.abs(x3)**2) / N
energia_frec_3   = np.sum(np.abs(X3)**2)
if energia_tiempo_3 == energia_frec_3:
    print("se cumple parseval")
else:
    print("no se cumple")



delta_f_padding = fs / (9*N)
Ff_padding = np.arange(9*N)*delta_f_padding

x1_padding = np.zeros(9*N)
x1_padding[:len(x1)] = x1
#ahora transformo, paso del espectro de tiempo a frecuencia
X1_padding_fft = fft(x1_padding)
X1_padding_fft_modulo = np.abs(X1_padding_fft)**2 #densidad espectral de potencia
#creo el eje x con las frecuencias pero con 10N para que no me tire el problema de dimensiones

x2_padding = np.zeros(9*N)
x2_padding[:len(x2)] = x2
#ahora transformo, paso del espectro de tiempo a frecuencia
X2_padding_fft = fft(x2_padding)
X2_padding_fft_modulo = np.abs(X2_padding_fft)**2 #densidad espectral de potencia
#creo el eje x con las frecuencias pero con 10N para que no me tire el problema de dimensiones

x3_padding = np.zeros(9*N)
x3_padding[:len(x3)] = x3
#ahora transformo, paso del espectro de tiempo a frecuencia
X3_padding_fft = fft(x3_padding)
X3_padding_fft_modulo = np.abs(X3_padding_fft)**2 #densidad espectral de potencia
#creo el eje x con las frecuencias pero con 10N para que no me tire el problema de dimensiones


plt.figure()

# X1 con zero padding
plt.subplot(3,1,1)
plt.plot(Ff_padding, 10*np.log10(X1_padding_fft_modulo), label="X1 con zero padding", color="blue")
plt.xlim([0,fs/2])
plt.ylabel("Potencia [dB]")
plt.legend()
plt.grid(True)

# X2 con zero padding
plt.subplot(3,1,2)
plt.plot(Ff_padding, 10*np.log10(X2_padding_fft_modulo), label="X2 con zero padding", color="red")
plt.xlim([0,fs/2])
plt.ylabel("Potencia [dB]")
plt.legend()
plt.grid(True)

# X3 con zero padding
plt.subplot(3,1,3)
plt.plot(Ff_padding, 10*np.log10(X3_padding_fft_modulo), label="X3 con zero padding", color="violet")
plt.xlim([0,fs/2])
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Potencia [dB]")
plt.legend()
plt.grid(True)



#Graficos individuales 

plt.figure()
plt.plot(Ff_padding, 10*np.log10(X1_padding_fft_modulo), label="X1 con zero padding")
plt.xlim([0,fs/2])
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Potencia [dB]")
plt.title("Espectro de la señal con Zero Padding")
plt.legend()
plt.grid()
plt.show()


plt.figure()
plt.plot(Ff_padding, 10*np.log10(X2_padding_fft_modulo), label="X2 con zero padding", color = "red")
plt.xlim([0,fs/2])
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Potencia [dB]")
plt.title("Espectro de la señal con Zero Padding")
plt.legend()
plt.grid()
plt.show()


plt.figure()
plt.plot(Ff_padding, 10*np.log10(X3_padding_fft_modulo), label="X3 con zero padding", color = "violet")
plt.xlim([0,fs/2])
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Potencia [dB]")
plt.title("Espectro de la señal con Zero Padding")
plt.legend()
plt.grid()
plt.show()





