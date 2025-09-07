#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 22:29:58 2025

@author: 
    
    
    EN ESTE ARCHIVO COMPRUEBO LA IDENRTIDAD DE PARSEVAL Y ADEMAS HAGO EL ZERO PADDING, AUMENTO LA RESOLUCUON DE LA 
    SEÑAL Y APARECEN LOS SINCS
    
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

Ff = np.arange(N)*delta_f #este es el eje x

plt.figure(1)
plt.plot(Ff,np.log10(X_abs_cuadrado)*10, "x", label = "X abs en dB") #espectro en potencia
plt.xlim([0,fs/2]) #fs/2 es la frecuencia de nyquist
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
    
    
    
    
x_padding = np.zeros(10*N)
x_padding[:len(x)] = x
#ahora transformo, paso del espectro de tiempo a frecuencia
X_padding_fft = fft(x_padding)
X_padding_fft_modulo = np.abs(X_padding_fft)**2 #densidad espectral de potencia
#creo el eje x con las frecuencias pero con 10N para que no me tire el problema de dimensiones
Ff_padding = np.arange(10*N)*delta_f 


f_nyquist_padding = (10*N)/2 #La nyquist cambio porque aumento N

plt.figure(figsize=(8,4))
plt.plot(Ff_padding, 10*np.log10(X_padding_fft_modulo), label="Zero padding")
plt.xlim([0,f_nyquist_padding])
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Potencia [dB]")
plt.title("Espectro de la señal con Zero Padding")
plt.legend()
plt.grid()
plt.show()

#Si yo le hago zoom, veo la sinc, veo los arcos de eso. 
#El padding sirve para aumentar la resolucion, basicamente aumento la cantidad de muestras haciendo lo de 10*N
#y ahi veo los picos, es como que interpolo con la sinc, como le doy mas puntos aparece la sinc que antes no se veia
#con el zero padding aumento la resolucion espectral sin modificar la cantidad de muestras (N) y la frcuencia de muestreo!
