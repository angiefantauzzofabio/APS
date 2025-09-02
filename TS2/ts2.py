#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 23:35:51 2025

@author: usuario
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import unit_impulse


#Funciones
def ec_sistema_lti(x):
    y = np.zeros_like(x)
    for k in range(len(x)):
        y[k] = 0.03*x[k] + 0.05*x[k-1] + 0.03*x[k-2] + 1.5*y[k-1] - 0.5*y[k-2]
    return y

def energia(x):
    energia = np.sum(np.abs(x)**2)
    return energia

def potencia (x):
    N = len(x) #cantidad de muestras               
    potencia = np.sum(np.abs(x)**2) / N
    return potencia 


fs = 20000  # Frecuencia de muestreo (20 kHz por seguridad, Nyquist >> 4kHz)
Ts = 1/fs
N =300
n = np.arange(N)
t = np.arange(0, N/fs, 1/fs)

f1 = 2000  # Hz
f_cuadrada = 4000
T_pulso = 0.01
N_pulso = int(T_pulso*fs)


x1 = np.sin(2*np.pi*f1*t)  # Senoidal 2 kHz
x2 = 2*np.sin(2*np.pi*f1*t + np.pi/2)  # Amplificada y desfazada
x3 = x1 * np.sin(2*np.pi*(f1/2)*t)  # AM con f1/2
x4 = np.clip(x2, -0.75, 0.75)
x5 = signal.square(2*np.pi*f_cuadrada*t)
x6 = np.zeros_like(t)  # Pulso rectangular 10 ms
x6[:N_pulso] = 1


"""
Graficar la señal de salida para cada una de las señales de entrada que generó en el TS1. 
Considere que las mismas son causales.

"""


y1 = ec_sistema_lti(x1)
y2 = ec_sistema_lti(x2)
y3 = ec_sistema_lti(x3)
y4 = ec_sistema_lti(x4)
y5 = ec_sistema_lti(x5)
y6 = ec_sistema_lti(x6)


plt.plot(t, y1, color='orange') 
plt.title('Señal de 2kHz')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)
plt.show()

plt.plot(t, y2, color='blue') 
plt.title('Señal de 2kHz desfasada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)
plt.show()

plt.plot(t, y3, color='red') 
plt.title('Señal de 2kHz modulada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)
plt.show()

plt.plot(t, y4, color='green') 
plt.title('Señal de 2kHz clippeada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)
plt.show()

plt.plot(t, y5, color='violet') 
plt.title('Señal cuadrada de 4KHz')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)
plt.show()

plt.plot(t, y1, color='pink') 
plt.title('Pulso')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)
plt.show()


"""
Hallar la respuesta al impulso y usando la misma, repetir la generación de la señal de 
salida para alguna de las señales de entrada consideradas en el punto anterior.

"""


impulso = unit_impulse(N)  #preguntar 
#rta1 = y1*impulso
print(len(impulso))
print(len(y1))

rta1 = np.convolve(y1,impulso)
print(len(rta1))

plt.plot(rta1, color='orange') 
plt.title('Respuesta al impulso')
plt.xlabel('Muestras')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)
plt.show()













