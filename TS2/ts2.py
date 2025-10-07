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
T_total = N / fs

f1 = 2000  # Hz
f_cuadrada = 4000
T_pulso = 0.01
N_pulso = int(T_pulso*fs)

impulso = unit_impulse(N)


x1 = np.sin(2*np.pi*f1*t)  # Senoidal 2 kHz
x2 = 2*np.sin(2*np.pi*f1*t + np.pi/2)  # Amplificada y desfazada
x3 = x1 * np.sin(2*np.pi*(f1/2)*t)  # AM con f1/2
x4 = np.clip(x2, -0.75, 0.75)
x5 = signal.square(2*np.pi*f_cuadrada*t)
x6 = np.zeros_like(t)  # Pulso rectangular 10 ms
x6[:N_pulso] = 1

a1 = np.array([1, -1.5, 0.5]) #coeficientes de y
b1= np.array([0.03, 0.05, 0.03]) #coeficientes de x

"""
Graficar la señal de salida para cada una de las señales de entrada que generó en el TS1. 
Considere que las mismas son causales.

"""



"""
para calcular la respuesta al impulso uso lfilter, le paso los coeficientes y el delta. 
Lo que me garantiza un LTI es que si yo conozco h (la respuesta al impulso) 
puedo concer la y[n] (la salida) convolucionando la x[n] (la sen1al) con la respuesta al impulso. 
Entonces la rta la hago con lfilter pasandole los coeficientes (es como la cajita que me genera esta linealidad e invarianza)
y desp para la salida con la convolucion

"""


salida1 = signal.lfilter(b1, a1, x1)
salida2 = signal.lfilter(b1, a1, x2)
salida3 = signal.lfilter(b1, a1, x3)
salida4 = signal.lfilter(b1, a1, x4)
salida5 = signal.lfilter(b1, a1, x5)
salida6 = signal.lfilter(b1, a1, x6)


plt.figure(figsize=(12, 10))

# Primera fila
plt.subplot(3, 2, 1)
plt.plot(t, salida1, color='orange')
plt.title('Salida de seno 2kHz')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(t, salida2, color='blue')
plt.title('Salida seno 2kHz desfasada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)

# Segunda fila
plt.subplot(3, 2, 3)
plt.plot(t, salida3, color='red')
plt.title('Salida seno 2kHz modulada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(t, salida4, color='green')
plt.title('Salida seno 2kHz clippeada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)

# Tercera fila
plt.subplot(3, 2, 5)
plt.plot(t, salida5, color='violet')
plt.title('Salida señal cuadrada de 4kHz')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(t, salida6, color='pink')
plt.title('Salida Pulso')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)

plt.tight_layout()
plt.show()


"""
Hallar la respuesta al impulso y usando la misma, repetir la generación de la señal de 
salida para alguna de las señales de entrada consideradas en el punto anterior.

"""


 
#----------Señal 2KHz---------
h1 = signal.lfilter(b1, a1, impulso)
y1 = np.convolve(x1, h1)[:N] #hace falta el :N]?

plt.plot(y1, color='blue') 
plt.title('Respuesta al impulso seno 2KHZ')
plt.xlabel('Muestras')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)
plt.show()

#----------Señal 2KHz defasada---------
h2 = signal.lfilter(b1, a1,impulso ) #es una funcion que aplica un lti devolviendo la rta al impulse, a y b son los coeficientes. 
y2 = np.convolve(x2, h2) #Aca obtengo la y (la salida)

plt.plot(y2, color='orange') 
plt.title('Respuesta al impulso Señal 2KHz defasada')
plt.xlabel('Muestras')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)
plt.show()

#----------Señal 2KHz modulada---------
h3 = signal.lfilter(b1, a1, impulso)
y3 = np.convolve(x3, h3)

plt.plot(h3, color='black') 
plt.title('Respuesta al impulso Señal 2KHz modulada')
plt.xlabel('Muestras')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)
plt.show()

#----------Señal 2KHz clippeada---------
h4 = signal.lfilter(b1, a1, impulso)
y4 = np.convolve(x4, h4)

plt.plot(h4, color='red') 
plt.title('Respuesta al impulso Señal 2KHz clippeada')
plt.xlabel('Muestras')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)
plt.show()

#----------Señal cuadrada 4KHz ---------
h5 = signal.lfilter(b1, a1, impulso)
y5 = np.convolve(x5, h5)

plt.plot(h5, color='pink') 
plt.title('Respuesta al impulso Señal cuadrada 4KHz')
plt.xlabel('Muestras')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)
plt.show()

#----------Pulso---------
h6 = signal.lfilter(b1, a1, impulso)
y6 = np.convolve(x6, h6)

plt.plot(h6, color='green') 
plt.title('Respuesta al impulso Pulso')
plt.xlabel('Muestras')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)
plt.show()


"""
En cada caso indique la frecuencia de muestreo, el tiempo de simulación y la potencia o energía de la señal de salida.

"""


print("Señal 1--> fs = %.1fHz, Tiempo de simulacion= %.3f, Potencia= %.3fW"% (fs, T_total, potencia(y1)))
print("Señal 2--> fs = %.1fHz, Tiempo de simulacion= %.3f, Potencia= %.3fW"% (fs, T_total, potencia(y2)))
print("Señal 3--> fs = %.1fHz, Tiempo de simulacion= %.3f, Potencia= %.3fW"% (fs, T_total, potencia(y3)))
print("Señal 4--> fs = %.1fHz, Tiempo de simulacion= %.3f, Potencia= %.3fW"% (fs, T_total, potencia(y4)))
print("Señal 5--> frecuecia = %.1fHz, Energia= %.3fJ"% (f_cuadrada, energia(y5)))
print("Señal 6--> Tiempo de simulacion= %.3f, Energia= %.3fJ"% (T_pulso, energia(y6)))



#-------------PUNTO 2-------------------

"""
Hallar la respuesta al impulso y la salida correspondiente a una señal de entrada senoidal en los
sistemas definidos mediante las siguientes ecuaciones en diferencias:

    - 𝑦[𝑛]=𝑥[𝑛]+3⋅𝑥[𝑛−10]
    - 𝑦[𝑛]=𝑥[𝑛]+3⋅𝑦[𝑛−10]

"""


a2 = np.array([1]) #coeficientes de y
b2 = np.zeros(11) #coeficientes de x
b2[0] = 1
b2[10] = 3

h1_p2 = signal.lfilter(b1, a1, impulso)  #respuesta al impulso
y1_p2= np.convolve(x1, h1_p2)[:len(x1)]

plt.figure()
plt.plot(t, y1_p2)
plt.title("Salida Sistema 1 (seno 2 kHz)")
plt.xlabel("Tiempo [s]"); plt.ylabel("Amplitud"); plt.grid(True)
plt.show()

plt.figure()
plt.plot(h1_p2)
plt.title("Respuesta al impulso H1")
plt.xlabel("n")
plt.ylabel("h1[n]")
plt.grid(True)

b2 = np.array([1.0])
a2 = np.zeros(11)
a2[0]=1
a2[10]=-3
h2_p2 = signal.lfilter(b2, a2, impulso)   # rta al impulso, uso lfilter xq tengo que resolverla de forma recursiva, como la ercuacion depende tmb de la salida es infinita h[x]
y2_p2 = signal.lfilter(b2, a2, x1)  # salida al seno


plt.figure()
plt.plot(t, y2_p2)
plt.title("Salida Sistema 2 (seno 2 kHz, se observa inestabilidad)")
plt.xlabel("Tiempo [s]"); plt.ylabel("Amplitud"); plt.grid(True)
plt.show()

plt.figure()
plt.plot(h2_p2)
plt.title("Respuesta al impulso H2")
plt.xlabel("n")
plt.ylabel("h2[n]")
plt.grid(True)






















