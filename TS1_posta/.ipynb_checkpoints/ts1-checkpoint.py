#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Angelina Fantauzzo Fabio 

Resolver
1) Sintetizar y graficar:

Una señal sinusoidal de 2KHz.
Misma señal amplificada y desfazada en π/2.
Misma señal modulada en amplitud por otra señal sinusoidal de la mitad de la frecuencia.
Señal anterior recortada al 75% de su amplitud.
Una señal cuadrada de 4KHz.
Un pulso rectangular de 10ms.
En cada caso indique tiempo entre muestras, número de muestras y potencia o energía según corresponda.
2) Verificar ortogonalidad entre la primera señal y las demás.

3) Graficar la autocorrelación de la primera señal y la correlación entre ésta y las demás.

3) Dada la siguiente propiedad trigonométrica:

2⋅𝑠𝑖𝑛(α)⋅𝑠𝑖𝑛(β)=𝑐𝑜𝑠(α−β)−𝑐𝑜𝑠(α+β)

Demostrar la igualdad
Mostrar que la igualdad se cumple con señales sinosoidales, considerando α=ω⋅𝑡 , el doble de β 
(Use la frecuencia que desee).

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

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


#modular es multiplicar la misma senial por el seno con la frecuencia que te piden. c
#como los tiempos son iguales solo multiplico las y (el espacio) 
def modular(vmax, dc, fm, ph, N, fs):
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
 w0 = 2*np.pi*fm
 x = vmax*np.sin(w0* tiempo + ph)+dc
 return x

def energia(x):
    energia = np.sum(np.abs(x)**2)
    return energia

    

#----------PARTE 1---------------------------


#Una señal sinusoidal de 2KHz.
"""
Lo unico que me pide la consigna es que la frecuencia debe ser de 2000hz. Deberia pensar que valores
deberian tomar los otros parametros. En particular la cantidad de muestras (N) y la frecuencia de sampleo
fs. 
fs:
    Según Nyquist,  fs≥2f → al menos 4000 Hz. (me conviene que sea mucho mas grande, ya que no tengo limitacion)
N:
    el periodod de la señal de 2khz es de 1/2khz o sea 0,5ms. cada ciclo de onda dura eso
    tengo esta ecuacion: T= N⋅Ts
    si tomo fs 6000hz, Ts es 0.16ms
    voy a probar valores de N, empiezo por N=100
"""

tt, yy = generador_de_señales(1, 0, 2000, 0, 30, 6000)
plt.figure(1)
plt.plot(tt, yy, color='red') 
plt.title('Señal Generada con 2KHz')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)
plt.show()
E1 = energia(yy)

print("Señal 2KHz: Ts=%.6f s, N=%d, Energía=%.4f" % (1/6000, 30, E1))



#Misma señal amplificada y desfasada π/2.
tt2, yy2 = generador_de_señales(2, 0, 2000, np.pi/2, 30, 6000)
plt.figure(1)
plt.plot(tt2, yy2, color='green') 
plt.title('Señal Generada con 2KHz amplificada y desfasada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)
plt.show()

E2 = energia(yy2)

print("Señal 2KHz amplificada y desfasada: Ts=%.6f s, N=%d, Energía=%.4f" % (1/6000, 30, E2))

#Misma señal modulada en amplitud por otra señal sinusoidal de la mitad de la frecuencia.
yy_modular = modular(2, 0, 1000, 0, 30, 6000)
plt.figure(1)
plt.plot(tt, yy*yy_modular, color='blue') 
plt.title('Señal Modulada en 2KHz')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)
plt.show()

E3 = energia(yy*yy_modular)

print("Señal modulada: Ts=%.6f s, N=%d, Energía=%.4f" % (1/6000, 30, E3))

#Señal anterior recortada al 75% de su amplitud.
yy_clip = np.clip(yy2, -0.75, 0.75)
plt.figure(1)
plt.plot(tt2, yy_clip, color='orange') 
plt.title('Señal clipeado')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)
plt.show()

E4 = energia(yy_clip)

print("Señal 2KHz recortada al 0.75 en amplitud: Ts=%.6f s, N=%d, Energía=%.4f" % (1/6000, 30, E4))

#Una señal cuadrada de 4KHz.
f = 4000         
t = np.linspace(0, 1, 1000)
w0 = 2 * np.pi * f
y = signal.square(w0*t)
plt.plot(t, y)
plt.title("Señal cuadrada")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()

E5 = energia(y)
print("Señal cuadrada: Energía=%.4f" % ( E5))


#Un pulso rectangular de 10ms.


#----------PARTE 2---------------------------


















