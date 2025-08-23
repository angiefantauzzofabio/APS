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

4) Dada la siguiente propiedad trigonométrica:

    2⋅𝑠𝑖𝑛(α)⋅𝑠𝑖𝑛(β)=𝑐𝑜𝑠(α−β)−𝑐𝑜𝑠(α+β)
    
    Demostrar la igualdad
    Mostrar que la igualdad se cumple con señales sinosoidales, considerando α=ω⋅𝑡 , el doble de β 
    (Use la frecuencia que desee).

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import wave
from scipy.io.wavfile import read

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
 Ts = 1/fs
 tiempo = np.arange(0,N*Ts,Ts)
 w0 = 2*np.pi*fm
 x = vmax*np.sin(w0* tiempo + ph)+dc
 return x

def energia(x):
    energia = np.sum(np.abs(x)**2)
    return energia

def ortogonalidad(x1,x2):
    if len(x1) != len(x2):
        print("distinto largo, no podemos calcular")

    res = np.dot(x1,x2)
    print ("el resultado del producto interno dio:", res)
    
    #para ver si esta cerca de cero
    if np.isclose(res, 0):
        print("Las señales son ortogonales")
    else:
        print("Las señales NO son ortogonales")
        
    return res

    
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
fs = 6000                 # frecuencia de muestreo [Hz]
T_total = 0.05            # duración total de la señal [s] (50 ms)
T_pulso = 0.01            # duración del pulso en segundos (10 ms)

# vector de tiempo
t = np.arange(0, T_total, 1/fs)

# Pulso rectangular: usamos signal.square con duty = T_pulso / T_total
pulso = signal.square(2 * np.pi * 1/T_total * t, duty=T_pulso/T_total)

# Graficar
plt.figure(figsize=(8,3))
plt.plot(t, pulso, color='purple')
plt.title("Pulso rectangular de 10 ms")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()


#----------PARTE 2: Verificar ortogonalidad entre la primera señal y las demás.---------------------------
"""
la sumatoria del prodeucto interno tiene que dar cero
"""
res = ortogonalidad(yy, yy2)
print(res)

res2 = ortogonalidad(yy, yy*yy_modular)
print(res2)

res3 = ortogonalidad(yy, yy_clip)
print(res3)


#----------PARTE 3: Graficar la autocorrelación de la primera señal y la correlación entre ésta y las demás---
auto_corr = np.correlate(yy, yy, mode='full')
plt.plot(auto_corr)
plt.legend()
plt.title("Autocorrelacion")
plt.grid(True)
plt.show()

corr_1= np.correlate(yy, yy2, mode='full')
plt.plot(corr_1)
plt.legend()
plt.title("Correlacion entre señal original y amplificada")
plt.grid(True)
plt.show()

corr_2 = np.correlate(yy, yy_modular, mode='full')
plt.plot(corr_2)
plt.legend()
plt.title("Correlacion entre señal original y la modulada")
plt.grid(True)
plt.show()

corr_3 = np.correlate(yy, yy_clip, mode='full')
plt.plot(corr_3)
plt.legend()
plt.title("Correlacion entre señal original y clipeada")
plt.grid(True)
plt.show()


#-----------------------------PARTE 4-------------------------------------------

#uso los parametros de la señal original
f = 2000         
fs = 6000       
N = 30          
Ts = 1/fs
t = np.arange(0, N*Ts, Ts)
w = 2*np.pi*f


alpha = w*tt
beta = alpha / 2  


izq = 2 * np.sin(alpha) * np.sin(beta)

der = np.cos(alpha - beta) - np.cos(alpha + beta)


if der.all() == izq.all():
    print("igualdad demostrada")
else:
    print("no se pudo verificar")


plt.figure(figsize=(8,4))
plt.plot(t, izq, label="2*sin(alpha)*sin(beta)", color='blue')
plt.plot(t, der, '--', label="cos(alpha-beta) - cos(alpha+beta)", color='red')
plt.title("Verificación de la igualdad trigonométrica")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)
plt.show()

"""
# Opcional: verificar correlación (debería ser máxima)
corr_LR = np.correlate(izq, der, mode='full')
print("Máximo de correlación entre L y R:", np.max(corr_LR))

"""


#--------------BONUS----------------------

#referencia: https://www.geeksforgeeks.org/python/plotting-various-sounds-on-graphs-using-python-and-matplotlib/
def visualize(path: str):
  
    # lectura del archivo de audio
    raw = wave.open(path, "rb")
    
    # leer todos los frames
    signal = raw.readframes(-1)
    signal = np.frombuffer(signal, dtype="int16")
    
    # obtener la frecuencia de muestreo
    f_rate = raw.getframerate()

    # vector de tiempo para el eje x
    time = np.linspace(
        0,             # inicio
        len(signal)/f_rate,  # tiempo final
        num=len(signal)
    )

    # crear figura y plot
    plt.figure(figsize=(8,4))
    plt.plot(time, signal)
    plt.title("Sound Wave")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
    return signal

# -------- EJEMPLO DE USO --------
# Solo llamás a la función con la ruta del archivo
archivo = "/Users/usuario/Desktop/APS/TS1_posta/bonus.wav"  # <-- reemplazá por tu path
bonus = visualize(archivo)

energia_bonus = energia(bonus)
print(energia_bonus)




# read audio samples. referencia: https://stackoverflow.com/questions/18625085/how-to-plot-a-wav-file
input_data = read("/Users/usuario/Desktop/APS/TS1_posta/bonus.wav")
audio = input_data[1]
plt.plot(audio[:,0])
# label the axes
plt.ylabel("Amplitude")
plt.xlabel("Time")
# set the title  
plt.title("Sample Wav")
# display the plot
plt.show()
















