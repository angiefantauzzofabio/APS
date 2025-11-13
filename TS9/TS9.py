#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 21:05:53 2025

@author: usuario
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io as sio
import matplotlib.patches as patches
from scipy.signal import medfilt
from scipy.interpolate import CubicSpline



# Lectura de ECG 
fs_ecg = 1000 # Hz
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()
N = len(ecg_one_lead)

"""
Creamos el filtro, usamos funcion medfilt. Este es el primer metodos, el de la mediana
"""

filtro_mediana = medfilt( volume=ecg_one_lead , kernel_size=201) #el volumen es la señal, kernel size es el tiempo en este caso

"""
El kernel size tiene que ser impar porque sino no tiene mitad 
los FIR tipo II eran impares, el retardo es la cantidad de coeficientes menos 2. 
Esto es para que el retardo sea entero. otra razon para elegirlo.
Igualmente esd para sistemas lineales pero igual tiene mas sentido que sea impar

"""

#La salida del filtro la vuelvo a filtrar pero con 600

filtro_mediana = medfilt( volume=filtro_mediana , kernel_size=601) #esto en definitiva devuelve la linea de base. los movimientos en bajas frecuencias

plt.figure()
plt.plot(ecg_one_lead[80750:89000], label='ECG crudo') #agarro algunas muestras asi no tarda tanto
plt.plot(filtro_mediana[80750:89000], label='ECG filtrado no lineal (Mediana)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('ECG con y sin filtrado')
plt.legend()
plt.grid(True)
plt.show()

ecg_limpio = ecg_one_lead - filtro_mediana 

plt.figure()
plt.plot(ecg_one_lead, label='ECG crudo') #agarro algunas muestras asi no tarda tanto
plt.plot(ecg_limpio, label='ECG limpio')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('ECG con y sin ruido')
plt.legend()
plt.grid(True)
plt.show()


# 3) Graficar resultados
# ===============================

# Zoom en una región representativa
ini, fin = 80750, 89000

plt.figure(figsize=(10,4))
plt.plot(ecg_one_lead[ini:fin], label='ECG original', color='gray')
plt.plot(filtro_mediana [ini:fin], label='Movimiento de línea de base estimado', color='orange')
plt.xlabel('Muestras')
plt.ylabel('Amplitud [mV]')
plt.title('Estimación del movimiento de línea de base con filtro de mediana')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ECG limpio
plt.figure(figsize=(10,4))
plt.plot(ecg_one_lead[ini:fin], label='ECG original', color='gray', alpha=0.5)
plt.plot(ecg_limpio[ini:fin], label='ECG filtrado (sin línea de base)', color='blue')
plt.xlabel('Muestras')
plt.ylabel('Amplitud [mV]')
plt.title('ECG limpio tras eliminación del movimiento de línea de base')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



"""
SEGUNDO METODO: SPLINE CUBICO
"""



qrs_detections = mat_struct['qrs_detections'].flatten() #agarro los qrs

# Anticipamos 80 ms respecto al QRS
n0_muestras = 150 #lo vi a ojo

# Puntos donde estimamos la línea de base
print(qrs_detections)
print(n0_muestras)
   
m_i = qrs_detections - n0_muestras #le restamos ese tiempo para garantizar que no paso la onda P ni Q
s_mi = ecg_one_lead[m_i] #agarro los valores de y, o sea que valor de ecg tengo en esas muestras


# ===============================
# genero funcion interpoladora, spline cubica
# ===============================
n = np.arange(N)
spline = CubicSpline(x= m_i, y = s_mi) #devuelve la funcion interpoladora
b_spline = spline(n) #saco los valores de la funcion en cada muestra . este es el ruido que estimamos


# ===============================
#saco el ruido
# ===============================
ecg_filtrado = ecg_one_lead - b_spline #le saco el ruido 

#gráficos


ini, fin = 80750, 89000

"""
En el primer gráfico, la línea naranja sigue el movimiento lento del ECG → es la estimación de b(n).

En el segundo, el ECG azul ya está “centrado” sin desplazamientos.
"""

plt.figure()
plt.plot(ecg_one_lead[ini:fin], label='ECG original', color='gray')
plt.plot(b_spline[ini:fin], label='Línea de base estimada de Spline (ruido)', color='orange')
plt.xlabel('Muestras')
plt.ylabel('Amplitud [mV]')
plt.title('Estimación del movimiento de línea de base por interpolación spline cúbica')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(ecg_one_lead[ini:fin], label='ECG original', color='green', alpha=0.5)
plt.plot(ecg_filtrado[ini:fin], label='ECG filtrado con Spline', color='blue')
plt.plot(b_spline[ini:fin], label='Línea de base estimada de Spline', color='orange')
plt.xlabel('Muestras')
plt.ylabel('Amplitud [mV]')
plt.title('ECG limpio tras eliminación de línea de base (Spline cúbico)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""
Conclusiones:
    
latidos histriónicos, con unos latidos que se ven como inversos y esos complican la estimacion
son unos picos que no tienen los segmentos isoelectricos, no esta la onda p, entonces esos latidos emporan la estimacion
porque nunca vas a interpolar bien ahi. si vas un poco para atras o adelante en muestras para que caigan
en ese latido emporas las otras muestras. Es una limitacion de este metodo, es muy bueno pero hasta ahi se llefa
Anda bien dependiendo si detectas bien los latidos, sino falla mucho (si eliminas latidos por ejemplo, estima muy mal)
3
Ademas computacionalmente es mejor.
"""


plt.figure()
plt.title('COMPARACION DE LINEAS DE BASE POR METODOS (ruido)')
plt.plot(filtro_mediana[ini:fin] , label='Línea de base estimada con Mediana', color='orange')
plt.plot(b_spline[ini:fin], label='Línea de base estimada de Spline', color='blue')
plt.xlabel('Muestras')
plt.ylabel('b hat')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()






























