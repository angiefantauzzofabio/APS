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











































