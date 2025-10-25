#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 19:45:38 2025

@author: usuario
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


#-------Plantilla------------------
wp = [1,35] #frecuencia de corte/paso (rad/s)
ws = [0.01, 40] #frecuencia de stop/deteida (rad/s)

alpha_p = 1 #atenuacion maxima a la wp, perdidas en banda de paso
alpha_s = 40 #atenuacion minima a la ws, minima atenuacion requerida en banda de paso

#----Aproximaciones de modulo--
f_aprox = 'butter'
#Las otras aproximan mejor en modulo pero no en fase, la de cauer y eso. mas retardo


#Frecuencia de muestreo
fs = 1000 #[Hz]


#----------Diseño de filtro-----------
b,a = signal.iirdesign(wp=wp, ws=ws, gpass=alpha_p, gstop=alpha_s, analog = False, ftype = f_aprox, output='sos', fs= fs )
#devuelve dos listas que son los coeficientes del polinomio


#------Respuesta en frecuencia------
w,h = signal.freqz_sos(b, a, worN=np.logspace(-1, 2, 1000)) #Calcula la respuesta en frecuencia del filtro 



phase = np.unwrap(np.angle(h)) #unwrap es para que no haya discontinuidades, par aque se amas comodo de visualizar. desenvuelve la preiodicidad de la fase

retardo = np.diff(phase)/np.diff(w)

#---Polos y ceros------
z,p,k = signal.tf2zpk(b, a) #lo pimero es los zeros, depsues los polos y k es 

# --- Gráficas ---
plt.figure(figsize=(12, 10))

# Magnitud
plt.subplot(2, 2, 1)
plt.semilogx(w, 20 * np.log10(abs(h)))
plt.title('Respuesta en Magnitud')
plt.xlabel('Pulsación angular [rad/s]')
plt.ylabel('|H(jω)| [dB]')
plt.grid(True, which='both', ls=':')

# Fase
plt.subplot(2, 2, 2)
plt.semilogx(w, np.degrees(phase))
plt.title('Fase')
plt.xlabel('Pulsación angular [rad/s]')
plt.ylabel('Fase [°]')
plt.grid(True, which='both', ls=':')

# Retardo de grupo
plt.subplot(2, 2, 3)
plt.semilogx(w[:-1], retardo)
plt.title('Retardo de Grupo')
plt.xlabel('Pulsación angular [rad/s]')
plt.ylabel('τg [s]')
plt.grid(True, which='both', ls=':')

# Diagrama de polos y ceros
plt.subplot(2, 2, 4)
plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label='Polos')
if len(z) > 0:
    plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label='Ceros')
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
plt.title('Diagrama de Polos y Ceros (plano s)')
plt.xlabel('σ [rad/s]')
plt.ylabel('jω [rad/s]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()















