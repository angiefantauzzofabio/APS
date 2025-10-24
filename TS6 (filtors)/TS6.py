#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 21:38:40 2025

@author: usuario
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- Definimos los tres filtros usando coeficientes del numerador y denominador ---

# T1(s) = (s^2 + 9)/(s^2 + sqrt(2) s + 1)
num_T1 = [1, 0, 9]             # a*s^2 + d*s + b
den_T1 = [1, np.sqrt(2), 1]    # s^2 + s/CR + 1/CL

# T2(s) = (s^2 + 0.2*s + 1)/(s^2 + sqrt(2) s + 1)
num_T2 = [1, 0.2, 1]
den_T2 = [1, np.sqrt(2), 1]

# T3(s) = (s^2 + 1/9)/(s^2 + 0.2*s + 1)
num_T3 = [1, 0, 1/9]
den_T3 = [1, 0.2, 1]

# Creamos listas para iterar fácilmente
filtros = {
    'T1': (num_T1, den_T1),
    'T2': (num_T2, den_T2),
    'T3': (num_T3, den_T3)
}

# --- Frecuencia para graficar ---
w = np.logspace(-1, 2, 1000)  # de 0.1 a 100 rad/s

plt.figure(figsize=(12, 8))

for i, (name, (num, den)) in enumerate(filtros.items(), 1):
    # Generamos la función de transferencia
    system = signal.TransferFunction(num, den)
    w_out, mag, phase = signal.bode(system, w)
    
    # Módulo en dB
    plt.subplot(2, 1, 1)
    plt.semilogx(w_out, mag, label=name)
    plt.xlabel('Frecuencia [rad/s]')
    plt.ylabel('Magnitud [dB]')
    plt.title('Respuesta en magnitud')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Fase en grados
    plt.subplot(2, 1, 2)
    plt.semilogx(w_out, phase, label=name)
    plt.xlabel('Frecuencia [rad/s]')
    plt.ylabel('Fase [°]')
    plt.title('Respuesta en fase')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

plt.subplot(2, 1, 1)
plt.legend()
plt.subplot(2, 1, 2)
plt.legend()
plt.tight_layout()
plt.show()
