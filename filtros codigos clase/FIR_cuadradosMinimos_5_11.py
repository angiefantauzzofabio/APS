#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 18:46:21 2025

@author: usuario
"""


"""
TP4 - Diseño de Filtro FIR por Cuadrados Mínimos + Aplicación a señal ECG
Basado en pytc2
este lo hizo el chat con otro codigo, usa el metodo de cuadrados minimos
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io as sio
#import pytc2

#from pytc2.sistemas_lineales import plot_plantilla

#plt.close('all')

#############################################
# 1. Plantilla del filtro
#############################################

# Frecuencia de muestreo
fs = 2000

# Bandas de paso y detención (Hz)
wp = [0.8, 35]
ws = [0.1, 35.7]

# Atenuaciones (en dB)
alpha_p = 1 / 2     # pérdida en banda de paso
alpha_s = 40 / 2    # atenuación mínima en banda detenida

# Vector de frecuencias normalizado y vector deseado
frecuencias = np.sort(np.concatenate(((0, fs / 2), wp, ws)))
deseado = [0, 0, 1, 1, 0, 0]

# Cantidad de coeficientes
cant_coef = 1999
retardo = (cant_coef - 1) // 2

#############################################
# 2. Diseño del filtro FIR por cuadrados mínimos
#############################################

h_firls = signal.firls(numtaps=cant_coef, bands=frecuencias, desired=deseado, fs=fs)

#############################################
# 3. Respuesta en frecuencia
#############################################

w, h = signal.freqz(h_firls, worN=np.logspace(-2, 2, 1000), fs=fs)

# Fase y retardo de grupo
fase = np.unwrap(np.angle(h))
w_rad = w / (fs / 2) * np.pi
gd = -np.diff(fase) / np.diff(w_rad)

#############################################
# 4. Gráficas del filtro
#############################################

#plot_plantilla()  # Figura base pytc2

#plt.figure(figsize=(10, 9))

# Magnitud
plt.subplot(3, 1, 1)
plt.plot(w, 20 * np.log10(abs(h)))
plt.title('Respuesta en Magnitud')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(f)| [dB]')
plt.grid(True, ls=':')

# Fase
plt.subplot(3, 1, 2)
plt.plot(w, fase)
plt.title('Fase')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [rad]')
plt.grid(True, ls=':')

# Retardo de grupo
plt.subplot(3, 1, 3)
plt.plot(w[:-1], gd)
plt.title('Retardo de Grupo')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('τg [muestras]')
plt.grid(True, ls=':')
plt.tight_layout()

#############################################
# 5. Lectura y filtrado de ECG
#############################################

fs_ecg = 1000  # Hz

mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()
N = len(ecg_one_lead)

# Filtrado de la señal ECG
ecg_filt = signal.lfilter(b=h_firls, a=1, x=ecg_one_lead)

#############################################
# 6. Comparación señal original vs filtrada
#############################################

plt.figure(figsize=(10, 5))
plt.plot(ecg_one_lead, label='ECG original')
plt.plot(ecg_filt, label='ECG filtrado (FIR cuadrados mínimos)', alpha=0.8)
plt.legend()
plt.title('ECG completo - crudo vs filtrado')
plt.ylabel('Amplitud')
plt.xlabel('Muestras (#)')
plt.grid(True)

#############################################
# 7. Zoom en regiones de interés
#############################################

# Zonas sin ruido
regs_interes = (
    [4000, 5500],
    [10000, 11000],
)

for reg in regs_interes:
    zoom_region = np.arange(max(0, reg[0]), min(N, reg[1]), dtype='uint')
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG crudo', linewidth=2)
    plt.plot(zoom_region, ecg_filt[zoom_region + retardo], label='ECG filtrado', linewidth=2, alpha=0.8)
    plt.title(f'ECG sin ruido ({reg[0]} a {reg[1]} muestras)')
    plt.legend()
    plt.grid(True)

# Zonas con ruido
regs_ruido = (
    np.array([5, 5.2]) * 60 * fs_ecg,
    np.array([12, 12.4]) * 60 * fs_ecg,
    np.array([15, 15.2]) * 60 * fs_ecg,
)

for reg in regs_ruido:
    zoom_region = np.arange(max(0, reg[0]), min(N, reg[1]), dtype='uint')
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG con ruido', linewidth=2)
    plt.plot(zoom_region, ecg_filt[zoom_region + retardo], label='ECG filtrado', linewidth=2, alpha=0.8)
    plt.title(f'ECG con ruido ({int(reg[0])} a {int(reg[1])} muestras)')
    plt.legend()
    plt.grid(True)

plt.show()
