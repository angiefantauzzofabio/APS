#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 15:49:19 2025

@author: usuario
"""

# guardar como fir_plots.py o ejecutar en un notebook
import numpy as np
import matplotlib.pyplot as plt

# --- parámetros ---
Nw = 2000
omega = np.linspace(0, 2*np.pi, Nw, endpoint=False)

# --- frecuencia: forma directa y forma factorizada ---
H = 0.25*(1 + np.exp(-1j*omega) + np.exp(-2j*omega))
# comprobación alternativa: H_alt = 0.25*np.exp(-1j*omega)*(1 + 2*np.cos(omega))

# módulo y fase
H_mag = np.abs(H)
H_phase = np.angle(H)                # fase en [-pi, pi]
H_phase_unwrap = np.unwrap(H_phase)  # fase desenrollada para ver la pendiente

# retardo de grupo numérico: -d/dw (fase desenrollada)
# usamos derivada centrada para interior y hacia adelante/atrás en extremos
dphase = np.gradient(H_phase_unwrap, omega)
group_delay = -dphase

# respuesta al impulso
h = np.array([0.25, 0.25, 0.25])
n = np.arange(0, len(h))

# --- plots ---
plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.plot(omega, H_mag)
plt.title('Módulo |H(ω)|')
plt.xlabel('ω [rad]')
plt.grid(True)

plt.subplot(2,2,2)
plt.plot(omega, H_phase, label='fase [-π,π]')
plt.plot(omega, H_phase_unwrap, '--', label='fase unwrapped')
plt.title('Fase H(ω)')
plt.xlabel('ω [rad]')
plt.legend()
plt.grid(True)

plt.subplot(2,2,3)
plt.plot(omega, group_delay)
plt.ylim(-1, 3)
plt.title('Retardo de grupo (numérico)')
plt.xlabel('ω [rad]')
plt.grid(True)

plt.subplot(2,2,4)
plt.stem(n, h, use_line_collection=True)
plt.title('Respuesta al impulso h[n]')
plt.xlabel('n')
plt.grid(True)

plt.tight_layout()
plt.show()

# --- información extra: localizar ceros teóricos ---
zeros_condition = 1 + 2*np.cos(omega)
zero_indices = np.where(np.isclose(zeros_condition, 0, atol=1e-3))[0]
print("Índices cercanos a ceros (ω):", omega[zero_indices])
