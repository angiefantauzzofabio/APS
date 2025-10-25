#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 13:03:53 2025

@author: usuario
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ===============================
# 1️⃣ Definir las funciones T1, T2, T3
# ===============================
T1 = signal.TransferFunction([1, 0, 9], [1, np.sqrt(2), 1])
T2 = signal.TransferFunction([1, 0, 1/9], [1, 1/9, 1])
T3 = signal.TransferFunction([1, 1/5, 1], [1, np.sqrt(2), 1])

# ===============================
# 2️⃣ Calcular polos y ceros
# ===============================
def plot_pzmap(tf, title):
    zeros, poles = tf.zeros, tf.poles
    plt.figure()
    plt.scatter(np.real(zeros), np.imag(zeros), marker='o', facecolors='none', edgecolors='b', label='Ceros')
    plt.scatter(np.real(poles), np.imag(poles), marker='x', color='r', label='Polos')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title(f"Diagrama de Polos y Ceros - {title}")
    plt.xlabel("Parte real (σ)")
    plt.ylabel("Parte imaginaria (jω)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# Graficar polos y ceros de cada función
plot_pzmap(T1, "T1(s)")
plot_pzmap(T2, "T2(s)")
plot_pzmap(T3, "T3(s)")

# ===============================
# 3️⃣ Calcular y graficar respuesta en frecuencia (Bode)
# ===============================
w = np.logspace(-1, 2, 1000)  # de 0.1 a 100 rad/s

# T1
w1, mag1, phase1 = signal.bode(T1, w)
# T2
w2, mag2, phase2 = signal.bode(T2, w)
# T3
w3, mag3, phase3 = signal.bode(T3, w)

plt.figure(figsize=(10, 7))

# Módulo
plt.subplot(2, 1, 1)
plt.semilogx(w1, mag1, label="T1(s)")
plt.semilogx(w2, mag2, label="T2(s)")
plt.semilogx(w3, mag3, label="T3(s)")
plt.title("Diagramas de Bode (Módulo y Fase)")
plt.ylabel("Módulo [dB]")
plt.legend()
plt.grid(True, which='both')

# Fase
plt.subplot(2, 1, 2)
plt.semilogx(w1, phase1, label="T1(s)")
plt.semilogx(w2, phase2, label="T2(s)")
plt.semilogx(w3, phase3, label="T3(s)")
plt.ylabel("Fase [°]")
plt.xlabel("Frecuencia [rad/s]")
plt.grid(True, which='both')

plt.tight_layout()
plt.show()
