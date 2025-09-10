#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 16:16:46 2025

@author: usuario
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window

# Definimos parámetros
N = 51  # longitud de la ventana
ventanas = {
    "Rectangular": "boxcar",
    "Hann": "hann",
    "Hamming": "hamming",
    "Blackman Harris": "blackmanharris",
    "Flattop": "flattop"
}

# Vector de frecuencia normalizada [-pi, pi]
w = np.linspace(-np.pi, np.pi, 2048)

plt.figure(figsize=(10,5))

for nombre, tipo in ventanas.items():
    win = get_window(tipo, N, fftbins=False)
    # FFT de la ventana con padding largo para mejor resolución
    W = np.fft.fftshift(np.fft.fft(win, 2048))
    W_dB = 20 * np.log10(np.abs(W) / np.max(np.abs(W)))  # normalizamos a 0 dB
    plt.plot(w, W_dB, label=nombre)

# Estética del gráfico
plt.title("Espectro de diferentes ventanas")
plt.xlabel("Frecuencia normalizada [rad/muestra]")
plt.ylabel("Magnitud [dB]")
plt.ylim(-80, 5)
plt.xlim(-np.pi, np.pi)
plt.legend()
plt.grid(True)
plt.show()