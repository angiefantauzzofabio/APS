#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 16:29:58 2025

@author: usuario
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

# Señal base (una combinación de senoidales)
fs = 1000        # Frecuencia de muestreo original (Hz)
N = 256          # Número de muestras
t = np.arange(N) / fs
x = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*120*t)   # señal original

# --- Interpolación ---
L = 2  # factor de interpolación
x_up = np.zeros(L * len(x))
x_up[::L] = x  # insertar ceros
fs_up = fs * L

# --- Diezmado ---
M = 2  # factor de diezmado
x_down = x[::M]
fs_down = fs / M

# --- Transformadas ---
def spectrum(sig, fs):
    N = len(sig)
    X = np.fft.fftshift(np.fft.fft(sig, 2048))
    f = np.linspace(-fs/2, fs/2, len(X))
    return f, np.abs(X) / np.max(np.abs(X))

f, X = spectrum(x, fs)
f_up, X_up = spectrum(x_up, fs_up)
f_down, X_down = spectrum(x_down, fs_down)

# --- Graficar ---
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(f, X)
plt.title("Espectro original (fs = %.0f Hz, Nyquist = %.0f Hz)" % (fs, fs/2))
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("|X(f)|")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(f_up, X_up)
plt.title("Después de interpolar x2 (fs = %.0f Hz, Nyquist = %.0f Hz)" % (fs_up, fs_up/2))
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("|X(f)|")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(f_down, X_down)
plt.title("Después de diezmar x2 (fs = %.0f Hz, Nyquist = %.0f Hz)" % (fs_down, fs_down/2))
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("|X(f)|")
plt.grid(True)

plt.tight_layout()
plt.show()
