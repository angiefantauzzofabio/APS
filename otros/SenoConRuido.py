#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 16:03:54 2025

@author: usuario
"""

import numpy as np
import matplotlib.pyplot as plt

def sen(ff, nn, amp=1, ph=0, dc=0, fs=2, ruido=False, nivel_ruido=0.1):
    """
    Genera una señal senoidal, con opción de agregar ruido.
    
    Parámetros:
    ff : frecuencia de la senoidal [Hz]
    nn : número de muestras
    amp : amplitud
    ph : fase en radianes
    dc : componente DC
    fs : frecuencia de muestreo [Hz]
    ruido : bool, si True se agrega ruido gaussiano
    nivel_ruido : amplitud relativa del ruido
    
    Retorna:
    t : vector de tiempo
    x : señal senoidal (con o sin ruido)
    """
    N = np.arange(nn)
    t = N / fs
    x = dc + amp * np.sin(2 * np.pi * ff * t + ph)

    if ruido:
        x += nivel_ruido * np.random.normal(0, 1, size=nn)

    return t, x



# Señal limpia
t1, x1 = sen(5, 200, amp=1, fs=100, ruido=False)

# Señal con ruido
t2, x2 = sen(5, 200, amp=1, fs=100, ruido=True, nivel_ruido=0.3)

# Graficamos
plt.figure(figsize=(10,5))
plt.plot(t1, x1, label="Señal limpia")
plt.plot(t2, x2, label="Señal con ruido", alpha=0.7)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("Senoidal con y sin ruido")
plt.legend()
plt.grid(True)
plt.show()