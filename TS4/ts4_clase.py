#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 19:29:05 2025

@author: usuario
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
from scipy.signal import windows
from numpy.fft import fftfreq
import pandas as pd


def eje_temporal(N,fs):
    Ts = 1/fs
    t_final = N*Ts
    tt= np.arange(0,t_final,Ts)
    return tt

def seno(tt,frec,amp,fase = 0, v_medio=0):
    xx = amp*np.sin(2*np.pi*frec*tt + fase) + v_medio
    return xx


SNR = 10 #SNR es db
amp_0= np.sqrt(2) #en Volts
N = 1000
fs = N # en Hertz
deltaf = fs/N # En Hertz, resolucion espectral.
mu = 0
realizaciones = 200  

fr = np.random.uniform(-2,2,size=realizaciones)*deltaf #Estas son nuestras frecuencias aleatornias, es uns distribucion normal
tt = eje_temporal(N = N, fs=fs).flatten()

tt = tt.reshape((-1,1))
fr = fr.reshape((1,-1))

w0 = N/4
w1 = 2*np.pi*((w0 + fr) * deltaf)


#Matriz tiempo
TT = np.tile(tt, (1,realizaciones))

#Matriz frecuencias randoms
FF = np.tile(fr, (N,1))

#Ruido
potencia_ruido = amp_0**2/(2*10**(SNR/10)) #esta seria la varianza tmb
print("Potencia/Varianza de ruido:", potencia_ruido)
desvio_estandar = np.sqrt(potencia_ruido)
R = np.random.normal(loc = 0, scale = desvio_estandar, size = (N,realizaciones))
varianza_ruido = np.var(R)
print("Potencia/Varianza de ruido:", varianza_ruido)
print("La matriz ruido:" , R)

#Matriz de señales con ruido
S = amp_0*np.sin(w1*TT) + R


#Ventanas
rect = np.ones((N,1))
flattop = windows.flattop(N).reshape(N,1)
blackmanharris = windows.blackmanharris(N).reshape(N,1)
blackman = windows.blackman(N).reshape(N,1)

#Venataneo la matriz S (señal de senos con ruido) con las diferentes ventanas
S_rect = S*rect
S_ventaneada_flattop = S*flattop
S_ventaneada_blackmanharris = S*blackmanharris
S_ventaneada_blackman = S*blackman

#Transformo FFT y escalo por 1/N
S_fft = fft(S_rect, axis=0)*(1/N)
S_flattop_fft = fft(S_ventaneada_flattop, axis=0)*(1/N)
S_blackmanharris_fft = fft(S_ventaneada_blackmanharris, axis=0)*(1/N)
S_blackman_fft = fft(S_ventaneada_blackman, axis=0)*(1/N)


#Genero los estimadores de amplitud a mitad de banda digital 
a0 = np.abs(S_fft[N//4, :])
a1 = np.abs(S_flattop_fft[N//4, :])
a2 = np.abs(S_blackmanharris_fft[N//4, :])
a3 = np.abs(S_blackman_fft[N//4, :])

#Genero los estimadores de frecuencia
N_half = N//2 + 1  # de 0 a fs/2


f1 = np.abs(np.argmax(S_fft[:N_half, :], axis=0))
f2 = np.abs(np.argmax(S_flattop_fft[:N_half, :], axis=0))
f3 = np.abs(np.argmax(S_blackmanharris_fft[:N_half, :], axis=0))
f4 = np.abs(np.argmax(S_blackman_fft[:N_half, :], axis=0))

# eje de frecuencias (Hz)
freqs = np.fft.fftfreq(N, d=1/fs)  # vector de frecuencias entre -fs/2 y fs/2
freqs_pos = freqs[:N_half]

# Los paso a  Hz
f1_hz = freqs_pos[f1]
f2_hz = freqs_pos[f2]
f3_hz = freqs_pos[f3]
f4_hz = freqs_pos[f4]


#Los paso a dB
a0_db = 20*np.log10(a0)
a1_db = 20*np.log10(a1)
a2_db = 20*np.log10(a2)
a3_db = 20*np.log10(a3)

amp_real = amp_0
amp_real_db = 20*np.log10(amp_real)

#Grafico los histogramas de aplitud
plt.figure()
plt.hist(a0_db, bins=10, alpha=0.3, label="sin ventana")
plt.hist(a1_db, bins=10, alpha=0.3, label="flattop")
plt.hist(a2_db, bins=10, alpha=0.3, label="blackman-harris")
plt.hist(a3_db, bins=10, alpha=0.3, label="blackman")
plt.axvline(amp_real_db, color='r', linestyle='--', linewidth=2, label="Amplitud real")
plt.legend()
plt.title("Histogramas de estimadores de amplitud")
plt.show()


#Grafico los histogramas de frecuencias
plt.figure()
plt.hist(f1_hz, bins=10, alpha=0.3, label="Rectangular")
plt.hist(f2_hz, bins=10, alpha=0.3, label="Flattop")
plt.hist(f3_hz, bins=10, alpha=0.3, label="Blackman-Harris")
plt.hist(f4_hz, bins=10, alpha=0.3, label="Blackman")
plt.xlabel("Frecuencia estimada (Hz)")
plt.ylabel("Número de realizaciones")
plt.title("Histogramas de los estimadores de frecuencia")
plt.legend()
plt.show()

#El histograma muestra cómo se distribuyen los valores del estimador de amplitud a lo largo de las realizaciones.
#Te da una aproximación empírica de la distribución del estimador.
#Ahí podés calcular la media (valor esperado ≈ sesgo) y la varianza (precisión) del estimador.

# --- Esperanza (media) ---
mu_a0 = np.mean(a0)  # media muestral
mu_a1 = np.mean(a1)
mu_a2 = np.mean(a2)
mu_a3 = np.mean(a3)

# --- Sesgo ---
sesgo_a0 = mu_a0 - amp_real
sesgo_a1 = mu_a1 - amp_real
sesgo_a2 = mu_a2 - amp_real
sesgo_a3 = mu_a3 - amp_real

# --- Varianza ---
var_a0 = np.var(a0, ddof=1)  # varianza de la muestra
var_a1 = np.var(a1, ddof=1)
var_a2 = np.var(a2, ddof=1)
var_a3 = np.var(a3, ddof=1)



f_real = ((w0 + fr.flatten()) * deltaf)  # igual que hiciste antes

# --- Esperanza (media) ---
mu_f1 = np.mean(f1_hz)
mu_f2 = np.mean(f2_hz)
mu_f3 = np.mean(f3_hz)
mu_f4 = np.mean(f4_hz)

# --- Sesgo ---
sesgo_f1 = mu_f1 - f_real.mean()  # le calculo la media al valor real xq es un vector 
sesgo_f2 = mu_f2 - f_real.mean()
sesgo_f3 = mu_f3 - f_real.mean()
sesgo_f4 = mu_f4 - f_real.mean()

var_f1 = np.var(f1_hz)
var_f2 = np.var(f2_hz)
var_f3 = np.var(f3_hz)
var_f4 = np.var(f4_hz)


# Amplitud
tabla_amp = pd.DataFrame({
    "Ventana": ["Rectangular", "Flattop", "Blackman-Harris", "Blackman"],
    "Sesgo (V)": [sesgo_a0, sesgo_a1, sesgo_a2, sesgo_a3],
    "Varianza (V^2)": [var_a0, var_a1, var_a2, var_a3]
})

# Frecuencia
tabla_freq = pd.DataFrame({
    "Ventana": ["Rectangular", "Flattop", "Blackman-Harris", "Blackman"],
    "Sesgo (Hz)": [sesgo_f1, sesgo_f2, sesgo_f3, sesgo_f4],
    "Varianza (Hz^2)": [var_f1, var_f2, var_f3, var_f4]
})

print("Tabla estimadores de amplitud (SNR=10 dB)")
print(tabla_amp)

print("Tabla estimadores de frecuencia (SNR=10 dB)")
print(tabla_freq)


"""
1️⃣ Para estimadores de amplitud

Sesgo:

Queremos que sea cercano a 0, es decir que en promedio no subestime ni sobreestime la amplitud.

Varianza:

Queremos que sea lo más pequeña posible, es decir que las estimaciones individuales no fluctúen demasiado.

💡 Interpretación:

Una ventana con bajo sesgo y baja varianza es la mejor para amplitud.

Ejemplo típico: la ventana flattop suele ser muy buena para estimar amplitudes porque minimiza el error de pico, aunque su resolución en frecuencia es menor.


2️⃣ Para estimadores de frecuencia

Sesgo:

Queremos que la media de los errores sea cercana a 0 → la frecuencia promedio estimada coincide con la real.

Varianza:

Queremos que sea baja → el estimador no “salte” demasiado entre realizaciones.

💡 Interpretación:

Para frecuencia, ventanas como Blackman-Harris o Blackman suelen dar menor dispersión en la estimación de frecuencia, aunque pueden introducir un sesgo pequeño.

Ventanas rectangulares (sin ventana) tienen mayor dispersión pero menos “ensanchamiento” del pico, pueden ser buenas si la frecuencia es exacta y SNR alta.

"""



#Para graficar armo grafico de frecuencias
ff =  np.linspace(0, fs, N)

plt.figure()
plt.plot(ff, 10*np.log10(2*np.abs(S_fft)**2))
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud [dB]")
plt.title("Ventana Rectangular")
plt.xlim([0, fs/2])
plt.grid(True)
plt.show()


plt.figure()
plt.plot(ff, 10*np.log10(2*np.abs(S_flattop_fft)**2))
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud [dB]")
plt.title("Ventana flattop")
plt.xlim([0, fs/2])
plt.grid(True)
plt.show()

plt.figure()
plt.plot(ff, 10*np.log10(2*np.abs(S_blackmanharris_fft)**2))
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud [dB]")
plt.title("Ventana blackmanharris")
plt.xlim([0, fs/2])
plt.grid(True)
plt.show()

plt.figure()
plt.plot(ff, 10*np.log10(2*np.abs(S_blackman_fft)**2))
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud [dB]")
plt.title("Ventana blackman")
plt.xlim([0, fs/2])
plt.grid(True)
plt.show()

#Cuando quiero localizar un valor esperado busco la mediana, uso la funcion median(), centro de masa de la distribucion
#Si le quito la mediana lo estaria insesgando. Su centro de masa se traslado a cero. pero su varianza sigue igual. no cambio 
#Basicamente le eliminamos el error esperado. ya no lo comete mas.
#Basicamen calculamos el sesgo.









