import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write


"""
La densidad espectral de potencia (PSD) describe cómo se distribuye la potencia de una señal en el dominio 
de la frecuencia. Es muy usada en bioseñales (ECG, PPG) y en audio porque permite ver las frecuencias dominantes 
y el nivel de ruido.
"""

##################
# Lectura de ECG #
##################

efs_ecg = 1000
mat_struct = sio.loadmat('ECG_TP4.mat')
ecg = mat_struct['ecg_lead'].squeeze()   # <-- convierte (N,1) en (N,)
ecg = ecg - np.mean(ecg)
N = len(ecg)
fs_ecg = 1000 # Hz

# Periodograma de ECG
"""
Básicamente toma la transformada de Fourier de la señal y calcula su módulo al cuadrado, normalizado.
Es un estimador insesgado, pero tiene mucha varianza → es ruidoso.
"""
f_per, P_per = sig.periodogram(ecg, fs=fs_ecg)


#welch 
"""
Divide la señal en segmentos, los solapa, a cada uno le aplica una ventana (Hann por defecto), 
calcula el periodograma de cada segmento y después promedia.
Ventajas: baja mucho la varianza (el espectro se ve “suavizado” y estable).
Desventaja: la resolución en frecuencia es menor que en el periodograma crudo.
"""
f_welch, P_welch = sig.welch(ecg, fs=fs_ecg)


#periodograma ventaneado con hann

"""
Es similar al periodograma clásico, pero se aplica una ventana a la señal completa 
para reducir el leakage espectral.
"""
window = sig.windows.hann(N)
f_win, P_win = sig.periodogram(ecg, fs=fs_ecg, window=window, scaling='density')


#blackman tuckey 
"""
Método basado en estimar la autocorrelación de la señal, luego aplicarle una ventana,
y finalmente obtener su transformada de Fourier.
Ventaja: puede controlar el compromiso entre resolución y suavizado 
ajustando la longitud de la ventana de autocorrelación.
"""
M = 200  # longitud máxima de autocorrelación a considerar

"""
En el método Blackman–Tukey, el parámetro clave es  𝑀 que define hasta qué retardo de autocorrelación se usa.
M determina un compromiso entre:
Resolución en frecuencia ⟶ mejora con 𝑀 grande
Varianza y ruido ⟶ empeora con  𝑀 grande
"""
rxx = sig.correlate(ecg, ecg, mode='full') / N
mid = len(rxx)//2
rxx = rxx[mid - M: mid + M + 1]  # truncamos autocorrelación
w_bt = sig.windows.blackman(len(rxx))
rxx_win = rxx * w_bt

# Densidad espectral de potencia (FFT de la autocorrelación)
P_bt = np.abs(np.fft.fftshift(np.fft.fft(rxx_win, 2048)))
f_bt = np.linspace(-fs_ecg/2, fs_ecg/2, len(P_bt))

plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(f_per, 10*np.log10(P_per))
plt.title('1. Periodograma (crudo) de ECG')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(f_welch, 10*np.log10(P_welch))
plt.title('2. Método de Welch de ECG')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(f_win, 10*np.log10(P_win))
plt.title('3. Periodograma ventaneado (Hann) de ECG')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(f_bt, 10*np.log10(P_bt / np.max(P_bt)))
plt.title('4. Método Blackman–Tukey de ECG')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.grid(True)

plt.tight_layout()
plt.show()

"""
   PORQUE SE VE TAN DIFERENTE EL DE BLACKMAN TUCKEY?
🔹 1. Forma del espectro en Blackman–Tukey

Lo que ves —un pico centrado en 0 Hz que cae simétricamente hacia ±fs/2— es totalmente normal para este método, porque:

El método Blackman–Tukey calcula la FFT de la autocorrelación.

La autocorrelación de una señal real es simétrica.

Por la transformada de Fourier de una función par, el espectro resultante también es real y simétrico respecto del eje 0.

Por eso ves ese espectro que no “sube y baja” como los periodogramas, sino que muestra una forma suave, centrada en 0, igual a su reflejo negativo.

🔹 2. Diferencia con el periodograma o Welch

Los otros métodos (periodogram, welch) te dan el espectro de una sola cara (0 → fs/2),
mientras que el Blackman–Tukey te lo está mostrando en las dos caras (−fs/2 → fs/2).

Eso explica por qué tu eje de frecuencia va de −500 a 500 Hz.
"""


####################################
# Lectura de pletismografía (PPG)  #
####################################

fs_ppg = 400 # Hz
# Cargar el archivo CSV como un array de NumPy
ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe
N_ppg = len(ppg)

# Periodograma de ECG
f2_per, P2_per = sig.periodogram(ppg, fs=fs_ppg)

#welch 
f2_welch, P2_welch = sig.welch(ppg, fs=fs_ppg)

#periodograma ventaneada
window = sig.windows.hann(N_ppg)
f2_win, P2_win = sig.periodogram(ppg, fs=fs_ppg,  window=window, scaling='density')

#blackman tuckey
M2 = 40  # longitud máxima de autocorrelación a considerar
rxx2 = sig.correlate(ppg, ppg, mode='full') / N_ppg 
mid2 = len(rxx2)//2
rxx2 = rxx2[mid2 - M2: mid2 + M2 + 1]  # truncamos autocorrelación
w_bt_2 = sig.windows.blackman(len(rxx2))
rxx_win2 = rxx2 * w_bt_2

# Densidad espectral de potencia (FFT de la autocorrelación)
P_bt_2 = np.abs(np.fft.fftshift(np.fft.fft(rxx_win2, 2048)))
f_bt_2 = np.linspace(-fs_ppg/2, fs_ppg/2, len(P_bt_2))


plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(f2_per, 10*np.log10(P2_per))
plt.title('1. Periodograma (crudo) de PPG')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(f2_welch, 10*np.log10(P2_welch))
plt.title('2. Método de Welch de PPG')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(f2_win, 10*np.log10(P2_win))
plt.title('3. Periodograma ventaneado (Hann) de PPG')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(f_bt_2, 10*np.log10(P_bt_2 / np.max(P_bt_2)))
plt.title('4. Método Blackman–Tukey de PPG')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.grid(True)

plt.tight_layout()
plt.show()

####################
# Lectura de audio #
####################

# Cargar el archivo CSV como un array de NumPy
fs_audio, audio = sio.wavfile.read('la cucaracha.wav')
audio = audio.astype(float)
audio = audio - np.mean(audio)
N_audio = len(audio)

#periodograma crudo
f_raw, P_raw = sig.periodogram(audio, fs=fs_audio, scaling='density')

#periodograma ventaneado (Hann)
window = sig.windows.hann(N_audio)
f_win, P_win = sig.periodogram(audio, fs=fs_audio, window=window, scaling='density')

#welch (promediado)
f_welch, P_welch = sig.welch(audio, fs=fs_audio, window='hann', nperseg=2048, noverlap=1024)

#blackman–Tukey
M = 500  # longitud máxima de autocorrelación
rxx = sig.correlate(audio, audio, mode='full') / N_audio
mid = len(rxx) // 2
rxx = rxx[mid - M : mid + M + 1]  # truncamos autocorrelación
w_bt = sig.windows.blackman(len(rxx))
rxx_win = rxx * w_bt

# Densidad espectral de potencia
P_bt = np.abs(np.fft.fftshift(np.fft.fft(rxx_win, 4096)))
f_bt = np.linspace(-fs_audio / 2, fs_audio / 2, len(P_bt))


plt.figure(figsize=(12, 12))

plt.subplot(4, 1, 1)
plt.plot(f_raw, 10 * np.log10(P_raw))
plt.title('1. Periodograma (crudo) de audio')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(f_win, 10 * np.log10(P_win))
plt.title('2. Periodograma ventaneado (Hann) de audio')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(f_welch, 10 * np.log10(P_welch))
plt.title('3. Método de Welch de audio')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(f_bt, 10 * np.log10(P_bt / np.max(P_bt)))
plt.title('4. Método Blackman–Tukey de audio')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.grid(True)

plt.tight_layout()
plt.show()




def estimar_BW(PSD, ff, cota):
    df = ff[1] - ff[0]
    energia_acumulada = np.cumsum(PSD * df)
    energia_total = energia_acumulada[-1]
    energia_corte = energia_total * cota
    idx_corte = np.where(energia_acumulada >= energia_corte)[0][0]
    frec_BW = ff[idx_corte]
    return frec_BW


# ==========================================================
# 🔹 Cálculo del ancho de banda efectivo con tu función
# ==========================================================

# --- ECG ---
BW_ecg_per = estimar_BW(P_per, f_per, 0.95)
BW_ecg_welch = estimar_BW(P_welch, f_welch, 0.95)
BW_ecg_win = estimar_BW(P_win, f_win, 0.95)

# En Blackman–Tukey el espectro está centrado en 0, por lo tanto
# usamos solo la mitad positiva para medir hasta fs/2
mask_bt_ecg = f_bt >= 0
BW_ecg_bt = estimar_BW(P_bt[mask_bt_ecg], f_bt[mask_bt_ecg], 0.95)

print("=== Ancho de banda ECG ===")
print(f"Periodograma:        {BW_ecg_per:.2f} Hz")
print(f"Welch:               {BW_ecg_welch:.2f} Hz")
print(f"Ventaneado (Hann):   {BW_ecg_win:.2f} Hz")
print(f"Blackman–Tukey:      {BW_ecg_bt:.2f} Hz\n")


# --- PPG ---
BW_ppg_per = estimar_BW(P2_per, f2_per, 0.95)
BW_ppg_welch = estimar_BW(P2_welch, f2_welch, 0.95)
BW_ppg_win = estimar_BW(P2_win, f2_win, 0.95)

mask_bt_ppg = f_bt_2 >= 0
BW_ppg_bt = estimar_BW(P_bt_2[mask_bt_ppg], f_bt_2[mask_bt_ppg], 0.95)

print("=== Ancho de banda PPG ===")
print(f"Periodograma:        {BW_ppg_per:.2f} Hz")
print(f"Welch:               {BW_ppg_welch:.2f} Hz")
print(f"Ventaneado (Hann):   {BW_ppg_win:.2f} Hz")
print(f"Blackman–Tukey:      {BW_ppg_bt:.2f} Hz\n")


# --- AUDIO ---
BW_audio_per = estimar_BW(P_raw, f_raw, 0.99)
BW_audio_welch = estimar_BW(P_welch, f_welch, 0.99)
BW_audio_win = estimar_BW(P_win, f_win, 0.99)

mask_bt_audio = f_bt >= 0
BW_audio_bt = estimar_BW(P_bt[mask_bt_audio], f_bt[mask_bt_audio], 0.99)

print("=== Ancho de banda AUDIO ===")
print(f"Periodograma:        {BW_audio_per:.2f} Hz")
print(f"Welch:               {BW_audio_welch:.2f} Hz")
print(f"Ventaneado (Hann):   {BW_audio_win:.2f} Hz")
print(f"Blackman–Tukey:      {BW_audio_bt:.2f} Hz\n")






