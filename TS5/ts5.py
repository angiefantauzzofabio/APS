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

# Periodograma de ECG
"""
Básicamente toma la transformada de Fourier de la señal y calcula su módulo al cuadrado, normalizado.

Es un estimador insesgado, pero tiene mucha varianza → es ruidoso.
"""
f_per, P_per = sig.periodogram(ecg, fs=fs_ecg)
plt.figure()
plt.plot(f_per, 10*np.log10(P_per))
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.title('Periodograma (crudo)')
plt.grid(True)
plt.show()

#welch 
"""
Divide la señal en segmentos, los solapa, a cada uno le aplica una ventana (Hann por defecto), 
calcula el periodograma de cada segmento y después promedia.

Ventajas: baja mucho la varianza (el espectro se ve “suavizado” y estable).

Desventaja: la resolución en frecuencia es menor que en el periodograma crudo.
"""
f_welch, P_welch = sig.welch(ecg, fs=fs_ecg)
plt.figure()
plt.plot(f_welch, 10*np.log10(P_welch))
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.title('Periodograma con metodo Welch')
plt.grid(True)
plt.show()




####################################
# Lectura de pletismografía (PPG)  #
####################################

fs_ppg = 400 # Hz

# Cargar el archivo CSV como un array de NumPy
ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe

# Periodograma de ECG
f2_per, P2_per = sig.periodogram(ppg, fs=fs_ppg)
plt.figure()
plt.plot(f2_per, 10*np.log10(P2_per))
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.title('Periodograma (crudo)')
plt.grid(True)
plt.show()

#welch 
f2_welch, P2_welch = sig.welch(ppg, fs=fs_ppg)
plt.figure()
plt.plot(f2_welch, 10*np.log10(P2_welch))
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.title('Periodograma con metodo Welch')
plt.grid(True)
plt.show()


##################
## PPG sin ruido
##################

ppg = np.load('ppg_sin_ruido.npy')

plt.figure()
plt.plot(ppg)


####################
# Lectura de audio #
####################

# Cargar el archivo CSV como un array de NumPy
fs_audio_1, wav_data_1 = sio.wavfile.read('la cucaracha.wav')
fs_audio_2, wav_data_2 = sio.wavfile.read('prueba psd.wav')
fs_audio_3, wav_data_3 = sio.wavfile.read('silbido.wav')


# Periodograma (crudo)
f1, Pxx1 = sig.periodogram(wav_data_1, fs=fs_audio_1)
plt.figure()
plt.plot(f1, 10*np.log10(Pxx1))
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD ')
plt.title('Periodograma (crudo)')
plt.grid(True)

# Welch (promediado)
f2, Pxx2 = sig.welch(wav_data_1, fs=fs_audio_1)
plt.figure()
plt.plot(f2, 10*np.log10(Pxx2))
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD ')
plt.title('PSD con Welch')
plt.grid(True)

plt.show()















