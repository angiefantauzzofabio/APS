# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 22:19:08 2025

@author: JGL
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 21:48:39 2025

@author: JGL
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 19:19:28 2025

@author: JGL
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io as sio
import matplotlib.patches as patches
from scipy.signal import firwin2, freqz, firls, remez

fs = 2000
wp = [0.8, 35] #freq de corte/paso (rad/s)
ws = [0.1, 35.7] #freq de stop/detenida (rad/s)

#si alpha_p es =3 -> max atenuacion, butter

alpha_p = 1/2 #atenuacion de corte/paso, alfa_max, perdida en banda de paso 
alpha_s = 40/2 #atenuacion de stop/detenida, alfa_min, minima atenuacion requerida en banda de paso 


# frecuencias= np.sort(np.concatenate(0, wp,ws, fs/2))
frecuencias= np.sort(np.concatenate(((0,fs/2), wp,ws)))

deseado = [0,1,0]
cant_coef = 1999
retardo = (cant_coef-1)//2


h_firls  = remez(numtaps = cant_coef, bands = frecuencias, desired =deseado ,weight=None,  fs = fs)



# --- Respuesta en frecuencia ---
w, h= freqz(h_firls, worN = np.logspace(-2, 2, 1000), fs = fs) #calcula rta en frq del filtro, devuelve w y vector de salida (h es numero complejo)

# --- Cálculo de fase y retardo de grupo ---

fase = np.unwrap(np.angle(h)) #unwrap hace grafico continuo

w_rad = w / (fs / 2) * np.pi
gd = -np.diff(fase) / np.diff(w_rad) #retardo de grupo [rad/rad]

# --- Polos y ceros ---

z, p, k = signal.sos2zpk(signal.tf2sos(h_firls,a= 1)) #ubicacion de polos y ceros, z=ubicacion de ceros(=0), p=ubicacion polos, k

# --- Gráficas ---
#plt.figure(figsize=(12,10))

# Magnitud
plt.subplot(3,1,1)
plt.plot(w, 20*np.log10(abs(h)))
plt.title('Respuesta en Magnitud')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('|H(z)| [dB]')
plt.grid(True, which='both', ls=':')

# Fase
plt.subplot(3,1,2)
plt.plot(w, fase)
plt.title('Fase')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('Fase [°]')
plt.grid(True, which='both', ls=':')

# Retardo de grupo
plt.subplot(3,1,3)
plt.plot(w[:-1], gd)
plt.title('Retardo de Grupo ')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('τg [# muestras]')
plt.grid(True, which='both', ls=':')


# plt.figure(figsize=(10,10))
# plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label=f'{f_aprox} Polos')
# axes_hdl = plt.gca()

# if len(z) > 0:
#     plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label=f'{f_aprox} Ceros')

# # Ejes y círculo unitario
# plt.axhline(0, color='k', lw=0.5)
# plt.axvline(0, color='k', lw=0.5)
# unit_circle = patches.Circle((0, 0), radius=1, fill=False, color='gray', ls='dotted', lw=2)
# axes_hdl.add_patch(unit_circle)

# # Ajustes visuales
# plt.axis([-1.1, 1.1, -1.1, 1.1])
# plt.title('Diagrama de Polos y Ceros (plano z)')
# plt.xlabel(r'$\Re(z)$')
# plt.ylabel(r'$\Im(z)$')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


##################
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz

##################
## ECG con ruido
##################

# para listar las variables que hay en el archivo
#io.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()
N = len(ecg_one_lead)

ecg_filt_win = signal.lfilter(b = h_firls, a = 1, x = ecg_one_lead)

plt.figure()

plt.plot(ecg_one_lead, label = 'ecg raw')
plt.plot(ecg_filt_win, label = 'cauer')

plt.legend()


plt.figure()
t = np.arange(N) / fs_ecg  # vector de tiempo en segundos
# plt.plot(t[4000:5500], ecg_one_lead[4000:5500], label='ECG crudo')
# plt.plot(t[4000:5500], ecg_filt_cauer[4000:5500], label='ECG filtrado (Cauer)')
plt.plot(ecg_one_lead[80750:89000], label='ECG crudo')
plt.plot(ecg_filt_win[80750:89000], label='ECG filtrado (Cauer)')
# plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('ECG con y sin filtrado')
plt.legend()
plt.grid(True)
plt.show()


#################################
# Regiones de interés sin ruido #
#################################

cant_muestras = len(ecg_one_lead)

regs_interes = (
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    # plt.plot(zoom_region, ecg_filt_win[zoom_region], label='Butterworth')
    plt.plot(zoom_region, ecg_filt_win[zoom_region + retardo], label='FIR Window')
   
    plt.title('ECG sin ruido desde ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()
 
#################################
# Regiones de interés con ruido #
#################################
 
regs_interes = (
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    # plt.plot(zoom_region, ecg_filt_win[zoom_region], label='Butterworth')
    plt.plot(zoom_region, ecg_filt_win[zoom_region + retardo], label='FIR Window')
   
    plt.title('ECG con ruido desde ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()
