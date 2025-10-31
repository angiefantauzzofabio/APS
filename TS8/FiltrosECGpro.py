
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io as sio
import matplotlib.patches as patches

#filtro normalizado -> todas las singularidades en el circulo unitario?
#--- Plantilla de diseño ---

fs = 1000
wp = [0.8, 35] #freq de corte/paso (rad/s)
ws = [0.1, 40] #freq de stop/detenida (rad/s)

#si alpha_p es =3 -> max atenuacion, butter

alpha_p = 1/2 #atenuacion de corte/paso, alfa_max, perdida en banda de paso 
alpha_s = 40/2 #atenuacion de stop/detenida, alfa_min, minima atenuacion requerida en banda de paso 

f_aprox = 'cauer'
f_aprox2 = 'butterworth'
mi_sos_cauer = signal.iirdesign(wp = wp, ws = ws, gpass = alpha_p, gstop = alpha_s, analog = False, ftype = f_aprox, output ='sos', fs=fs) #devuelve dos listas de coeficientes, b para P y a para Q
mi_sos_butterworth = signal.iirdesign(wp = wp, ws = ws, gpass = alpha_p, gstop = alpha_s, analog = False, ftype = f_aprox2, output ='sos', fs=fs)

# %%
mi_sos = mi_sos_cauer

# --- Respuesta en frecuencia ---
w, h= signal.freqz_sos(mi_sos, worN = np.logspace(-2, 1.9, 1000), fs = fs) #calcula rta en frq del filtro, devuelve w y vector de salida (h es numero complejo)

# --- Cálculo de fase y retardo de grupo ---

fase = np.unwrap(np.angle(h)) #unwrap hace grafico continuo

w_rad = w / (fs / 2) * np.pi
gd = -np.diff(fase) / np.diff(w_rad) #retardo de grupo [rad/rad]

# --- Polos y ceros ---

z, p, k = signal.sos2zpk(mi_sos) #ubicacion de polos y ceros, z=ubicacion de ceros(=0), p=ubicacion polos, k

# --- Gráficas ---
#plt.figure(figsize=(12,10))

# Magnitud
plt.subplot(3,1,1)
plt.plot(w, 20*np.log10(abs(h)), label = f_aprox)
plt.title('Respuesta en Magnitud')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('|H(jω)| [dB]')
plt.grid(True, which='both', ls=':')

# Fase
plt.subplot(3,1,2)
plt.plot(w, fase, label = f_aprox)
plt.title('Fase')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('Fase [°]')
plt.grid(True, which='both', ls=':')

# Retardo de grupo
plt.subplot(3,1,3)
plt.plot(w[:-1], gd, label = f_aprox)
plt.title('Retardo de Grupo ')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('τg [# muestras]')
plt.grid(True, which='both', ls=':')

# # Diagrama de polos y ceros
# plt.subplot(2,2,4)
# plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label=f'{f_aprox} Polos')
# if len(z) > 0:
#     plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label=f'{f_aprox} Ceros')
# plt.axhline(0, color='k', lw=0.5)
# plt.axvline(0, color='k', lw=0.5)
# plt.title('Diagrama de Polos y Ceros (plano z)')
# plt.xlabel('σ [rad/s]')
# plt.ylabel('jω [rad/s]')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()

# %% Gráfico adicional del plano Z con círculo unitario

plt.figure(figsize=(10,10))
plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label=f'{f_aprox} Polos')
axes_hdl = plt.gca()

if len(z) > 0:
    plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label=f'{f_aprox} Ceros')

# Ejes y círculo unitario
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
unit_circle = patches.Circle((0, 0), radius=1, fill=False, color='gray', ls='dotted', lw=2)
axes_hdl.add_patch(unit_circle)

# Ajustes visuales
plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.title('Diagrama de Polos y Ceros (plano z)')
plt.xlabel(r'$\Re(z)$')
plt.ylabel(r'$\Im(z)$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#%%
##############################################################
#DISEÑO DE FIR CONC METODO DE VENTANAS USANDO firwin2
##############################################################

wp = [0.8, 35] #frecuencia de corte/paso
ws = [0.1,37.5] # frecuencia de stop/detenida

frecuencias = np.sort( np.concatenate(((0,fs/2), wp, ws)) )
deseado = [0, 0 ,1, 1, 0, 0] #es la respuesta deseada, tiene que ser un pasa banda. para cada frecuencia tengo que tener un valor deseado

cant_coeficientes = 100 #los coeficientes b, es un filtro par
retardo = (cant_coeficientes-1)//2


fir_win_rec = signal.firwin2(numtaps= cant_coeficientes, freq= frecuencias, gain = deseado, window='boxcar' ,fs = fs) #este es el filtrado ventana, la default es la de hamming

w,h = signal.freqz(b = fir_win_hamming,worN = np.logspace(-2, 1.9, 1000), fs = fs) 

# --- Cálculo de fase y retardo de grupo ---

fase = np.unwrap(np.angle(h)) #unwrap hace grafico continuo

w_rad = w / (fs / 2) * np.pi
gd = -np.diff(fase) / np.diff(w_rad) #retardo de grupo [rad/rad]

# --- Polos y ceros ---

z, p, k = signal.sos2zpk(signal.tf2sos(b= fir_win_hamming, a=1)) #primero lo paso a analogico con sos

# --- Gráficas ---
#plt.figure(figsize=(12,10))

plt.figure(figsize=(8, 10))  # aumenta el tamaño total de la figura

# Magnitud
plt.subplot(3, 1, 1)
plt.plot(w, 20*np.log10(abs(h)), label=f_aprox)
plt.title('Respuesta en Magnitud cACAA')
plt.xlabel('Pulsación angular [rad/s]')
plt.ylabel('|H(jω)| [dB]')
plt.grid(True, which='both', ls=':')
plt.legend()

# Fase
plt.subplot(3, 1, 2)
plt.plot(w, fase, label=f_aprox)
plt.title('Fase')
plt.xlabel('Pulsación angular [rad/s]')
plt.ylabel('Fase [°]')
plt.grid(True, which='both', ls=':')
plt.legend()

# Retardo de grupo
plt.subplot(3, 1, 3)
plt.plot(w[:-1], gd, label=f_aprox)
plt.title('Retardo de Grupo')
plt.xlabel('Pulsación angular [rad/s]')
plt.ylabel('τg [# muestras]')
plt.grid(True, which='both', ls=':')
plt.legend()

plt.tight_layout()  # <-- ajusta los espacios automáticamente
plt.show()

# --- Gráfico de polos y ceros ---
plt.figure(figsize=(6,6))
plt.title('Diagrama de Polos y Ceros del FIR')

# Círculo unitario
circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
plt.gca().add_artist(circle)

# Ceros (marcados con "o") y polos (con "x")
plt.scatter(np.real(z), np.imag(z), marker='o', facecolors='none', edgecolors='b', label='Ceros')
plt.scatter(np.real(p), np.imag(p), marker='x', color='r', label='Polos')

# Ejes y detalles
plt.xlabel('Parte Real')
plt.ylabel('Parte Imaginaria')
plt.grid(True, linestyle=':')
plt.axis('equal')
plt.legend()
plt.show()

"""
Notas de la clase: 
   si mejoro el orden, aumento la cantidad de coeficientes, mucho no mejora, mejora pocos dB.
   esto es porque los filtros fir son mucho mas dificiles de diseñar, hay que usar la intuicion para mejorarlos
   aunque lo duplicamos, hay cosas que andan mal.
   que puede pasar?
   puede tener que ver con la plantilla, vamos a poner mas muestras interpolando mas finamente con
   nfreqs. 
   finalmente la solucion era usar otra ventana, la rectangular, al tener un lobulo bajo nos sirve. 
   Recordar que a fase es lineall, tiene mucho costo computacional, pero es el mejor. 
   es dificil aplicarlo y pensar la plantilla pero vale la pena, porque desp
   la implementacion es muy facil.
   fase lineal, retardo de grupo constante (pero se puede arreglar)
"""



#%%

##################
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz

##################
## ECG con ruido
##################

# para listar las variables que hay en el archivo
#io.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()
N = len(ecg_one_lead)

ecg_filt_cauer = signal.sosfiltfilt(mi_sos_cauer, ecg_one_lead)
ecg_filt_butterworth = signal.sosfiltfilt(mi_sos_butterworth, ecg_one_lead)
ecg_filt_fir= signal.lfilter(b = fir_win_rec, a=1, ecg_one_lead) #uso filtro fir, no hay retardo de fase, pero si de grupo pero constante


# plt.figure()

# plt.plot(ecg_one_lead, label = 'ecg raw')
# plt.plot(ecg_filt_cauer, label = 'cauer')

# plt.legend()


plt.figure()
t = np.arange(N) / fs_ecg  # vector de tiempo en segundos
plt.plot(t[5000:8000], ecg_one_lead[5000:8000], label='ECG crudo')
plt.plot(t[5000:8000], ecg_filt_cauer[5000:8000], label='ECG filtrado (Cauer)')
plt.plot(t[5000:8000], ecg_filt_butterworth[5000:8000], label='ECG filtrado (butterworth)')
plt.xlabel('Tiempo [s]')
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
    plt.plot(zoom_region, ecg_filt_butterworth[zoom_region], label='Butterworth')
    #plt.plot(zoom_region, ECG_f_win[zoom_region + demora], label='FIR Window')
   
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
    plt.plot(zoom_region, ecg_filt_butterworth[zoom_region], label='Butterworth')
   
    plt.title('ECG con ruido desde ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()
