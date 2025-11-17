#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 21:05:53 2025

@author: usuario
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io as sio
import matplotlib.patches as patches
from scipy.signal import medfilt
from scipy.interpolate import CubicSpline



# Lectura de ECG 
fs_ecg = 1000 # Hz
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()
N = len(ecg_one_lead)

"""
Creamos el filtro, usamos funcion medfilt. Este es el primer metodos, el de la mediana
"""

filtro_mediana = medfilt( volume=ecg_one_lead , kernel_size=201) #el volumen es la señal, kernel size es el tiempo en este caso

"""
El kernel size tiene que ser impar porque sino no tiene mitad 
los FIR tipo II eran impares, el retardo es la cantidad de coeficientes menos 2. 
Esto es para que el retardo sea entero. otra razon para elegirlo.
Igualmente esd para sistemas lineales pero igual tiene mas sentido que sea impar

"""

#La salida del filtro la vuelvo a filtrar pero con 600

filtro_mediana = medfilt( volume=filtro_mediana , kernel_size=601) #esto en definitiva devuelve la linea de base. los movimientos en bajas frecuencias

plt.figure()
plt.plot(ecg_one_lead[80750:89000], label='ECG crudo') #agarro algunas muestras asi no tarda tanto
plt.plot(filtro_mediana[80750:89000], label='ECG filtrado no lineal (Mediana)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('ECG con y sin filtrado')
plt.legend()
plt.grid(True)
plt.show()

ecg_limpio = ecg_one_lead - filtro_mediana 

plt.figure()
plt.plot(ecg_one_lead, label='ECG crudo') #agarro algunas muestras asi no tarda tanto
plt.plot(ecg_limpio, label='ECG limpio')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('ECG con y sin ruido')
plt.legend()
plt.grid(True)
plt.show()


# 3) Graficar resultados
# ===============================

# Zoom en una región representativa
ini, fin = 80750, 89000

plt.figure(figsize=(10,4))
plt.plot(ecg_one_lead[ini:fin], label='ECG original', color='gray')
plt.plot(filtro_mediana [ini:fin], label='Movimiento de línea de base estimado', color='orange')
plt.xlabel('Muestras')
plt.ylabel('Amplitud [mV]')
plt.title('Estimación del movimiento de línea de base con filtro de mediana')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ECG limpio
plt.figure(figsize=(10,4))
plt.plot(ecg_one_lead[ini:fin], label='ECG original', color='gray', alpha=0.5)
plt.plot(ecg_limpio[ini:fin], label='ECG filtrado (sin línea de base)', color='blue')
plt.xlabel('Muestras')
plt.ylabel('Amplitud [mV]')
plt.title('ECG limpio tras eliminación del movimiento de línea de base')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



"""
SEGUNDO METODO: SPLINE CUBICO
"""



qrs_detections = mat_struct['qrs_detections'].flatten() #agarro los qrs

# Anticipamos 80 ms respecto al QRS
n0_muestras = 150 #lo vi a ojo

# Puntos donde estimamos la línea de base
print(qrs_detections)
print(n0_muestras)
   
m_i = qrs_detections - n0_muestras #le restamos ese tiempo para garantizar que no paso la onda P ni Q
s_mi = ecg_one_lead[m_i] #agarro los valores de y, o sea que valor de ecg tengo en esas muestras


# ===============================
# genero funcion interpoladora, spline cubica
# ===============================
n = np.arange(N)
spline = CubicSpline(x= m_i, y = s_mi) #devuelve la funcion interpoladora
b_spline = spline(n) #saco los valores de la funcion en cada muestra . este es el ruido que estimamos


# ===============================
#saco el ruido
# ===============================
ecg_filtrado = ecg_one_lead - b_spline #le saco el ruido 

#gráficos


ini, fin = 80750, 89000

"""
En el primer gráfico, la línea naranja sigue el movimiento lento del ECG → es la estimación de b(n).

En el segundo, el ECG azul ya está “centrado” sin desplazamientos.
"""

plt.figure()
plt.plot(ecg_one_lead[ini:fin], label='ECG original', color='gray')
plt.plot(b_spline[ini:fin], label='Línea de base estimada de Spline (ruido)', color='orange')
plt.xlabel('Muestras')
plt.ylabel('Amplitud [mV]')
plt.title('Estimación del movimiento de línea de base por interpolación spline cúbica')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(ecg_one_lead[ini:fin], label='ECG original', color='green', alpha=0.5)
plt.plot(ecg_filtrado[ini:fin], label='ECG filtrado con Spline', color='blue')
plt.plot(b_spline[ini:fin], label='Línea de base estimada de Spline', color='orange')
plt.xlabel('Muestras')
plt.ylabel('Amplitud [mV]')
plt.title('ECG limpio tras eliminación de línea de base (Spline cúbico)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""
Conclusiones:
    
latidos histriónicos, con unos latidos que se ven como inversos y esos complican la estimacion
son unos picos que no tienen los segmentos isoelectricos, no esta la onda p, entonces esos latidos emporan la estimacion
porque nunca vas a interpolar bien ahi. si vas un poco para atras o adelante en muestras para que caigan
en ese latido emporas las otras muestras. Es una limitacion de este metodo, es muy bueno pero hasta ahi se llefa
Anda bien dependiendo si detectas bien los latidos, sino falla mucho (si eliminas latidos por ejemplo, estima muy mal)
3
Ademas computacionalmente es mejor.
"""


plt.figure()
plt.title('COMPARACION DE LINEAS DE BASE POR METODOS (ruido)')
plt.plot(filtro_mediana[ini:fin] , label='Línea de base estimada con Mediana', color='orange')
plt.plot(b_spline[ini:fin], label='Línea de base estimada de Spline', color='blue')
plt.xlabel('Muestras')
plt.ylabel('b hat')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
"""
Ahora el filtro adaptado. Nos dan el patron 
"""


patron = mat_struct['qrs_pattern1'].flatten() #el problema de este patron es que el valor medio no va a estar siempre igual
#este matron es un pico random del ECG. Cuando grasfique el patron me conviene hacerlo restandole la media
patron_sm = patron - patron.mean() #AREA NETA NULA (USAR ESTE PATRON PARA QUE NO TENGA AREA NETA NEGATIVA)
plt.figure()
plt.plot(patron,  color='orange')
plt.plot(patron_sm, color='blue')
plt.show()

#el patron lo quiero usar como coeficiente de un filtro uso lfilter

ecg_detection = signal.lfilter(b=patron_sm, a=1, x=ecg_one_lead) #si lo grafico tiene demasiada amplitud pero esta centrada en cero
ecg_detection = np.abs(ecg_detection)

#Normalizo las dos para que ambas tengas varianza unitaria (sino son muy diferentes y no las puedo comparar)
plt.figure()
plt.plot(ecg_detection/(np.std(ecg_detection)),  color='orange')
plt.plot(ecg_one_lead/(np.std(ecg_one_lead)), color='blue')
plt.show()

#Al hacerle zoom vemos que el ecg con el patron tiene retardo. El retardo se debe a que el sistema lineal tiene retardo y como no es simetrico va a tener cualq retardo.
#si me adelanto unas muestras del ecg_detection se va a ver medio mejor
#el siguiente paso es detectar los picos, se que son 1903

mis_qrs = signal.find_peaks(x = ecg_detection, height=1 , distance=300)[0] #la distacia es un dato fisiologico, dist entre picos
print(mis_qrs)
qrs_det = mat_struct['qrs_detections'].flatten()

#Hago la matriz de confusion para detectar los falsos positivos o negativos
from scipy.spatial import distance

def matriz_confusion_qrs(mis_qrs, qrs_det, tolerancia_ms=150, fs=1000):
    """
    Calcula matriz de confusión para detecciones QRS usando solo NumPy y SciPy
    
    Parámetros:
    - mis_qrs: array con tiempos de tus detecciones (muestras)
    - qrs_det: array con tiempos de referencia (muestras)  
    - tolerancia_ms: tolerancia en milisegundos (default 150ms)
    - fs: frecuencia de muestreo (default 360 Hz)
    """
    
    # Convertir a arrays numpy
    mis_qrs = np.array(mis_qrs)
    qrs_det = np.array(qrs_det)
    
    # Convertir tolerancia a muestras
    tolerancia_muestras = tolerancia_ms * fs / 1000
    
    # Inicializar contadores
    TP = 0  # True Positives
    FP = 0  # False Positives
    FN = 0  # False Negatives
    
    # Arrays para marcar detecciones ya emparejadas
    mis_qrs_emparejados = np.zeros(len(mis_qrs), dtype=bool)
    qrs_det_emparejados = np.zeros(len(qrs_det), dtype=bool)
    
    # Encontrar True Positives (detecciones que coinciden dentro de la tolerancia)
    for i, det in enumerate(mis_qrs):
        diferencias = np.abs(qrs_det - det)
        min_diff_idx = np.argmin(diferencias)
        min_diff = diferencias[min_diff_idx]
        
        if min_diff <= tolerancia_muestras and not qrs_det_emparejados[min_diff_idx]:
            TP += 1
            mis_qrs_emparejados[i] = True
            qrs_det_emparejados[min_diff_idx] = True
    
    # False Positives (tus detecciones no emparejadas)
    fp_index = np.where(~mis_qrs_emparejados)[0] #el cero para que lo devuelva fuera de tupla
    tp_index = np.where(mis_qrs_emparejados)[0]
    FP = np.sum(~mis_qrs_emparejados)
    
    # False Negatives (detecciones de referencia no emparejadas)
    fn_index = np.where(~qrs_det_emparejados)[0]
    FN = np.sum(~qrs_det_emparejados)
    
    # Construir matriz de confusión
    matriz = np.array([
        [TP, FP],
        [FN, 0]  # TN generalmente no aplica en detección de eventos
    ])
    
    return matriz, TP, FP, FN, fp_index, fn_index, tp_index

# Ejemplo de uso

matriz, tp, fp, fn,fp_index, fn_index, tp_index = matriz_confusion_qrs(mis_qrs, qrs_det)

print("Matriz de Confusión:")
print(f"           Predicho")
print(f"           Sí    No")
print(f"Real Sí:  [{tp:2d}   {fn:2d}]")
print(f"Real No:  [{fp:2d}    - ]")
print(f"\nTP: {tp}, FP: {fp}, FN: {fn}")

# Calcular métricas de performance
if tp + fp > 0:
    precision = tp / (tp + fp)
else:
    precision = 0

if tp + fn > 0:
    recall = tp / (tp + fn)
else:
    recall = 0

if precision + recall > 0:
    f1_score = 2 * (precision * recall) / (precision + recall)
else:
    f1_score = 0

print(f"\nMétricas:")
print(f"Precisión: {precision:.3f}")
print(f"Sensibilidad: {recall:.3f}")
print(f"F1-score: {f1_score:.3f}")

ecg_norm = ecg_one_lead/(np.std(ecg_one_lead))

plt.figure()
plt.plot(ecg_norm)
plt.plot(mis_qrs[tp_index],ecg_norm[mis_qrs[tp_index]], "og")
plt.plot(mis_qrs[fp_index],ecg_norm[mis_qrs[fp_index]], "dr") #valores falsos que estaban en mis qrs
#plt.plot(qrs_det[fn_index],ecg_norm[qrs_det[fp_index]], "ob") #valoresq no detecte. los busco en la otra lista.
plt.show()

#con lo de la matriz de confusion arreglo mis_qrs y hago una matriz de 1905 (que son los picos) y 113 que es el patron
#despiues pruebo promediando, quiero buscar la que mejor prlmedia a las 1905 realizaciones. 


"""
Arreglo mi vector de detecciones eliminando los falsos positivos
"""

mis_qrs_corrected = np.delete(mis_qrs, fp_index)
mis_qrs_corrected = np.concatenate([mis_qrs_corrected, qrs_det[fn_index]])
mis_qrs_corrected = np.sort(mis_qrs_corrected)

qrs_mat = np.array([ecg_one_lead[ii-60:ii+60] for ii in mis_qrs_corrected])

#plt.figure()
#plt.plot(qrs_mat.transpose())
#plt.show()
#al graficarlos todos tiene un linea continua diferente. 
#lo sulucionamos quitandole el nivel medio. lo hacemos nuelo
qrs_mat = qrs_mat -np.mean(qrs_mat, axis=1).reshape((-1,1))
#plotea devuelta sin el valor medio
plt.figure()
plt.plot(qrs_mat.transpose()) #con esto alineamos los latidos. Ahora quiero ver que si lo promediamos vamos a encontrar un latido medio con menos ruido
plt.show()

"""
para afinar las muestras usamos el algoritmo de woody. hace un patros prmedio, es el latido medio
una vez que calcula ese latido medio, calcula la corelacion cruzada. calcula la correlacion entre ese latido medio
y las realizaciones. El maxio de correlacion va a haber cuando haya mayor solapamiento
esto va a pasar cuando este mas alineado. esta correlacion nos va a dar el defasaje adecuado para moder nuestros latidos y alinearlo mas al 
latido medio. 
si repetis esto, termina convergiendo a alinear todo!
Es algo extra esto pero esta interesante para saber!
"""



























