#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 18:49:40 2025

@author: Angelina Fantauzzo Fabio

"""

import numpy as np
import matplotlib.pyplot as plt

n= np.arange(0,2*np.pi,0.01) #secuencia de puntos parte de la funcion evaluados en la funcion seno


#paso a radianes
N = len(n)
print("esto es N:",N)
r=n*(2*np.pi) #la formula es r=n.(2pi/N)
print("esto son los radianes:",r)
f = 2000

fs = 2000
w = f*2*np.pi #esta es la frecuencia angular 

#fs = 1/Ts
#muestras por periodo= N/f
#Nos queda definir el fs, por el teorema de muestreo nos dice que tiene que ser ms grande que la frecuencia del ancho de banda 
#La frecuencia del ancho de banda es 
#el delta f nos vincula N y fs. 
#Si elijo fs=N , delta f =1 y N.Ts =1s 
#los ciclos, ajustandolo as, me temrinan diciendo cuantos hertz tengo. Si tengo dos ciclos 2hz, un ciclo 1hz (ciclos son las ondas, los picos)


#nyqwst = fs/2 
#resolucion espectral --> deltaf=fs/N
#tiempo de registro --> 1/deltaf=N.Ts 


fun = np.sin(r)

plt.plot(n, fun, color = 'red', marker='*')
plt.title("seno")
plt.xlabel("X: radianes")
plt.ylabel("Y")
plt.show()

#-------------------------ejercicio ts0------------------


fs=1000
N = fs
deltaf=fs/N
Ts = 1/(N*deltaf)



tiempo = np.arange(0,N*Ts,Ts) #el segundo parametro es 1 segundo y el tercero tmbien es 1 xq fs=N=1000
#me falta el omega


f=1000  #si juego con la frecuencia, hago el bonus. 500 es nyqwst

w0 = 2*np.pi*f

x = np.sin(w0* tiempo)


plt.plot(tiempo, x, color = 'red', marker='*')
plt.title("TS0: Con 1000hz")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

#en el 999 estoy un poco antes del sampling, se replico para el lado negastiva. quedo en 12 hz pero con la fase contraria. Miro una senoidal pero en ocntra fase, se invertio. 
#naturaleza periodica del dominio. 
#entraq en contra fase. 
#se ve como la replica 



def generador_de_señales(vmax, dc, ff, ph, N, fs):
 '''
  PARAMETROS:
  vmax:amplitud max de la senoidal [Volts]
  dc:valor medio [Volts]
  ff:frecuencia [Hz]
  ph:fase en [rad]
  nn:cantidad de muestras
  fs:frecuencia de muestreo [Hz]
 '''
 N = fs
 Ts = 1/fs
 tiempo = np.arange(0,N*Ts,Ts)
 w0 = 2*np.pi*ff
 x = vmax*np.sin(w0* tiempo + ph)+dc
 return tiempo,x


tt, yy = generador_de_señales(1, 0, 1, 0, 1000, 1000)
plt.figure(1)
plt.plot(tt, yy, color='red') 
plt.title('Señal Generada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)
plt.show()
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    

    







