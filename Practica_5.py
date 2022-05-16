import cv2
import numpy as np

#Imagen a editar
imagen= cv2.imread(r'C:\Users\elycu\OneDrive\Escritorio\VisualStudio_Ejemplo\Practicas_Vision_7mo\Perros.jpg',cv2.IMREAD_COLOR)
grises=cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
cv2.imshow('original', imagen)

#----------------------UMBRALIZACIONES BÁSICA--------------------------------

#Umbralizacion(1): Binaria.
"""
Si la intensidad del pixel es mayor al umbral establecido, 
le pixel de destino establece al maximo valor definido 
"""
#Se puede modificar el valor del umbral treshold(imagen, valor de umbral, valor max, metodo de umbralizacion)
vt,th = cv2.threshold(grises, 220, 255, cv2.THRESH_BINARY) 
cv2.imshow('Umbralizacion Binaria, T=220', th)

#Umbralizacion(2): Binaria invertida
"""
Si la intensidad supera el umbral definido el valor será cero,
se establece al valor maximo definido, caso contario que el binario
"""
vt,th = cv2.threshold(grises, 200, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('Umbralizacion Binaria Inversa, T=200', th)

#Umbraliacion(3): Truncar
"""
Si La intensidad del pixel es superior al umbral entonces el pixel destino se establece
a el valor del umbral, de lo contario sera igual al original
"""
vt,th = cv2.threshold(grises, 124, 255, cv2.THRESH_TRUNC)
cv2.imshow('Umbralizacion Truncar, T=124', th)

#Umbralizacion(4): Ajustar a Cero
"""
Cualquier pixel que no supere el umbral se establece en cero
"""
vt,th = cv2.threshold(grises, 124, 255, cv2.THRESH_TOZERO)
cv2.imshow('Umbralizacion Ajuste a Cero, T=124', th)

#Umbralizacion(5): Ajustar a Cero Invertido
"""
Cualquier pixel que supere el umbral establecido, el valor de salida se ajustara a cero
"""
vt,th = cv2.threshold(grises, 240, 255, cv2.THRESH_TOZERO_INV)
cv2.imshow('Umbralizacion Ajustar a Cero Invertido, T=240', th)


#----------------------UMBRALIZACIÓN ADAPTATIVA------------------------------------------

#Umbraliazción(6): Mean
"""threshold = cv2.adaptiveThreshold(img,maxValue,adaptiveMethod,thresholdType,blockSize,C) 
blockSize: tamaño de vecindad de pixeles para calcular el valor de umbral(3,5,7...)
C: constante restada de la media o media ponderada"""
th1 = cv2.adaptiveThreshold(grises, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow('Umbralizacion Mean, BS=11, C=2', th1)

#Umbralización(7): Gaussiana
th2 = cv2.adaptiveThreshold(grises, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow('Umbralizacion Gaussiana, BS=11, C=2', th1)

#Umbralización OTSU (8)
"""Calcula atm un umbral para toda la imagen, no varía por regiones"""
vt,th = cv2.threshold(grises, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('Umbralizacion Otsu', th)

cv2.waitKey(0)
cv2.destroyAllWindows()