import matplotlib.pyplot as plt
import numpy as np
from math import exp, log, sqrt

# Funcion para la tasa de aprendizaje
def tasaAprendizaje(n, n0=0.1, t=1000): # n: numero de epocas
    return (n0)*(exp(-n/t))

# Funcion de vecindad
def vecindad(i,j,n, sigma0=10):
    def sigma(v):
        return sigma0*(exp(-v/(1000/log(sigma0))))

    distancia = sqrt( ((i[0]-j[0])**2) + ((i[1]-j[1])**2) )

    return exp( (-(distancia**2)) / (2*(sigma(n)**2)) )



# Generamos los datos
datos = []
centros = [[0,0,0,0,0,0,0,0],[4,0,0,0,0,0,0,0],[4,4,0,0,0,0,0,0],[0,4,0,0,0,0,0,0]]

for i in range(0,4): # Genero datos con los 4 centros
    for j in range(0,25):# Genero X cantidad de datos
        dato = []
        for k in range(0,8):
            dato.append(np.random.normal(loc=centros[i][k]))
        datos.append([dato, i])


# Iniciamos la red tomando como pesos iniciales algunos puntos de los datos
SOM = []
cantColumnas = 10
cantFilas = 10
pesosAleatorios = np.random.randint(0,len(datos), size=cantColumnas*cantFilas)
for i in range(0,cantFilas):
    filaNeuronas = []
    for j in range(0,cantColumnas):
        filaNeuronas.append(datos[pesosAleatorios[(i*cantColumnas)+j]]) 
    SOM.append(filaNeuronas)

print("Topologia Inicial:")
for i in range(0,cantFilas):
    filaNeuronas = []
    for j in range(0,cantColumnas):
        filaNeuronas.append(SOM[i][j][1])
    print(filaNeuronas)

# FASE DE ORDENAMIENTO
epocas = 1000
for epoca in range(0,epocas):
    if epoca % 10 == 0:
        print("Epoca: " + str(epoca))
    for dato in datos:
        # Determinamos la neurona ganadora para el dato
        neuronaGanadora = (0,0)
        distanciaGanadora = 1000000000
        # Recorremos todas las neuronas
        for i in range(0,cantFilas):
            for j in range(0,cantColumnas):
                distancia = 0
                for k in range(0,len(dato[0])):
                    distancia += (SOM[i][j][0][k] - dato[0][k])**2
                distancia = sqrt(distancia)
                if distancia < distanciaGanadora:
                    distanciaGanadora = distancia
                    neuronaGanadora = (i,j)

        # Actualizamos los pesos
        for i in range(0,cantFilas):
            for j in range(0,cantColumnas):
                for k in range(0,len(dato[0])):                
                    SOM[i][j][0][k] += tasaAprendizaje(epoca)* vecindad(neuronaGanadora, [i,j], epoca) *(dato[0][k]-SOM[i][j][0][k])
# /FASE DE ORDENAMIENTO

print("Topologia tras Ordenamiento:")
for i in range(0,cantFilas):
    filaNeuronas = []
    for j in range(0,cantColumnas):
        claseGanadora = -1
        distanciaGanadora = 1000000000
        for centro in range(0, len(centros)):
            distancia = 0
            for k in range(0,len(centros[centro])):
                distancia += (SOM[i][j][0][k] - centros[centro][k])**2
            distancia = sqrt(distancia)
            if distancia < distanciaGanadora:
                distanciaGanadora = distancia
                claseGanadora = centro
        filaNeuronas.append(claseGanadora)
        SOM[i][j][1] = claseGanadora
    print(filaNeuronas)

# FASE DE CONVERGENCIA
epocas = cantColumnas*cantColumnas*500
for epoca in range(0,epocas):
    if epoca % 10 == 0:
        print("Epoca: " + str(epoca))
    for dato in datos:
        # Determinamos la neurona ganadora para el dato
        neuronaGanadora = (0,0)
        distanciaGanadora = 1000000000
        # Recorremos todas las neuronas
        for i in range(0,cantFilas):
            for j in range(0,cantColumnas):
                distancia = 0
                for k in range(0,len(dato[0])):
                    distancia += (SOM[i][j][0][k] - dato[0][k])**2
                distancia = sqrt(distancia)
                if distancia < distanciaGanadora:
                    distanciaGanadora = distancia
                    neuronaGanadora = (i,j)

        # Actualizamos los pesos
        for i in range(0,cantFilas):
            for j in range(0,cantColumnas):
                for k in range(0,len(dato[0])):                
                    SOM[i][j][0][k] += tasaAprendizaje(epoca,n0=0.01)* vecindad(neuronaGanadora, [i,j], epoca, sigma0=2) *(dato[0][k]-SOM[i][j][0][k])
# /FASE DE CONVERGENCIA 

print("Topologia Final:")
for i in range(0,cantFilas):
    filaNeuronas = []
    for j in range(0,cantColumnas):
        claseGanadora = -1
        distanciaGanadora = 1000000000
        for centro in range(0, len(centros)):
            distancia = 0
            for k in range(0,len(centros[centro])):
                distancia += (SOM[i][j][0][k] - centros[centro][k])**2
            distancia = sqrt(distancia)
            if distancia < distanciaGanadora:
                distanciaGanadora = distancia
                claseGanadora = centro
        filaNeuronas.append(claseGanadora)
        SOM[i][j][1] = claseGanadora
    print(filaNeuronas)

# /Verificacion
errores = 0
for dato in datos:
    # Verificamos la clase real del dato
    claseGanadora = -1
    distanciaGanadora = 1000000000
    for centro in range(0, len(centros)):
        distancia = 0
        for k in range(0,len(centros[centro])):
            distancia += (dato[0][k] - centros[centro][k])**2
        distancia = sqrt(distancia)
        if distancia < distanciaGanadora:
            distanciaGanadora = distancia
            claseGanadora = centro
    dato[1] = claseGanadora

    # Determinamos la neurona ganadora para el dato
    neuronaGanadora = (0,0)
    distanciaGanadora = 1000000000
    # Recorremos todas las neuronas
    for i in range(0,cantFilas):
        for j in range(0,cantColumnas):
            distancia = 0
            for k in range(0,len(dato[0])):
                distancia += (SOM[i][j][0][k] - dato[0][k])**2
            distancia = sqrt(distancia)
            if distancia < distanciaGanadora:
                distanciaGanadora = distancia
                neuronaGanadora = (i,j)

    if dato[1] != SOM[ neuronaGanadora[0] ][ neuronaGanadora[1]][1]:
        errores += 1

print("La red cometio " + str(errores) + " errores.")

# Topologia Inicial:                                                                                                      
# [1, 1, 2, 1, 0, 1, 2, 1, 2, 1]                                                                                          
# [0, 0, 1, 0, 1, 3, 2, 2, 0, 1]                                                                                          
# [0, 2, 0, 2, 0, 1, 0, 3, 2, 3]                                                                                          
# [3, 0, 3, 1, 2, 1, 2, 0, 0, 1]                                                                                          
# [2, 3, 2, 0, 1, 1, 1, 0, 1, 2]                                                                                          
# [2, 0, 1, 0, 3, 2, 1, 0, 2, 3]                                                                                          
# [1, 1, 0, 1, 1, 0, 3, 1, 1, 2]                                                                                          
# [0, 2, 2, 1, 0, 1, 0, 0, 3, 1]                                                                                          
# [3, 1, 2, 2, 3, 1, 3, 2, 2, 1]                                                                                          
# [2, 0, 2, 2, 1, 3, 2, 0, 2, 3]  

# Topologia tras Ordenamiento:                                                                                            
# [3, 3, 3, 3, 3, 3, 2, 0, 0, 0]                                                                                          
# [0, 3, 3, 3, 3, 0, 1, 0, 0, 0]                                                                                          
# [3, 3, 3, 3, 0, 0, 1, 0, 0, 0]                                                                                          
# [2, 2, 3, 3, 2, 1, 0, 3, 0, 0]                                                                                          
# [2, 2, 3, 3, 1, 1, 1, 0, 0, 0]                                                                                          
# [2, 2, 2, 2, 1, 1, 1, 1, 1, 1]                                                                                          
# [2, 2, 1, 2, 2, 2, 1, 3, 0, 1]                                                                                          
# [2, 2, 2, 1, 1, 0, 1, 1, 1, 1]                                                                                          
# [2, 2, 2, 2, 1, 1, 2, 1, 1, 1]                                                                                          
# [3, 1, 2, 2, 3, 1, 1, 1, 1, 1]  

# Topologia Final:                                                                                                        
# [3, 3, 3, 3, 3, 3, 2, 0, 0, 0]                                                                                          
# [0, 3, 3, 3, 3, 0, 1, 1, 0, 0]                                                                                          
# [0, 3, 3, 3, 0, 0, 1, 0, 0, 0]                                                                                          
# [2, 2, 3, 3, 2, 1, 1, 0, 0, 0]                                                                                          
# [2, 3, 3, 3, 1, 1, 1, 0, 0, 1]                                                                                          
# [2, 2, 2, 2, 1, 1, 1, 1, 1, 1]                                                                                          
# [2, 2, 1, 2, 2, 2, 1, 3, 0, 1]                                                                                          
# [2, 2, 2, 1, 1, 0, 1, 1, 1, 1]                                                                                          
# [2, 2, 2, 2, 1, 1, 2, 1, 1, 1]                                                                                          
# [3, 1, 2, 2, 3, 1, 1, 1, 1, 1]                                                                                          
# La red cometio 2 errores.  