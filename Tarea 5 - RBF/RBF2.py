import csv
from random import uniform, randint, sample
from math import exp, sin
import matplotlib.pyplot as plt
import numpy as np


def funcionBaseRadial(r, sigma = 1):
    return(exp(-((r**2)/(2*(sigma**2))))) 

tamaño = 100
xs = np.random.uniform(0,1,size=tamaño)
xs.sort()
ys = []
datosEsperados = []
for i in range(0,len(xs)):
    ys.append(0.5 + (0.4*sin(2*3.1416*xs[i])))
    datosEsperados.append(0.5 + (0.4*sin(2*3.1416*xs[i])))
plt.plot(xs,ys, label="Funcion Original sin Ruido con " + str(tamaño) + " puntos")

# Añadimos el ruido a los resultados
varianza = 0.1
ruidoBlanco = np.random.normal(size=tamaño, scale=varianza)
for i in range(0,tamaño):
    ys[i] = ys[i] + ruidoBlanco[i]
plt.plot(xs,ys, label="Funcion Original con Ruido de varianza " + str(varianza))
plt.legend()
plt.show()

# Seleccionamos los centros
porcentajes = [5]
for porcentaje in porcentajes:
    plt.plot(xs,ys, label="Funcion Original con Ruido de varianza " + str(varianza))
    centros = sample(range(0, len(xs)), (int((len(xs)*porcentaje)/100)))
    centros.sort()
    x = []
    y = []
    for i in centros:
        x.append(xs[i])
        y.append(ys[i])
    plt.scatter(x, np.zeros(len(x)), label= str(str((int((len(xs)*porcentaje)/100)))) + " Centros")
################# ENTRENAMIENTO ######################
    sigmas = [0.01,0.1,0.2,0.3,0.5,1]
    errores = []
    for sigma in sigmas:
        matrizInterpolacion = []
        for i in range(0,len(xs)):
            vectorSigmas = []
            for j in range(0,len(x)):
                # print("Restando " + str(x[i]) + " con "+ str(x[j]) + " igual a " + str(x[i]-x[j]) + " con exp " + str(funcionBaseRadial(x[i]-x[j])))
                vectorSigmas.append(funcionBaseRadial(abs(xs[i]-x[j]), sigma=sigma))
            matrizInterpolacion.append(np.array(vectorSigmas))

        pseudoInversaMatrizInterpolacion = np.linalg.pinv(matrizInterpolacion)
        D = np.array(ys)
        pesos = np.matmul(pseudoInversaMatrizInterpolacion,D)
        # print(pesos)

        # Probamos la red con los mismos datos
        errorCuadraticoMedio = 0
        salidas = []
        for i in range(0,len(xs)):
            salida = 0
            for j in range(0,len(x)):
                salida += pesos[j] * funcionBaseRadial(abs(xs[i]-x[j]), sigma=sigma)
            salidas.append(salida)
            error = float(datosEsperados[i])-float(salida)
            errorCuadraticoMedio = float(errorCuadraticoMedio) + (float(error)**2)
        plt.plot(xs,salidas, label="Funcion con la Red de " + str((int((len(xs)*porcentaje)/100))) + " neuronas y sigma " + str(sigma))

        # Finalizar el calculo del error cuadratico medio
        errorCuadraticoMedio = errorCuadraticoMedio/len(xs)
        errores.append(errorCuadraticoMedio)
    print("Errores:")
    print(errores)

    plt.legend()
    plt.show()
    # Graficamos Errores
    # plt.xlabel("Sigmas")
    # plt.ylabel("Error")
    # plt.plot(sigmas,errores, label="Errores Cuadraticos medios para las redes con " + str((int((len(xs)*porcentaje)/100))) + " neuronas.")
    # plt.legend()
    # plt.show()