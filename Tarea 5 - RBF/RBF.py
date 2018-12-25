import csv
from random import uniform, randint, sample
from math import exp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def leerCSV(archivoDatos):
    # Leer datos
    with open(archivoDatos, newline='') as csvfile:
        datos = csv.reader(csvfile, delimiter=',')
        entradas = []
        for fila in datos:
            entradas.append(fila)

    y = []
    x = []
    for elemento in entradas:
        y.append(float(elemento[-1]))
        elemento.pop(-1)
        x.append(float(elemento[0]))

    return x, y

def funcionBaseRadial(r, sigma = 1):
    return(exp(-((r**2)/(2*(sigma**2))))) 

def preprocesarMinMax(arreglo):
    minimo = min(arreglo)
    maximo = max(arreglo)
    nuevoArreglo = []
    for i in arreglo:
        nuevoArreglo.append((maximo-i)/(maximo-minimo))
    return nuevoArreglo, minimo, maximo

def preprocesarMedVar(arreglo):
    media = np.mean(arreglo)
    varianza = np.std(arreglo)
    nuevoArreglo = []
    for i in arreglo:
        nuevoArreglo.append((i-media)/varianza)
    return nuevoArreglo, media, varianza

# xs,ys = leerCSV("rabbit.csv")
xs = [15,15,15,18,28,29,37,37,44,50,50,60,61,64,65,65,72,75,75,82,85,91,91,97,98,125,142,142,147,147,150,159,165,183,192,
195,218,218,219,224,225,227,232,232,237,246,258,276,285,300,301,305,312,317,338,347,354,357,375,394,513,535,554,591,648,
660,705,723,756,768,860]
ys = [21.66,22.75,22.3,31.25,44.79,40.55,50.25,46.88,52.03,63.47,61.13,81,73.09,79.09,79.51,65.31,71.9,86.1,94.6,92.5,105,
101.7,102.9,110,104.3,134.9,130.68,140.58,155.3,152.2,144.5,142.15,139.81,153.22,145.72,161.1,174.18,173.03,173.54,178.86,
177.68,173.73,159.98,161.29,187.07,176.13,183.4,186.26,189.66,186.09,186.7,186.8,195.1,216.41,203.23,188.38,189.7,195.31,
202.63,224.82,203.3,209.7,233.9,234.7,244.3,231,242.4,230.77,242.57,232.12,246.7]
# plt.plot(xs,ys, label="Funcion Original")
# plt.show()

# Preprocesamos las Y
# ys, minimo, maximo = preprocesarMedVar(ys)
xs, minimo, maximo = preprocesarMedVar(xs)
# plt.plot(xs,ys, label="Funcion Original tras preprocesar")


# Seleccionamos los centros
porcentajes = [75,50,25,10]
# porcentaje = 75
for porcentaje in porcentajes:
    plt.plot(xs,ys, label="Funcion Original tras preprocesar")
    centros = sample(range(0, len(xs)), (int((len(xs)*porcentaje)/100)))
    centros.sort()
    x = []
    y = []
    for i in centros:
        x.append(xs[i])
        y.append(ys[i])
    plt.scatter(x, np.zeros(len(x)), label= str((int((len(xs)*porcentaje)/100))) + " Centros")

################# ENTRENAMIENTO ######################
    sigmas = [7,20]
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
            error = float(ys[i])-float(salida)
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