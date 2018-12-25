import csv
from random import uniform, randint
from math import exp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def leerCSV(archivoDatos, aleatoriamente=False):
    # Leer datos
    with open(archivoDatos, newline='') as csvfile:
        datos = csv.reader(csvfile, delimiter=',')
        datosArreglo = []
        for fila in datos:
            datosArreglo.append(fila)

    # Datos Ordenados
    if not(aleatoriamente):
        entradas = datosArreglo
    else:
    # Datos Aleatoriamente
        entradas = []
        while(len(datosArreglo)>0):
            random = randint(0,len(datosArreglo)-1)
            entradas.append(datosArreglo[random])
            datosArreglo.pop(random)
    # /Datos Aleatoriamente


    respuestasDeseadas = []
    x = []
    for elemento in entradas:
        respuestasDeseadas.append(float(elemento[-1]))
        elemento.pop(-1)
        x.append(float(elemento[0]))
        # Agregamos el sesgo
        elemento.insert(0,1)

    return x, entradas, respuestasDeseadas

class Perceptron:
    def __init__(self, tasaAprendizaje, cantidadParametros, funcionActivacion):
        self.tasaAprendizaje=tasaAprendizaje
        self.cantidadParametros=cantidadParametros
        self.pesos=[]
        self.funcionActivacion=funcionActivacion
        for i in range(0,cantidadParametros+1):
            self.pesos.append(0)
            #self.pesos.append(uniform(0,1))

    def funcionLogistica(self, elemento, alpha=1):
        return 1 / (1 + exp(-alpha*elemento))

    def funcionActivacionLogistica(self, elemento):
        salida = self.pesos[0]
        for i in range(1,self.cantidadParametros+1):
            salida = salida + (float(elemento[i]) * float(self.pesos[i]))
        salida = self.funcionLogistica(salida)
        return salida            

    def funcionActivacionLineal(self, elemento):
        salida = self.pesos[0]
        for i in range(1,self.cantidadParametros+1):
            salida = salida + (float(elemento[i]) * float(self.pesos[i]))
        return salida

    def activarNeurona(self, elemento):
        if self.funcionActivacion == "Logistica":
            return self.funcionActivacionLogistica(elemento)
        elif self.funcionActivacion == "Lineal":
            return self.funcionActivacionLineal(elemento)

    def derivadaFuncionLogistica(self, elemento):
        return self.funcionLogistica(elemento) * (1 - self.funcionLogistica(elemento))


# Leemos los datos y separamos las entradas
# Entrenamiento
x, entradas, respuestasDeseadas = leerCSV("reglin_train.csv")
# Prueba
xPrueba, entradasPrueba, respuestasDeseadasPrueba = leerCSV("reglin_test.csv")
plt.plot(xPrueba, respuestasDeseadasPrueba, label="Funcion con la muestra dada para probar")
plt.plot(x, respuestasDeseadas, label="Funcion con la muestra dada para entrenar")
# for i in range(0, len(x)):
#     print(x[i])
#     print(entradas[i])
#     print(respuestasDeseadas[i])
#     print("")
# # Graficar funcion
# plt.plot(x, respuestasDeseadas, 'ro')
# plt.xlabel('Entrada')
# plt.ylabel('Respuesta Deseada')
# plt.title("Funcion a estimar")
# plt.show()


# Arreglo de neuronas para la capa oculta
# Cantidad de Neuronas en la capa oculta
erroresEntrenamiento = []
erroresPrueba = []
cantNeuronas = [1,2,3,4,6,8,12,20,40]
for n in cantNeuronas:
    capaOculta = []
    for i in range(0,n):
        capaOculta.append(Perceptron(0.1, 1,"Logistica"))
    # Creamos neurona de salida con funcion de activacion lineal
    neuronaSalida = Perceptron(0.1, n, "Lineal")
    epoca = 0
    # Para graficar
    epocas = []
    errores = []

    # Condicion parada backpropagation
    while(epoca <= 700):
        if epoca % 350 == 0:
            print("Epoca: " + str(epoca))
        epoca += 1
        epocas.append(epoca)
        errorCuadraticoMedio = 0

        # Procesamos las entradas
        for i in range(0,len(entradas)): 
            # Pasamos la entrada a cada neurona de la capa oculta
            salidasCapaOculta = []
            for neurona in capaOculta:
                salidasCapaOculta.append(neurona.activarNeurona(entradas[i]))
            # Agregamos el sesgo de entrada a la capa de salida
            salidasCapaOculta.insert(0,1)
            # Pasamos las salidas de la capa oculta a la neurona de salida
            salidaFinal = neuronaSalida.activarNeurona(salidasCapaOculta)
            error = float(respuestasDeseadas[i])-float(salidaFinal)
            errorCuadraticoMedio = float(errorCuadraticoMedio) + (float(error)**2)
            gradienteSalida = error # por la derivada de la funcion linea: 1 
            # Actualizo los pesos de la neurona salida
            pesosViejos = []
            for k in range(0, len(neuronaSalida.pesos)):
                deltaW = (float(neuronaSalida.tasaAprendizaje) * gradienteSalida * float(salidasCapaOculta[k]))
                pesosViejos.append(neuronaSalida.pesos[k])
                neuronaSalida.pesos[k] = float(neuronaSalida.pesos[k]) + deltaW

                                        
            # Actualizo los pesos de las neuronas de la capa oculta
            for z in range(0, len(capaOculta)):
                # Calculamos gradienteLocal
                sumatoriaGradientesPosteriores = float(gradienteSalida) * float(pesosViejos[z])
                # Calculamos el estimulo recibido por la neurona
                estimulo = capaOculta[z].pesos[0]
                for d in range(1,capaOculta[z].cantidadParametros+1):
                    estimulo = estimulo + (float(entradas[i][d]) * float(capaOculta[z].pesos[d]))
                gradienteLocal = capaOculta[z].derivadaFuncionLogistica(estimulo) # Derivada Funcion Activacion evaluada en el estimulo
                gradienteLocal = gradienteLocal * sumatoriaGradientesPosteriores
                # Calculo deltaW y actualizo cada peso
                for k in range(0, len(capaOculta[z].pesos)):
                    deltaW = (float(capaOculta[z].tasaAprendizaje) * float(gradienteLocal) * float(entradas[i][k]))
                    capaOculta[z].pesos[k] = float(capaOculta[z].pesos[k]) + deltaW


        # Finalizar el calculo del error cuadratico medio
        errorCuadraticoMedio = errorCuadraticoMedio/len(entradas)
        errores.append(errorCuadraticoMedio)
    print("Error cuadratico medio del conjunto de entrenamiento con " + str(n) + " Neuronas: " + str(errorCuadraticoMedio) + "")
    erroresEntrenamiento.append(errorCuadraticoMedio)

    # Grafica 
    # plt.plot(epocas,errores)
    # plt.xlabel('Epocas')
    # plt.ylabel('Error Cuadratico Medio')
    # plt.title("Historial de Errores")
    # plt.show()

    # Verificacion ###########################################################################################
    # Procesamos las entradas
    errorCuadraticoMedio = 0
    salidasPrueba = []
    for i in range(0,len(entradasPrueba)): 
        # Pasamos la entrada a cada neurona de la capa oculta
        salidasCapaOculta = []
        for neurona in capaOculta:
            salidasCapaOculta.append(neurona.activarNeurona(entradasPrueba[i]))
        # Agregamos el sesgo de entrada a la capa de salida
        salidasCapaOculta.insert(0,1)
        # Pasamos las salidas de la capa oculta a la neurona de salida
        salidaFinal = neuronaSalida.activarNeurona(salidasCapaOculta)
        salidasPrueba.append(salidaFinal)
        error = float(respuestasDeseadasPrueba[i])-float(salidaFinal)
        errorCuadraticoMedio = float(errorCuadraticoMedio) + (float(error)**2)

    errorCuadraticoMedio = errorCuadraticoMedio/len(entradasPrueba)
    erroresPrueba.append(errorCuadraticoMedio)
    print("Error cuadratico medio del conjunto de prueba con " + str(n) + " Neuronas: " + str(errorCuadraticoMedio) + "")
    # Graficar funcion
    if n in [2,8,40]:
        plt.plot(xPrueba, salidasPrueba, label="Funcion estimada por la red con " + str(n) + " neuronas.")

plt.xlabel('Entrada')
plt.ylabel('Respuesta')
plt.title("Funcion estimada")
# plt.plot(xPrueba, respuestasDeseadasPrueba, label="Funcion con la muestra dada para probar")
# plt.plot(x, respuestasDeseadas, label="Funcion con la muestra dada para entrenar")
plt.legend()
plt.show()

# Graficar Errores
plt.xlabel('Cantidad de Neuronas')
plt.ylabel('Error Cuadratico Medio')
plt.title("Errores por cantidad de Neuronas")
plt.plot(cantNeuronas, erroresEntrenamiento, "o", label="Errores Entrenamiento")
plt.plot(cantNeuronas, erroresPrueba, "o", label="Errores Prueba")
plt.legend()
plt.show()
