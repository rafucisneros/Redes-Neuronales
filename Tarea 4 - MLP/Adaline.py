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

    return x, entradas, respuestasDeseadas

class adalinePoliminomico:
    def __init__(self, tasaAprendizaje, gradoPolinomio):
        self.tasaAprendizaje=tasaAprendizaje
        self.gradoPolinomio=gradoPolinomio
        self.pesos=[]
        for i in range(0,gradoPolinomio+1):
            self.pesos.append(uniform(0,1))

    def funcionLineal(self, elemento):
        salida = self.pesos[0]
        for i in range(1,self.gradoPolinomio+1):
            salida = salida + ((float(elemento)**i)* float(self.pesos[i]))
        return salida

    def entrenar(self, archivoDatos, aleatoriamente, epocas):
        # Leer datos
        x, entradas, respuestasDeseadas = leerCSV(archivoDatos)
        # print("RESPUESTAS DESEADAS")
        # print(respuestasDeseadas)
        epoca = 0
        while(epoca<epocas):
            salidas = []
            error = 0
            epoca = epoca + 1
            # print("Epoca #" + str(epoca))
            for k in range(0,len(entradas)):
                respuestaDeseadaActual = respuestasDeseadas[k]
                salida = self.funcionLineal(entradas[k][0])
                salidas.append(salida)
                #print("COMPARANDO " + str(respuestaDeseadaActual) + "CON " + str(salida))

                deltaW = float(respuestaDeseadaActual) - float(salida)
                # Para calcular error cuadratico medio
                # print("Error " + str(error))
                # print("Delta W " + str(deltaW))
                error = float(error) + float(deltaW)**2
                # Actualizo los Pesos
                for i in range(0, len(self.pesos)):
                    self.pesos[i] = float(self.pesos[i]) + (float(self.tasaAprendizaje) * deltaW * (float(entradas[k][0])**i))
                # print(self.pesos)

            # Calcular error cuadratico medio
            error = error/2

            # print("PESOS FINAL EPOCA:")
            # print(self.pesos)
            # print(salidas)
            # print(respuestasDeseadas)
        #print("Usando una tasa de aprendizaje de " + str(self.tasaAprendizaje) + " y " + str(epoca) + " épocas, se logró el entrenamiento con un error de " + str(error) + ".")
        return error

    def clasificar(self, archivoDatos):
        with open(archivoDatos, newline='') as csvfile:
            datos = csv.reader(csvfile, delimiter=',')
            datosArreglo = []
            for fila in datos:
                # Agregamos el sesgo
                fila.insert(0,1)
                datosArreglo.append(fila)

        respuestas = []  
        error = 0          
        for i in range(0, len(datosArreglo)):
            salida = self.funcionLineal(float(datosArreglo[i][1]))
            respuestas.append(salida)
            error = error + ((salida - float(datosArreglo[i][2]))**2)
        error = error / len(datosArreglo)
        return respuestas, error
#gradosPolinomios = [10,20,30,40] #,8,12,20,40]
gradosPolinomios = []
errores = []
x, entradas, respuestasDeseadas = leerCSV("reglin_test.csv")
plt.plot(x, respuestasDeseadas, label="Funcion con la muestra dada")

#for n in gradosPolinomios:
for n in range(0,50):
    # Tasa 0.01
    adaline = adalinePoliminomico(0.1, n) 
    aleatario = False
    adaline.entrenar("reglin_train.csv", aleatario, 100)
    #print("Los pesos finales son:")
    #print(adaline.pesos)
    #print("")
    #print("Clasificamos los nuevos datos y estan son las respuestas:")

    # Leemos los datos y separamos las entradas
    x, entradas, respuestasDeseadas = leerCSV("reglin_test.csv")
    salidas, error = adaline.clasificar("reglin_test.csv")
    errores.append(error)
    gradosPolinomios.append(n)
    # Graficar funcion
    plt.plot(x, salidas, label="Funcion estimada por el Adaline con polinomio de grado: " + str(adaline.gradoPolinomio))
plt.ylabel('Respuesta')
plt.xlabel('Entrada')
plt.title("Funcion estimada")
plt.legend()
plt.show()

plt.plot(gradosPolinomios, errores)
plt.xlabel('Grado de polinomio')
plt.ylabel('Error cuadratico medio')
plt.title("Errores segun grado del Polinomio")
plt.show()