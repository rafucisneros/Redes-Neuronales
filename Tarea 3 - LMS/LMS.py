import csv
from random import uniform, randint
from math import exp


class perceptronLMS:
    def __init__(self, tasaAprendizaje, cantidadParametros):
        self.tasaAprendizaje=tasaAprendizaje
        self.cantidadParametros=cantidadParametros
        self.pesos=[]
        for i in range(0,cantidadParametros+1):
            self.pesos.append(uniform(0,1))

    def funcionLogistica(self, elemento, alpha):
        salida = self.pesos[0]
        for i in range(1,self.cantidadParametros+1):
            salida = salida + (float(elemento[i]) * float(self.pesos[i]))
        salida = 1 / (1 + exp(-alpha*salida))
        return salida

    def entrenar(self, archivoDatos, aleatoriamente, epocas):
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
        for elemento in entradas:
            respuestasDeseadas.append(elemento[-1])
            elemento.pop(-1)
            # Agregamos el sesgo
            elemento.insert(0,1)

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
                salida = self.funcionLogistica(entradas[k], 1)
                salidas.append(salida)
                #print("COMPARANDO " + str(respuestaDeseadaActual) + "CON " + str(salida))

                deltaW = float(respuestaDeseadaActual) - float(salida)
                # Para calcular error cuadratico medio
                error = float(error) + float(deltaW)**2
                # Actualizo los Pesos
                for i in range(0, len(self.pesos)):
                    self.pesos[i] = float(self.pesos[i]) + (float(self.tasaAprendizaje) * deltaW * float(entradas[k][i]))

            # Calcular error cuadratico medio
            error = error/2

            # print(salidas)
            # print(respuestasDeseadas)
            #print(self.pesos)
        print("Usando una tasa de aprendizaje de " + str(self.tasaAprendizaje) + " y " + str(epoca) + " épocas, se logró el entrenamiento con un error de " + str(error) + ".")
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
        for i in range(0, len(datosArreglo)):
            respuesta = self.funcionLogistica(datosArreglo[i],1)
            if respuesta < 0.5:
                respuestas.append([0, respuesta])
            else:
                respuestas.append([1, respuesta])

        return respuestas

print("En Orden Aleatorio")
# Tasa 0.01
neuronaSigmoideLogistica = perceptronLMS(0.01, 2) 
aleatario = True
neuronaSigmoideLogistica.entrenar("classdata.csv", aleatario, 10000)
print("Los pesos finales son:")
print(neuronaSigmoideLogistica.pesos)
print("")
print("Clasificamos los nuevos datos y estan son las respuestas:")
respuestas = neuronaSigmoideLogistica.clasificar("classdataVal.csv")
for i in respuestas:
    print(i)
