import csv
from random import randint

class PerceptronCompetitivo:
    def __init__(self, tasaAprendizaje, cantidadParametros, cantidadClases):
        self.tasaAprendizaje=tasaAprendizaje
        self.cantidadClases = cantidadClases
        self.cantidadParametros=cantidadParametros
        self.pesos=[]
        for j in range(0,cantidadClases):
            pesosClaseJ=[]
            for i in range(0,cantidadParametros+1):
                pesosClaseJ.append(0)
            self.pesos.append(pesosClaseJ)


    def entrenar(self, archivoDatos, aleatoriamente):
        # Leer datos
        with open(archivoDatos, newline='') as csvfile:
            datos = csv.reader(csvfile, delimiter=',')
            datosArreglo = []
            for fila in datos:
                datosArreglo.append(fila)
        datosArreglo.pop(0)

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
            # Sacamos la respuesta del vector X(n)
            respuestasDeseadas.append(elemento[-1])
            elemento.pop(-1)
            # Agregamos el sesgo
            elemento.insert(0,1)

        # print("RESPUESTAS DESEADAS")
        # print(respuestasDeseadas)

        errores = 1
        epoca = 0
        while(errores>0):
            errores = 0
            epoca = epoca + 1
            # print("Epoca #" + str(epoca))
            for k in range(0,len(entradas)): # Para todas las entradas
                respuestaDeseadaActual = respuestasDeseadas[k]
                mejorClase = 1 # Cualquier Clase
                mejorPuntaje = float("-inf")
                # Calculamos puntaje para cada clase
                for j in range(0,self.cantidadClases):    # Para cada vector de pesos
                    claseActual = j+1
                    salida = self.pesos[j][0]
                    for z in range(1,self.cantidadParametros+1):
                        salida = salida + (float(entradas[k][z]) * float(self.pesos[j][z]))
                    if salida > mejorPuntaje:
                        mejorPuntaje = salida
                        mejorClase = claseActual
                # Si clasifico mal
                if int(mejorClase) != int(respuestaDeseadaActual):
                    errores = errores + 1
                    for p in range(0,self.cantidadParametros+1):
                    # Refuerzo la clase correcta
                        self.pesos[int(respuestaDeseadaActual)-1][p] = self.pesos[int(respuestaDeseadaActual)-1][p] + (float(self.tasaAprendizaje) * float(entradas[k][p]))
                    # Penalizo la clase que clasifique mal
                        self.pesos[int(mejorClase)-1][p] = self.pesos[int(mejorClase)-1][p] - (float(self.tasaAprendizaje) * float(entradas[k][p]))

                # print("COMPARANDO " + str(respuestaDeseadaActual) + " CON " + str(mejorClase) + " " + str(int(mejorClase) == int(respuestaDeseadaActual)))
                k = k + 1
            # print(respuestasDeseadas)
            # print("Tenemos " + str(errores) + " instancias mal clasificadas.")
            # print(self.pesos)
        print("Usando una tasa de aprendizaje de " + str(self.tasaAprendizaje) + ", se finalizó el entrenamiento en " + str(epoca) + " epocas.")
        return epoca

print("Usando Datos Aleatorios")
perceptron4D = PerceptronCompetitivo(0.1, 4, 4)
aleatorio = True
perceptron4D.entrenar("4D.csv", aleatorio)
print("Los pesos finales son:")
print(perceptron4D.pesos[3])
print("")

perceptron4D = PerceptronCompetitivo(0.01, 4, 4)
aleatorio = True
perceptron4D.entrenar("4D.csv", aleatorio)
print("Los pesos finales son:")
print(perceptron4D.pesos[3])
print("")

perceptron4D = PerceptronCompetitivo(0.000000005, 4, 4)
aleatorio = True
perceptron4D.entrenar("4D.csv", aleatorio)
print("Los pesos finales son:")
print(perceptron4D.pesos[3])
print("")

print("Usando Datos Ordenados")
perceptron4D = PerceptronCompetitivo(0.1, 4, 4)
aleatorio = False
perceptron4D.entrenar("4D.csv", aleatorio)
print("Los pesos finales son:")
print(perceptron4D.pesos[3])
print("")

perceptron4D = PerceptronCompetitivo(0.01, 4, 4)
aleatorio = False
perceptron4D.entrenar("4D.csv", aleatorio)
print("Los pesos finales son:")
print(perceptron4D.pesos[3])
print("")

perceptron4D = PerceptronCompetitivo(0.000000005, 4, 4)
aleatorio = False
perceptron4D.entrenar("4D.csv", aleatorio)
print("Los pesos finales son:")
print(perceptron4D.pesos[3])
print("")


