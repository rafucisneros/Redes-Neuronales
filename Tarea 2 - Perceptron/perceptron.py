import csv
from random import randint

class Perceptron:
    def __init__(self, tasaAprendizaje, cantidadParametros):
        self.tasaAprendizaje=tasaAprendizaje
        self.cantidadParametros=cantidadParametros
        self.pesos=[]
        for i in range(0,cantidadParametros+1):
            self.pesos.append(0)

    def clasificarFuncionSigno(self, elemento):
        salida = self.pesos[0]
        for i in range(1,self.cantidadParametros+1):
            salida = salida + (float(elemento[i]) * float(self.pesos[i]))
        if salida < 0:
            return -1
        else:
            return 1

    def entrenar(self, archivoDatos, aleatoriamente):
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
            # Sacamos la respuesta del vector X(n)
            if elemento[-1] == "Iris-setosa":
                respuestasDeseadas.append(1)
            else:
                respuestasDeseadas.append(-1)
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
            salidas = []
            for k in range(0,len(entradas)):
                respuestaDeseadaActual = respuestasDeseadas[k]
                #Funcion Signo
                salida = self.clasificarFuncionSigno(entradas[k])
                #/Funcion Signo
                #print("COMPARANDO " + str(respuestaDeseadaActual) + "CON " + str(salida))
                if salida != respuestaDeseadaActual:
                    errores = errores + 1

                # Actualizo los Pesos
                for i in range(0, len(self.pesos)):
                    self.pesos[i] = float(self.pesos[i]) + (float(self.tasaAprendizaje) * (respuestaDeseadaActual - salida) * float(entradas[k][i]))
                k = k + 1
            # print(salidas)
            # print(respuestasDeseadas)
            #print("Tenemos " + str(errores) + " instancias mal clasificadas.")
            #print(self.pesos)
        print("Usando una tasa de aprendizaje de " + str(self.tasaAprendizaje) + ", se finalizó el entrenamiento en " + str(epoca) + " epocas.")
        return epoca

print("En Orden Aleatorio")
# Tasa 0.01
perceptronIris = Perceptron(0.01, 4)
aleatario = True
perceptronIris.entrenar("datos.csv", aleatario)
print("Los pesos finales son:")
print(perceptronIris.pesos)
print("")

# Tasa 0.1
perceptronIris = Perceptron(0.1, 4)
aleatario = True
perceptronIris.entrenar("datos.csv", aleatario)
print("Los pesos finales son:")
print(perceptronIris.pesos)
print("")


# Tasa 0.0000000005
perceptronIris = Perceptron(0.0000000005, 4)
aleatario = True
perceptronIris.entrenar("datos.csv", aleatario)
print("Los pesos finales son:")
print(perceptronIris.pesos)
print("")

print("Sin orden Aleatorio")
# Tasa 0.01
perceptronIris = Perceptron(0.01, 4)
aleatario = False
perceptronIris.entrenar("datos.csv", aleatario)
print("Los pesos finales son:")
print(perceptronIris.pesos)
print("")

# Tasa 0.1
perceptronIris = Perceptron(0.1, 4)
aleatario = False
perceptronIris.entrenar("datos.csv", aleatario)
print("Los pesos finales son:")
print(perceptronIris.pesos)
print("")


# Tasa 0.0000000005
perceptronIris = Perceptron(0.0000000005, 4)
aleatario = False
perceptronIris.entrenar("datos.csv", aleatario)
print("Los pesos finales son:")
print(perceptronIris.pesos)
print("")


print("")
print("")
print("")