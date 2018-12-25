import matplotlib.pyplot as plt
import numpy as np
from math import exp, log, sqrt
import csv

# Funcion para la tasa de aprendizaje
def tasaAprendizaje(n, n0=0.1, t=1000): # n: numero de epocas
    return (n0)*(exp(-n/t))

# Funcion de vecindad
def vecindad(i,j,n, sigma0=10):
    def sigma(v):
        return sigma0*(exp(-v/(1000/log(sigma0))))

    distancia1 = sqrt( ((i[0]-j[0])**2) + ((i[1]-j[1])**2) )

    return exp( (-(distancia1**2)) / (2*(sigma(n)**2)) )


# Cargamos los datos:
def leerCSV(archivoDatos1, archivoDatos2):
    # Leemos las caracteristicas del archivo 1 y el nombrel del animal en el archivo 2
    with open(archivoDatos1, newline='') as csvfile:
        datos = csv.reader(csvfile, delimiter=',')
        caracteristicas = []
        for fila in datos:
            elementos = []
            for item in fila:
                elementos.append(float(item))
            caracteristicas.append(elementos)

    with open(archivoDatos2, newline='') as csvfile:
        datos = csv.reader(csvfile, delimiter=',')
        nombres = []
        for fila in datos:
            nombres.append(fila)

    entradas = []
    for i in range(0, len(caracteristicas)):
        entradas.append([caracteristicas[i], nombres[i][0]])

    return entradas

datos = leerCSV("caract_animal.csv", "animales.csv")

# Iniciamos la red tomando como pesos iniciales algunos puntos de los datos
SOM = []
cantColumnas = 10
cantFilas = 10
pesosAleatorios = np.random.randint(0,len(datos), size=cantColumnas*cantFilas)
for i in range(0,cantFilas):
    filaNeuronas = []
    for j in range(0,cantColumnas):
        x = []
        for y in range(0,len(datos[pesosAleatorios[(i*cantColumnas)+j]][0])):
            x.append(datos[pesosAleatorios[(i*cantColumnas)+j]][0][y])
        filaNeuronas.append(x) 
    SOM.append(filaNeuronas)


print("Topologia Inicial:")
for dato in range(0,len(datos)):
    # Determinamos la neurona ganadora para el dato
    neuronaGanadora = (0,0)
    distanciaGanadora = 1000000000
    # Recorremos todas las neuronas
    for i in range(0,cantFilas):
        for j in range(0,cantColumnas):
            distancia = 0
            for k in range(0,len(datos[dato][0])):
                distancia += (float(SOM[i][j][k]) - float(datos[dato][0][k]))**2
            distancia = sqrt(distancia)
            if distancia < distanciaGanadora:
                distanciaGanadora = distancia
                neuronaGanadora = (i,j)
    print("Neurona ganadora para el animal " +str(datos[dato][1]) + " es " + str(neuronaGanadora) + " con distancia " + str(distanciaGanadora))
    # print("Y caracteristicas " + str(datos[dato][0]))

# FASE DE ORDENAMIENTO
epocas = 1000
for epoca in range(0,epocas):
    if epoca % 10 == 0:
        print("Epoca: " + str(epoca))
    for dato in range(0,len(datos)):
        # Determinamos la neurona ganadora para el dato
        neuronaGanadora = (0,0)
        distanciaGanadora = 1000000000
        # Recorremos todas las neuronas
        for i in range(0,cantFilas):
            for j in range(0,cantColumnas):
                distancia = 0
                for k in range(0,len(datos[dato][0])):
                    distancia = distancia +  ( (SOM[i][j][k] - datos[dato][0][k])**2)
                distancia = sqrt(distancia)
                if distancia < distanciaGanadora:
                    distanciaGanadora = distancia
                    neuronaGanadora = (i,j)

        # Actualizamos los pesos
        for i in range(0,cantFilas):
            for j in range(0,cantColumnas):
                for k in range(0,len(datos[dato][0])): 
                    SOM[i][j][k] = SOM[i][j][k] + (tasaAprendizaje(epoca) * vecindad(neuronaGanadora, [i,j], epoca) * (datos[dato][0][k]-SOM[i][j][k]))
# /FASE DE ORDENAMIENTO

print("Topologia tras Ordenamiento:")
for dato in range(0,len(datos)):
    # Determinamos la neurona ganadora para el dato
    neuronaGanadora = (0,0)
    distanciaGanadora = 1000000000
    # Recorremos todas las neuronas
    for i in range(0,cantFilas):
        for j in range(0,cantColumnas):
            distancia = 0
            for k in range(0,len(datos[dato][0])):
                distancia += (float(SOM[i][j][k]) - float(datos[dato][0][k]))**2
            distancia = sqrt(distancia)
            if distancia < distanciaGanadora:
                distanciaGanadora = distancia
                neuronaGanadora = (i,j)
    print("Neurona ganadora para el animal " +str(datos[dato][1]) + " es " + str(neuronaGanadora) + " con distancia " + str(distanciaGanadora))
    # print("Y caracteristicas " + str(datos[dato][0]))


# FASE DE CONVERGENCIA
epocas = cantColumnas*cantColumnas*500
for epoca in range(0,epocas):
    if epoca % 10 == 0:
        print("Epoca: " + str(epoca))
    for dato in range(0,len(datos)):
        # Determinamos la neurona ganadora para el dato
        neuronaGanadora = (0,0)
        distanciaGanadora = 1000000000
        # Recorremos todas las neuronas
        for i in range(0,cantFilas):
            for j in range(0,cantColumnas):
                distancia = 0
                for k in range(0,len(datos[dato][0])):
                    distancia = distancia +  ( (SOM[i][j][k] - datos[dato][0][k])**2)
                distancia = sqrt(distancia)
                if distancia < distanciaGanadora:
                    distanciaGanadora = distancia
                    neuronaGanadora = (i,j)

        # Actualizamos los pesos
        for i in range(0,cantFilas):
            for j in range(0,cantColumnas):
                for k in range(0,len(datos[dato][0])): 
                    SOM[i][j][k] += tasaAprendizaje(epoca,n0=0.01)* vecindad(neuronaGanadora, [i,j], epoca, sigma0=2) *(datos[dato][0][k]-SOM[i][j][k])
# /FASE DE CONVERGENCIA 

print("Topologia Final:")
for dato in range(0,len(datos)):
    # Determinamos la neurona ganadora para el dato
    neuronaGanadora = (0,0)
    distanciaGanadora = 1000000000
    # Recorremos todas las neuronas
    for i in range(0,cantFilas):
        for j in range(0,cantColumnas):
            distancia = 0
            for k in range(0,len(datos[dato][0])):
                distancia += (float(SOM[i][j][k]) - float(datos[dato][0][k]))**2
            distancia = sqrt(distancia)
            if distancia < distanciaGanadora:
                distanciaGanadora = distancia
                neuronaGanadora = (i,j)
    print("Neurona ganadora para el animal " +str(datos[dato][1]) + " es " + str(neuronaGanadora) + " con distancia " + str(distanciaGanadora))
    # print("Y caracteristicas " + str(datos[dato][0]))

################### RESULTADOS

# Topologia Final:                                                                                                                                                                 
# Neurona ganadora para el animal Antilope  es (0, 3) con distancia 3.813111419805034e-06                                                                                          
# Neurona ganadora para el animal Perca      es (9, 0) con distancia 0.24974505661663665                                                                                            
# Neurona ganadora para el animal Oso      es (0, 0) con distancia 0.21271682859509142                                                                                            
# Neurona ganadora para el animal Jabali      es (0, 1) con distancia 1.371800074696004e-05                                                                                          
# Neurona ganadora para el animal Bufalo   es (0, 3) con distancia 3.813111419805034e-06                                                                                          
# Neurona ganadora para el animal Bagre   es (9, 0) con distancia 0.24974505661663665                                                                                            
# Neurona ganadora para el animal cheetah   es (0, 1) con distancia 1.371800074696004e-05                                                                                          
# Neurona ganadora para el animal Gallina   es (9, 9) con distancia 0.005471343965378313                                                                                           
# Neurona ganadora para el animal Almeja      es (9, 3) con distancia 0.14481794910636275                                                                                            
# Neurona ganadora para el animal Cangrejo      es (4, 9) con distancia 0.07314245492063512                                                                                            
# Neurona ganadora para el animal Cangrejo de rio  es (2, 9) con distancia 0.010697483298795163                                                                                           
# Neurona ganadora para el animal Cuervo      es (7, 9) con distancia 0.0258375839401051                                                                                             
# Neurona ganadora para el animal Venado      es (0, 3) con distancia 3.813111419805034e-06                                                                                          
# Neurona ganadora para el animal Delfin   es (7, 1) con distancia 0.4482922224145543                                                                                             
# Neurona ganadora para el animal Paloma      es (9, 9) con distancia 0.005471343965378313                                                                                           
# Neurona ganadora para el animal Pato      es (8, 9) con distancia 0.5202363579223458                                                                                             
# Neurona ganadora para el animal Elefante  es (0, 3) con distancia 3.813111419805034e-06                                                                                         
# Neurona ganadora para el animal flamingo  es (9, 6) con distancia 0.21918770137753807                                                                                            
# Neurona ganadora para el animal Pulga      es (2, 7) con distancia 0.007040550375397287                                                                                           
# Neurona ganadora para el animal Rana      es (4, 7) con distancia 0.5015341586476579                                                                                             
# Neurona ganadora para el animal Muercielago  es (4, 0) con distancia 0.025420589438782164                                                                                           
# Neurona ganadora para el animal Jirafa   es (0, 3) con distancia 3.813111419805034e-06                                                                                          
# Neurona ganadora para el animal Cabra      es (2, 3) con distancia 0.00043315632405358                                                                                            
# Neurona ganadora para el animal Gorila   es (5, 2) con distancia 0.07754034555401575                                                                                            
# Neurona ganadora para el animal Gaviota      es (6, 9) con distancia 0.1094836804511335                                                                                             
# Neurona ganadora para el animal hamster   es (3, 4) con distancia 0.07295227931717069                                                                                            
# Neurona ganadora para el animal Liebre      es (4, 4) con distancia 0.07266335148149325                                                                                            
# Neurona ganadora para el animal Halcon      es (7, 9) con distancia 0.0258375839401051                                                                                             
# Neurona ganadora para el animal Abeja  es (0, 6) con distancia 0.45002355872482086                                                                                            
# Neurona ganadora para el animal Mosca  es (1, 6) con distancia 0.02211654671976301                                                                                            
# Neurona ganadora para el animal kiwi      es (7, 5) con distancia 0.13554945043712574                                                                                            
# Neurona ganadora para el animal leopard   es (0, 1) con distancia 1.371800074696004e-05                                                                                          
# Neurona ganadora para el animal lion      es (0, 1) con distancia 1.371800074696004e-05                                                                                          
# Neurona ganadora para el animal Langosta   es (2, 9) con distancia 0.010697483298795163                                                                                           
# Neurona ganadora para el animal Polilla      es (1, 6) con distancia 0.02211654671976301                                                                                            
# Neurona ganadora para el animal Pulpo   es (0, 9) con distancia 1.2244538322101812                                                                                             
# Neurona ganadora para el animal Zarigueya   es (2, 0) con distancia 0.10383955259631121                                                                                            
# Neurona ganadora para el animal Avestruz   es (7, 6) con distancia 0.7072043304555712                                                                                             
# Neurona ganadora para el animal Perico  es (9, 9) con distancia 0.005471343965378313                                                                                           
# Neurona ganadora para el animal Pingunino   es (7, 6) con distancia 0.7073011199850027                                                                                             
# Neurona ganadora para el animal piranha   es (9, 0) con distancia 0.24974505661663665                                                                                            
# Neurona ganadora para el animal pony      es (2, 3) con distancia 0.00043315632405358                                                                                            
# Neurona ganadora para el animal puma      es (0, 1) con distancia 1.371800074696004e-05                                                                                          
# Neurona ganadora para el animal Gato  es (2, 1) con distancia 0.10237969177834479                                                                                            
# Neurona ganadora para el animal Mapache   es (0, 1) con distancia 1.371800074696004e-05                                                                                          
# Neurona ganadora para el animal Reno  es (2, 3) con distancia 0.00043315632405358                                                                                            
# Neurona ganadora para el animal scorpion  es (0, 9) con distancia 1.2250389787533713                                                                                             
# Neurona ganadora para el animal Caballo de Mar  es (9, 0) con distancia 0.750261566351913                                                                                              
# Neurona ganadora para el animal Foca      es (7, 0) con distancia 0.4733338478745387                                                                                             
# Neurona ganadora para el animal Focaion   es (5, 0) con distancia 0.24334118334253774                                                                                            
# Neurona ganadora para el animal Avispa Marina   es (9, 2) con distancia 0.15552444532526313                                                                                            
# Neurona ganadora para el animal Babosa      es (9, 4) con distancia 0.016286319662358672                                                                                           
# Neurona ganadora para el animal GOrrion   es (8, 9) con distancia 0.4813338012513206                                                                                             
# Neurona ganadora para el animal Ardilla  es (6, 3) con distancia 0.069740865812344                                                                                              
# Neurona ganadora para el animal Estrella de Mar  es (3, 9) con distancia 0.04538894499196082                                                                                            
# Neurona ganadora para el animal Mantarraya  es (9, 1) con distancia 0.23468254404918848                                                                                            
# Neurona ganadora para el animal Cisne      es (9, 7) con distancia 0.2441637768370228                                                                                             
# Neurona ganadora para el animal termite   es (2, 7) con distancia 0.007040550375397287                                                                                           
# Neurona ganadora para el animal Sapo      es (4, 7) con distancia 0.49854787532301187                                                                                            
# Neurona ganadora para el animal Tortuga  es (5, 6) con distancia 0.1023882190085004                                                                                             
# Neurona ganadora para el animal Atun      es (8, 0) con distancia 0.17063210341873397                                                                                            
# Neurona ganadora para el animal vampire   es (4, 0) con distancia 0.025420589438782164                                                                                          
# Neurona ganadora para el animal Buitre   es (8, 7) con distancia 0.13028460531118596                                                                                            
# Neurona ganadora para el animal Avispa      es (0, 7) con distancia 0.4265593544937843                                                                                             
# Neurona ganadora para el animal Lobo      es (0, 1) con distancia 1.371800074696004e-05                                                                                          
# Neurona ganadora para el animal Gusano es (9, 4) con distancia 0.016286319662358672  
# Topologia tras Ordenamiento:                                                                                                                                                     
# Neurona ganadora para el animal Antilope  es (0, 3) con distancia 0.11368765276112708                                                                                            
# Neurona ganadora para el animal Perca      es (9, 0) con distancia 0.41131421830755255                                                                                            
# Neurona ganadora para el animal Oso      es (0, 0) con distancia 0.9126472677253918                                                                                             
# Neurona ganadora para el animal Jabali      es (0, 0) con distancia 0.0995875938039657                                                                                             
# Neurona ganadora para el animal Bufalo   es (0, 3) con distancia 0.11368765276112708                                                                                            
# Neurona ganadora para el animal Bagre   es (9, 0) con distancia 0.41131421830755255                                                                                            
# Neurona ganadora para el animal cheetah   es (0, 0) con distancia 0.0995875938039657                                                                                             
# Neurona ganadora para el animal Gallina   es (9, 9) con distancia 0.3470914519543795                                                                                             
# Neurona ganadora para el animal Almeja      es (9, 3) con distancia 0.7218355119870428                                                                                             
# Neurona ganadora para el animal Cangrejo      es (4, 9) con distancia 0.49715694312478753                                                                                            
# Neurona ganadora para el animal Cangrejo de rio  es (2, 9) con distancia 0.25642543327129197                                                                                            
# Neurona ganadora para el animal Cuervo      es (7, 9) con distancia 0.5554103703566223                                                                                             
# Neurona ganadora para el animal Venado      es (0, 3) con distancia 0.11368765276112708                                                                                            
# Neurona ganadora para el animal Delfin   es (7, 0) con distancia 0.773859364421526                                                                                              
# Neurona ganadora para el animal Paloma      es (9, 9) con distancia 0.3470914519543795                                                                                             
# Neurona ganadora para el animal Pato      es (7, 9) con distancia 0.8790597561321951                                                                                             
# Neurona ganadora para el animal Elefante  es (0, 3) con distancia 0.11368765276112708                                                                                            
# Neurona ganadora para el animal flamingo  es (9, 7) con distancia 0.4615660435026639                                                                                             
# Neurona ganadora para el animal Pulga      es (2, 7) con distancia 0.4259094538413052                                                                                             
# Neurona ganadora para el animal Rana      es (4, 7) con distancia 0.7105429651241113                                                                                             
# Neurona ganadora para el animal Muercielago  es (4, 0) con distancia 0.4837568348519431                                                                                             
# Neurona ganadora para el animal Jirafa   es (0, 3) con distancia 0.11368765276112708                                                                                            
# Neurona ganadora para el animal Cabra      es (2, 3) con distancia 0.252885647853523                                                                                              
# Neurona ganadora para el animal Gorila   es (5, 1) con distancia 0.8850783837434101                                                                                             
# Neurona ganadora para el animal Gaviota      es (6, 9) con distancia 0.5421103525691268                                                                                             
# Neurona ganadora para el animal hamster   es (3, 3) con distancia 0.6058823370843247                                                                                             
# Neurona ganadora para el animal Liebre      es (4, 3) con distancia 0.5874286317585762                                                                                            
# Neurona ganadora para el animal Halcon      es (7, 9) con distancia 0.5554103703566223                                                                                             
# Neurona ganadora para el animal Abeja  es (0, 6) con distancia 0.9437864999038892                                                                                             
# Neurona ganadora para el animal Mosca  es (0, 6) con distancia 0.5302799235285937                                                                                             
# Neurona ganadora para el animal kiwi      es (7, 5) con distancia 0.6833466928273133                                                                                             
# Neurona ganadora para el animal leopard   es (0, 0) con distancia 0.0995875938039657                                                                                             
# Neurona ganadora para el animal lion      es (0, 0) con distancia 0.0995875938039657                                                                                             
# Neurona ganadora para el animal Langosta   es (2, 9) con distancia 0.25642543327129197                                                                                            
# Neurona ganadora para el animal Polilla      es (0, 6) con distancia 0.5302799235285937                                                                                             
# Neurona ganadora para el animal Pulpo   es (0, 9) con distancia 1.205677430131079                                                                                              
# Neurona ganadora para el animal Zarigueya   es (2, 0) con distancia 0.6572549877119881                                                                                             
# Neurona ganadora para el animal Avestruz   es (8, 6) con distancia 0.6749998269243048                                                                                             
# Neurona ganadora para el animal Perico  es (9, 9) con distancia 0.3470914519543795                                                                                             
# Neurona ganadora para el animal Pingunino   es (7, 6) con distancia 0.7823767895069031                                                                                             
# Neurona ganadora para el animal piranha   es (9, 0) con distancia 0.41131421830755255                                                                                            
# Neurona ganadora para el animal pony      es (2, 3) con distancia 0.252885647853523                                                                                              
# Neurona ganadora para el animal professor es (0, 0) con distancia 0.0995875938039657                                                                                             
# Neurona ganadora para el animal puma      es (0, 0) con distancia 0.0995875938039657                                                                                             
# Neurona ganadora para el animal Gato  es (1, 1) con distancia 0.7731637523333942                                                                                             
# Neurona ganadora para el animal Mapache   es (0, 0) con distancia 0.0995875938039657                                                                                             
# Neurona ganadora para el animal Reno  es (2, 3) con distancia 0.252885647853523                                                                                              
# Neurona ganadora para el animal scorpion  es (0, 9) con distancia 1.3532325630922282                                                                                             
# Neurona ganadora para el animal Caballo de Mar  es (9, 0) con distancia 0.9051594408945073                                                                                             
# Neurona ganadora para el animal Foca      es (7, 0) con distancia 1.0239791725056084                                                                                             
# Neurona ganadora para el animal Focaion   es (6, 2) con distancia 0.6528075323075115                                                                                             
# Neurona ganadora para el animal Avispa Marina   es (9, 3) con distancia 1.0987597674112883                                                                                             
# Neurona ganadora para el animal Babosa      es (9, 4) con distancia 0.5958127961788102                                                                                             
# Neurona ganadora para el animal GOrrion   es (8, 9) con distancia 0.483302143048806                                                                                              
# Neurona ganadora para el animal Ardilla  es (4, 0) con distancia 0.5869133071740101                                                                                             
# Neurona ganadora para el animal Estrella de Mar  es (3, 9) con distancia 0.2763841346317205                                                                                             
# Neurona ganadora para el animal Mantarraya  es (9, 1) con distancia 0.9955157435455293                                                                                             
# Neurona ganadora para el animal Cisne      es (9, 7) con distancia 0.7104797473356466                                                                                             
# Neurona ganadora para el animal termite   es (2, 7) con distancia 0.4259094538413052                                                                                             
# Neurona ganadora para el animal Sapo      es (4, 6) con distancia 0.6935212810057335                                                                                             
# Neurona ganadora para el animal Tortuga  es (4, 5) con distancia 0.9053751966274723                                                                                             
# Neurona ganadora para el animal Atun      es (9, 1) con distancia 0.6709533926532615                                                                                             
# Neurona ganadora para el animal vampire   es (4, 0) con distancia 0.4837568348519431                                                                                             
# Neurona ganadora para el animal Buitre   es (7, 7) con distancia 0.6341823066743029                                                                                             
# Neurona ganadora para el animal Avispa      es (0, 6) con distancia 0.5792499025968934                                                                                             
# Neurona ganadora para el animal Lobo      es (0, 0) con distancia 0.0995875938039657                                                                                             
# Neurona ganadora para el animal Gusano es (9, 4) con distancia 0.5958127961788102  
# Topologia Inicial:                                                                                                                                                               
# Neurona ganadora para el animal Antilope  es (0, 2) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Perca      es (1, 0) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Oso      es (7, 2) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Jabali      es (2, 1) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Bufalo   es (0, 2) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Bagre   es (1, 0) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal cheetah   es (2, 1) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Gallina   es (6, 2) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Almeja      es (0, 0) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Cangrejo      es (2, 8) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Cangrejo de rio  es (3, 4) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Cuervo      es (6, 1) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Venado      es (0, 2) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Delfin   es (3, 1) con distancia 1.4142135623730951                                                                                             
# Neurona ganadora para el animal Paloma      es (6, 2) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Pato      es (1, 4) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Elefante  es (0, 2) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal flamingo  es (1, 2) con distancia 0.0                                                                                                           
# Neurona ganadora para el animal Pulga      es (0, 7) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Rana      es (0, 4) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Muercielago  es (1, 5) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Jirafa   es (0, 2) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Cabra      es (0, 8) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Gorila   es (1, 3) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Gaviota      es (1, 4) con distancia 1.0                                                                                                            
# Neurona ganadora para el animal hamster   es (3, 2) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Liebre      es (0, 2) con distancia 1.0                                                                                                            
# Neurona ganadora para el animal Halcon      es (6, 1) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Abeja  es (0, 3) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Mosca  es (2, 2) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal kiwi      es (7, 3) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal leopard   es (2, 1) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal lion      es (2, 1) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Langosta   es (3, 4) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Polilla      es (2, 2) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Pulpo   es (5, 3) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Zarigueya   es (4, 5) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Avestruz   es (0, 1) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Perico  es (6, 2) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Pingunino   es (0, 1) con distancia 1.4142135623730951                                                                                             
# Neurona ganadora para el animal piranha   es (1, 0) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal pony      es (0, 8) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal professor es (2, 1) con distancia 0.0      
# Neurona ganadora para el animal puma      es (2, 1) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Gato  es (0, 8) con distancia 1.0                                                                                                            
# Neurona ganadora para el animal Mapache   es (2, 1) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Reno  es (0, 8) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal scorpion  es (8, 5) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Caballo de Mar  es (4, 7) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Foca      es (3, 1) con distancia 0.0                                                                                                           
# Neurona ganadora para el animal Focaion   es (5, 1) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Avispa Marina   es (4, 1) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Babosa      es (0, 6) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal GOrrion   es (8, 8) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Ardilla  es (3, 5) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Estrella de Mar  es (9, 5) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Mantarraya  es (7, 6) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Cisne      es (6, 5) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal termite   es (0, 7) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Sapo      es (0, 4) con distancia 1.0                                                                                                            
# Neurona ganadora para el animal Tortuga  es (1, 9) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Atun      es (1, 7) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal vampire   es (1, 5) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Buitre   es (6, 9) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Avispa      es (0, 5) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Lobo      es (2, 1) con distancia 0.0                                                                                                            
# Neurona ganadora para el animal Gusano es (0, 6) con distancia 0.0   
