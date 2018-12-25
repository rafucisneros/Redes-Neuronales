import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def descensoGradiente(pesos, tasaAprendizaje):
    def funcionCosto(const, pesos):
    	return  ( ((const**2)/2) - (0.8182 * pesos[0] ) - (0.354 * pesos[1]) + 
                 ( (pesos[0]**2)/2 ) + (0.8182 * pesos[0] * pesos[1]) + (
                 (pesos[1]**2)/2) 
                )

    def derivadas(i, pesos):
        if i == 0:
            return (-0.8182 + pesos[0] + (0.8182 * pesos[1]))
        elif i == 1:
            return (-0.354 + (0.8182 * pesos[0]) + pesos[1])
     
    historialPesosX = []
    historialPesosY = []
    historialPesosZ = []
    epocas = 0
    while(True): 
        epocas += 1
        historialPesosX.append(pesos[0])
        historialPesosY.append(pesos[1])
        historialPesosZ.append(funcionCosto(1,pesos))
        viejosPesos = [pesos[0], pesos[1]]
        for i in range(0, len(pesos)):
            pesos[i] = viejosPesos[i] - (tasaAprendizaje * derivadas(i, viejosPesos))
        # print("Nuevo: " + str(funcionCosto(1, pesos)) + " contra Viejos: " + str(funcionCosto(1, viejosPesos)))
        if funcionCosto(1, pesos) > funcionCosto(1, viejosPesos):
            pesos = viejosPesos
            break
        # print("Gradientes Actuales: " + str(derivadas(0,pesos)) + " y " + str(derivadas(1,pesos)))
        # print("Pesos Actuales: " + str(pesos[0]) + " y " + str(pesos[1]))
        # print("Funcion Costo: " + str(funcionCosto(1, pesos)))
        # print()

    print("Finalizado en " + str(epocas) + " epocas.")
    print("Punto Mínimo = " + str(funcionCosto(const = 1, pesos=pesos)) + " con pesos " + str(pesos[0]) + " y " + 
          str(pesos[1]) + " y Gradientes: " + str(derivadas(0,pesos)) + " y " + str(derivadas(1,pesos)))
    print("Deberia dar (segun la parta 2a de la tarea) los pesos: " + str(1.6007) + str(-0.9556))

    # Grafica 2D
    plt.plot(historialPesosX,historialPesosY, 'r')
    plt.xlabel('Pesos W1')
    plt.ylabel('Pesos W2')
    plt.title("Pesos")
    plt.show()

    # Grafica 3D
    figura3D = plt.figure()
    ax3D = figura3D.gca(projection='3d')
    ax3D.plot(historialPesosX, historialPesosY, historialPesosZ, 'r')
    ax3D.set_xlabel('Pesos W1')
    ax3D.set_ylabel('Pesos W2')
    ax3D.set_zlabel('Funcion Costo')
    plt.title("Pesos con resultado funcion costo")
    # plt.show()

    # Graficar la funcion completa
    x = []
    y = []
    z = []
    for i in range(0,170):
    # for i in range(-600,600):
        i = i / 100
        for j in range(-100,100):
        # for j in range(-600,600):
            j = j / 100
            x.append(i)
            y.append(j)
            z.append(((1**2)/2) - (0.8182 * i) - (0.354 * j) + 
                    ((i**2)/2 ) + (0.8182 * i * j) + ((j**2)/2))

    x = np.array(x)
    y = np.array(y)
    z = (((1**2)/2) - (0.8182 * x) - (0.354 * y) + ((x**2)/2) + 
        (0.8182 * x * y) + ((y**2)/2))
    # fig = plt.figure()
    # ax3D = fig.gca(projection='3d')
    ax3D.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
    plt.show()

pesos = [1,1]
descensoGradiente(pesos, 0.3)
pesos = [1,1]
descensoGradiente(pesos, 1)

