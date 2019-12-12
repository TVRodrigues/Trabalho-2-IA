# -*- coding: utf-8 -*-
"""Perceptron.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1p5NxrHbALayqhw3AxWtfMAtpOgk74v6W
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt1

X = np.array([
    [-2,4,-1],
    [4,1,-1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1],

])

y = np.array([-1,-1,1,1,1])

def perceptron(X, Y):
    '''
    Treina o Perceptron e exibe a curva de erro de treinamento.
    
    :return: vetor de pesos sinápticos num mumpy array
    '''
    w = np.zeros(len(X[0]))
    eta = 1
    n = 30
    errors = []

    for t in range(n):
        total_error = 0
        for i, x in enumerate(X):
            if (np.dot(X[i], w)*Y[i]) <= 0:
                total_error += (np.dot(X[i], w)*Y[i])
                w = w + eta*X[i]*Y[i]
        errors.append(total_error*-1)
        
    plt.plot(errors)
    plt.xlabel('Épocas')
    plt.ylabel('Erro Total')
    
    return w
  
w = perceptron(X,y)
print(w)

for d, sample in enumerate(X):
    # Exibe dados da Classe -1
    if d < 2:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    # Exibe dados da Classe +1
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

# Insere dados de um conjunto de teste (2 exemplos)
plt.scatter(2,2, s=120, marker='_', linewidths=3, color='yellow')
plt.scatter(4,3, s=120, marker='+', linewidths=3, color='blue')

# Exibe o hiperplano de separação dos dados dado pelo perceptron()
x2=[w[0],w[1],-w[1],w[0]]
x3=[w[0],w[1],w[1],-w[0]]

x2x3 = np.array([x2,x3])
X,Y,U,V = zip(*x2x3)
ax = plt.gca()
ax.quiver(X,Y,U,V,scale=1, color='blue')