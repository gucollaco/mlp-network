# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 10:56:26 2018

@author: gustavo.collaco
"""

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# activation function (F(y))
def activation(y):
    return 1 / (1 + np.exp(-y))

# activation function (F'(y))
def activation_derivative(y):
    return np.exp(-y) / (1 + np.exp(-y))**2

# momentum function
def momentum(learning, alfa, weight_diff):
    return learning + alfa * weight_diff

# dataset preparation function
def dataset(layers):
    # read csv file
    data = pd.read_csv("iris_dataset.csv", header=None)

    # inputs
    values = data.iloc[:, :-1]
    values["bias"] = 1
    values = values.values

    # expected outputs / populating the output matrix which will have its dimension
    # based on the amount of 'unique' values X amount of inputs
    answers_factorized = pd.factorize(data[np.size(values,1)-1])[0]
    n_answers = len(set(answers_factorized))
    answers = np.zeros(shape=[np.size(values,0), n_answers])
    count = 0
    for i in answers_factorized:
        for j  in np.unique(answers_factorized): answers[count][j] = 1 if (i==j) else 0
        count += 1

    # weights matrix
    weights = []
    n_inputs = len(values[0])
    layers.append(n_answers)
    for i in range(len(layers)):
        if(i==0): aux = [np.array([random.uniform(0, 0.1) for i in range(n_inputs)]) for n in range(layers[i])]
        else: aux = [np.array([random.uniform(0, 0.1) for i in range(layers[i-1]+1)]) for n in range(layers[i])]
        aux = np.array(aux)
        weights.append(aux)
    weights = np.array(weights)

    # print matrices
    # print(values)
    # print(weights)
    # print(answers)

    # returning inputs, weights, outputs and layers
    return values, weights, answers, layers

# adjust the weights
def train_network(inputs, answers, weights, layers, learning_rate=0.1, momentum_term=0.3, n_epoch=100):
    # iterate through all epochs
    for epoch in range(n_epochs):
        n_inputs = len(inputs)
        n_layers = len(layers)
        # iterate through all inputs
        for i in range(n_inputs):
            # arrays to store the sum on each perceptron and the errors
            perceptron_sums = [[] for j in range(n_layers)]
            errors = [[] for j in range(n_layers)]
            # iterate through the layers
            # calculating the perceptron sums
            for k in range(n_layers):
                # first layer
                if(k==0):
                    for l in range(layers[k]+1):
                        if(l==layers[k]): perceptron_sums[l].append(1) # bias
                        else: perceptron_sums[k].append(np.dot(inputs[i], pesos[k][l]) # inputs from this epoch
                # other layers
                else:
                    for l in range(layers[k]+1):
                        if(l==layers[k] and k!=(n_layers-1)): perceptron_sums[l].append(1) # bias
                        else: perceptron_sums[k].append(np.dot(perceptron_sums[k-1], pesos[k][l]) # last layer's perceptron sum X weights

            # updating the weights
            for k in reversed(range(n_layers)):
                # output layer
                if(k==(n_layers-1)):
                    for l in range(layers[k]):

                # first hidden
                elif(k==0):
                    for l in range(layers[k]):

                # other hiddens
                else:
                    for l in range(layers[k]):


# test if the weights were well adjusted
def test_network(inputs, answers, weights):

# main function
if __name__ == "__main__":
    # two hidden layers with 4 perceptrons each
    n_hidden_perceptron = [4,4]

    # returning values, weights, answers from the dataset function
    values, weights, answers, layers = dataset(n_hidden_perceptron)

    # separating values between training and test
    training_set, test_set, training_tags, test_tags = train_test_split(values, answers, test_size=0.2, random_state=30)

    # train network to adjust the weights
    train_network(training_set, training_tags, weights, layers)
