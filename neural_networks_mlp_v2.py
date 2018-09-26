# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 10:56:26 2018

@author: gustavo.collaco
"""

import math
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
    return y * (1-y)
    #return math.exp(-y) / (1.0 + math.exp(-y))**2

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
    random.seed(30)
    for i in range(len(layers)):
        if(i==0): aux = [np.array([random.uniform(0, 0.1) for i in range(n_inputs)]) for n in range(layers[i])]
        else: aux = [np.array([random.uniform(0, 0.1) for i in range(layers[i-1]+1)]) for n in range(layers[i])]
        aux = np.array(aux)
        weights.append(aux)
    weights = np.array(weights)

    # print matrices
    print(values)
    print(weights)
    print(answers)

    # returning inputs, weights, outputs and layers
    return values, weights, answers, layers

# adjust the weights
def train_network(inputs, answers, inputs_test, answers_test, weights, layers, learning_rate=0.2, momentum_term=0.3, n_epochs=50, acceptable_error=0.1):
    n_inputs = len(inputs)
    n_inputs_test = len(inputs_test)
    n_layers = len(layers)
    
    epochs = []
    epoch = 0
    
    error_train = []
    error_test = []
    
    e = 1

    # iterate while error is too high
    while(e > acceptable_error):
    #for epoch in range(n_epochs):

        error_epoch = []

        # iterate through all inputs (training)
        for i in range(n_inputs):

            # arrays to store the sum on each perceptron (with activation) and the errors
            perceptron_sums_active = [[] for j in range(n_layers)]
            deltas = [[] for j in range(n_layers)]
            
            error_output_train = []

            # iterate through the layers
            # calculating the perceptron sums
            for k in range(n_layers):
                # first layer
                if(k==0):
                    # for each perceptron on the current layer
                    for l in range(layers[k]):
                        perceptron_sums_active[k].append(activation(np.dot(inputs[i], weights[k][l]))) # inputs from this epoch

                    perceptron_sums_active[k].append(1) # bias

                # other layers
                else:
                    # for each perceptron on the current layer
                    for l in range(layers[k]):
                        perceptron_sums_active[k].append(activation(np.dot(perceptron_sums_active[k-1], weights[k][l]))) # last layer's perceptron sum X weights

                    if(k!=(n_layers-1)): perceptron_sums_active[k].append(1) # bias
                    
            # updating the weights
            for k in reversed(range(n_layers)):
                # output layer
                if(k==(n_layers-1)):
                    # per perceptron
                    for l in range(layers[k]):
                        
                        error_output = answers[i][l] - perceptron_sums_active[k][l]
                        error_output_train.append(error_output)
                        
                        # append to deltas array
                        deltas[k].append(error_output * activation_derivative(perceptron_sums_active[k][l]))

                        # update weights matrix
                        #for m in range(len(weights[k][l])):
                        #    weights[k][l][m] = weights[k][l][m] + (learning_rate * perceptron_sums_active[k-1][m] * deltas[k][l])

                # first hidden
                elif(k==0):
                    for l in range(layers[k]):
                        err_sum = 0
                        for m in range(layers[k+1]):
                            err_sum += weights[k+1][m][l] * deltas[k+1][m]

                        # append to deltas array
                        deltas[k].append(activation_derivative(perceptron_sums_active[k][l]) * err_sum)

                        # update weights matrix
                        #for m in range(len(weights[k][l])):
                        #    weights[k][l][m] = weights[k][l][m] + (learning_rate * inputs[i][m] * deltas[k][l])

                # other hiddens
                else:
                    for l in range(layers[k]):
                        err_sum = 0
                        for m in range(layers[k+1]):
                            err_sum += weights[k+1][m][l] * deltas[k+1][m]

                        # append to deltas array
                        deltas[k].append(activation_derivative(perceptron_sums_active[k][l]) * err_sum)

                        # update weights matrix
                        #for m in range(len(weights[k][l])):
                        #    weights[k][l][m] = weights[k][l][m] + (learning_rate * perceptron_sums_active[k-1][m] * deltas[k][l])

            
            error_mean_sqr = np.mean([e**2 for e in error_output_train])
            
            # checks if needs to update the weights
            if(error_mean_sqr > acceptable_error):
                
                for k in reversed(range(n_layers)):
                    for l in range(layers[k]):
                        if(k == 0):
                            for m in range(len(weights[k][l])):
                                weights[k][l][m] = weights[k][l][m] + (learning_rate * inputs[i][m] * deltas[k][l])

                        else:
                            for m in range(len(weights[k][l])):
                                weights[k][l][m] = weights[k][l][m] + (learning_rate * perceptron_sums_active[k-1][m] * deltas[k][l]) # momentum                
                
            error_epoch.append(error_mean_sqr)
            
        e = np.mean(error_epoch)
        error_train.append(e)
        
        epoch += 1
        epochs.append(epoch)
        
        print('______________')
        print('epoch', epoch)
        print('epoch error', e)
        
        
        # keep the errors
        error_mean_sqr_test = []

        # iterate through all inputs (test)
        for i in range(n_inputs_test):
            
            # arrays to store the sum on each perceptron (with activation)
            perceptron_sums_active_test = [[] for j in range(n_layers)]

            for k in range(n_layers):   
                # first layer
                if(k == 0):
                    # for each perceptron on the current layer
                    for l in range(layers[k]):
                        perceptron_sums_active_test[k].append(activation(np.dot(inputs_test[i], weights[k][l])))
                        
                    perceptron_sums_active_test[k].append(1) # bias
                
                # other layers
                else:
                    # for each perceptron on the current layer
                    for l in range(layers[k]):
                        perceptron_sums_active_test[k].append(activation(np.dot(perceptron_sums_active_test[k-1], weights[k][l])))
                        
                    if(k!=(n_layers-1)): perceptron_sums_active_test[k].append(1) # bias
        
            aux = []
            for m in range(layers[n_layers-1]):
                # epoch error calculation
                aux.append(test_tags[i][m] - perceptron_sums_active_test[n_layers-1][m])

            error_mean_sqr_test.append(np.mean([e**2 for e in aux]))
        
        print('epoch error test', np.mean([e**2 for e in aux]))
        error_test.append(np.mean(error_mean_sqr_test))
        
    # plotting the values
    plt.plot(epochs, error_train, label='train')
    plt.plot(epochs, error_test, label='test')
    plt.xlabel("Epochs")
    plt.ylabel("Error epoch")
    plt.legend()
    plt.show()

# test if the weights were well adjusted
# def test_network(inputs, answers, weights):

# main function
if __name__ == "__main__":
    # two hidden layers with 4 perceptrons each
    n_hidden_perceptron = [4,4]

    # returning values, weights, answers from the dataset function
    values, weights, answers, layers = dataset(n_hidden_perceptron)

    # separating values between training and test
    training_set, test_set, training_tags, test_tags = train_test_split(values, answers, test_size=0.2, random_state=30)

    # train network to adjust the weights
    train_network(training_set, training_tags, test_set, test_tags, weights, layers)
