# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 10:56:26 2018

@author: gustavo.collaco
"""

import copy
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
def momentum(alfa, weight_diff):
    return alfa * weight_diff

# accuracy function
def accuracy(output, expected_output):
    output = [np.argmax(i) for i in output]
    expected_output = [np.argmax(i) for i in expected_output]
    
    accuracy = [1 if output[i] == expected_output[i] else 0 for i in range(len(output))]
    accuracy = sum(accuracy)/float(len(accuracy))

    return accuracy

# dataset preparation function
def dataset(layers):
    # read csv file
    data = pd.read_csv("dataset_iris.csv", header=None)

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
    #print(values)
    print(weights)
    #print(answers)

    # returning inputs, weights, outputs and layers
    return values, weights, answers, layers

# adjust the weights
def train_network(inputs, answers, inputs_test, answers_test, weights, layers, learning_rate=0.2, momentum_term=0.7, n_epochs=50, acceptable_error=0.1):
    n_inputs = len(inputs)
    n_inputs_test = len(inputs_test)
    n_layers = len(layers)
    
    epochs = []
    epoch = 0
    
    error_train = []
    error_test = []
    accu = []
    
    e = 1
    
    # keep t-1 weight values
    weights_old = copy.deepcopy(weights)
    
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
                        
                        if(k==(n_layers-1)):
                            error_output = answers[i][l] - perceptron_sums_active[k][l]
                            error_output_train.append(error_output)
                        
                    if(k!=(n_layers-1)): perceptron_sums_active[k].append(1) # bias
                    
            # updating the weights
            for k in reversed(range(n_layers)):
                # output layer
                if(k==(n_layers-1)):
                    # per perceptron
                    for l in range(layers[k]):                        
                        # append to deltas array
                        deltas[k].append((answers[i][l] - perceptron_sums_active[k][l]) * activation_derivative(perceptron_sums_active[k][l]))

                # first hidden
                elif(k==0):
                    for l in range(layers[k]):
                        err_sum = 0
                        for m in range(layers[k+1]):
                            err_sum += weights[k+1][m][l] * deltas[k+1][m]

                        # append to deltas array
                        deltas[k].append(activation_derivative(perceptron_sums_active[k][l]) * err_sum)

                # other hiddens
                else:
                    for l in range(layers[k]):
                        err_sum = 0
                        for m in range(layers[k+1]):
                            err_sum += weights[k+1][m][l] * deltas[k+1][m]

                        # append to deltas array
                        deltas[k].append(activation_derivative(perceptron_sums_active[k][l]) * err_sum)
            
            error_mean_sqr = np.mean([e**2 for e in error_output_train])
            
            # checks if needs to update the weights
            if(error_mean_sqr > acceptable_error):
                
                for k in reversed(range(n_layers)):
                    for l in range(layers[k]):
                        if(k == 0):
                            for m in range(len(weights[k][l])):
                                weights_difference = weights[k][l][m] - weights_old[k][l][m]
                                print(weights_difference)
                                weights_old[k][l][m] = weights[k][l][m]
                                weights[k][l][m] = weights[k][l][m] + (learning_rate * inputs[i][m] * deltas[k][l]) + momentum(momentum_term, weights_difference)

                        else:
                            for m in range(len(weights[k][l])):
                                weights_difference = weights[k][l][m] - weights_old[k][l][m]
                                print(weights_difference)
                                weights_old[k][l][m] = weights[k][l][m]
                                weights[k][l][m] = weights[k][l][m] + (learning_rate * perceptron_sums_active[k-1][m] * deltas[k][l]) + momentum(momentum_term, weights_difference)            

            error_epoch.append(error_mean_sqr)
            
        e = np.mean(error_epoch)
        error_train.append(e)
        
        epoch += 1
        epochs.append(epoch)
        
    #    print('______________')
  #      print('epoch', epoch)
  #      print('epoch error', e)
        
        # keep the last layers' results
        results_test = []
        
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
            results_test.append(perceptron_sums_active_test[n_layers-1])
        
  #      print('epoch error test', np.mean([e**2 for e in aux]))
        error_test.append(np.mean(error_mean_sqr_test))
        accu.append(accuracy(results_test, answers_test))
        
        
    # plotting the values
    plt.subplot(2, 1, 1)
    plt.plot(epochs, error_train, label='train')
    plt.plot(epochs, error_test, label='test')
    plt.ylabel("Error")
    plt.legend()
    
    
    plt.subplot(2, 1, 2)
    plt.plot(epochs, accu, label='accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

    print('WEIGHTS', weights)
    print('WEIGHTS OLD', weights_old)
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
