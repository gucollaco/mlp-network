# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 10:56:26 2018

@author: gustavo.collaco
"""

import pandas as pd
import numpy as np

def activation(y):
    return 1 / (1 + np.exp(-y))

def activation_derivative(y):
    return np.exp(-y) / (1 + np.exp(-y))**2

def dataset(n_hidden_perceptron):
    # read csv file
    data = pd.read_csv("rede_mlp_iris.csv", header=None)
    
    # inputs
    values = data.iloc[:, :-1].values
    
    # expected outputs
    answers_factorized = pd.factorize(data[np.size(values, 1)])[0]
    answers = np.zeros(shape=[np.size(values, 0), np.size(values, 1)-1])
    
    count = 0
    for i in answers_factorized:
        for j  in np.unique(answers_factorized):
            answers[count][j] = 1 if (i == j) else 0
        count += 1
    
    
    weights_matrix = 0
    layers = len(n_hidden_perceptron)
    for i in range(layers+1):
        if i == 0:
            weights_matrix += np.size(values, 1) * n_hidden_perceptron[i]
            #print('a',weights_matrix)
        elif i == layers:
            weights_matrix += np.size(answers, 1) * n_hidden_perceptron[i-1]
            #print('b',weights_matrix)
        else:
            weights_matrix += n_hidden_perceptron[i] * n_hidden_perceptron[i-1]
            #print('c',weights_matrix)
    
    # weight
    weights = np.random.uniform(low=0.0, high=0.1, size=(weights_matrix, 1))
    
    # print matrices
    print(values)
    print(weights)
    print(answers)
    
    # returning inputs, weights and outputs
    return values, weights, answers

def train_network(inputs, n_hidden_perceptron, weights, answers, learn_rate):
    # matrix to keep the perceptron sum values
    p = 0
    for z in n_hidden_perceptron:
        p += z
    p += len(inputs[0])
    p += len(answers[0])
    perceptron_sums = np.zeros((p, 1))
    perceptron_sums_activated = np.zeros((p, 1))
    
    delta = np.zeros((p, 1))
    #delta = np.zeros((np.size(inputs, 0), len(answers[0])))
    
    # iterating through all inputs
    for i in range(len(inputs)):
        w_index = 0
        p_index = 0
        layers = len(n_hidden_perceptron)
        
        # get the output, and calculate the error
        for j in range(layers+1):
            p_sum = 0
            #print('layer', j)
            if j == 0:
                for k in range(n_hidden_perceptron[j]):
                    for l in range(len(inputs[i])):
                        p_sum += inputs[i][l] * weights[w_index][0]
                        w_index += 1
                    perceptron_sums[p_index] = p_sum
                    perceptron_sums_activated[p_index] = activation(p_sum)
                    delta[p_index] = 0
                    p_index += 1
                    #print('0', activation(p_sum))
                    #print('0', p_index)
                    p_sum = 0
            elif j == layers:
                print('input number ', i)
                for k in range(len(answers[i])):
                    y = n_hidden_perceptron[j-1]
                    for l in range(n_hidden_perceptron[j-1]):
                        p_sum += perceptron_sums[p_index-y][0] * weights[w_index][0]
                        y -= 1
                        w_index += 1
                    #w_index -= 1
                    perceptron_sums[p_index] = p_sum
                    perceptron_sums_activated[p_index] = activation(p_sum)
                    delta[p_index] = (answers[i][k] - activation(p_sum)) * activation_derivative(p_sum)
                    
                    w_index -= 4
                    for l in range(n_hidden_perceptron[j-1]):
                        weights[w_index] = weights[w_index] + learn_rate * perceptron_sums_activated[p_index] * delta[p_index]
                        w_index += 1
                    #weights[w_index-1][0] = delta[p_index]
                    p_index += 1
                    print('output', activation(p_sum))
                    #print('2', activation(p_sum))
                    #print('2', p_index)
                    print(weights)
                    p_sum = 0
            else:
                for k in range(n_hidden_perceptron[j]):
                    y = n_hidden_perceptron[j-1]
                    for l in range(n_hidden_perceptron[j-1]):
                        p_sum += perceptron_sums[p_index-y][0] * weights[w_index][0]
                        y -= 1
                        w_index += 1
                    perceptron_sums[p_index] = p_sum
                    perceptron_sums_activated[p_index] = activation(p_sum)
                    delta[p_index] = 0
                    p_index += 1
                    #print('1', activation(p_sum))
                    #print('1', p_index)
                    p_sum = 0
                #print('layer val', l)
             
        # recalculate the weights
        #w_index -= 1
        #p_index -= 1
        '''for m in range(layers):
            e_sum = 0
            if m == (layers-1):
                z = n_hidden_perceptron[m]
                v = n_hidden_perceptron[m] * len(answers[i])
                for k in range(n_hidden_perceptron[m]):
                    v = n_hidden_perceptron[m] * len(answers[i])
                    for l in range(len(answers[i])):
                        e_sum += delta[p_index-y][0] * weights[v-1][0]
                        y -= 1
                        v += n_hidden_perceptron[m]
                    #weights[w_index-len(answers[i])-z] = perceptron_sums_activated[p_index] * e_sum
                    z += 1
                    p_index -= 1
                    w_index -= len(answers[i])
                    #print('1', activation(p_sum))
                    #print('1', p_index)
                    p_sum = 0
            else:
                for k in range(n_hidden_perceptron[m]):
                    y = n_hidden_perceptron[m+1]
                    for l in range(n_hidden_perceptron[m+1]):
                        p_sum += perceptron_sums[p_index-y][0] * weights[w_index][0]
                        y -= 1
                        w_index += 1
                    perceptron_sums[p_index] = p_sum
                    perceptron_sums_activated[p_index] = activation(p_sum)
                    p_index += 1
                    #print('1', activation(p_sum))
                    #print('1', p_index)
                    p_sum = 0'''
                    
        #print(w_index)
        #print()

if __name__ == "__main__":
    n_hidden_perceptron=[4,4]
    learn_rate=0.1
    
    values, weights, answers = dataset(n_hidden_perceptron)
    train_network(values, n_hidden_perceptron, weights, answers, learn_rate)
    
    
    