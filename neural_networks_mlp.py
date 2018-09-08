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

def train_network(inputs, n_hidden_perceptron, weights, answers, learning_rate):
    # matrix to keep the perceptron sum values
    p = 0
    for z in n_hidden_perceptron:
        p += z
    #p += len(inputs[0])
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
                    
                    #w_index -= 4
                    w_index -= n_hidden_perceptron[j-1]
                    for l in range(n_hidden_perceptron[j-1]):
                        weights[w_index] = weights[w_index] + learning_rate * perceptron_sums_activated[p_index] * delta[p_index]
                        
                        print('perceptron updated:', p_index, '/10 (', n_hidden_perceptron[j-1], ' weights)')
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
        print('w index', w_index)
        print('p_index', p_index)
        
        for m in reversed(range(layers)):
            e_sum = 0
            
            if m == (layers-1):
                delta_index = p_index - len(answers[i]) - n_hidden_perceptron[m]
                delta_index_update = p_index - len(answers[i]) - n_hidden_perceptron[m]
                #print(update_index)
                update_weight_index = w_index - (n_hidden_perceptron[m] * len(answers[i]) + (n_hidden_perceptron[m]) * n_hidden_perceptron[m-1])
                print('UP', update_weight_index)
                count = 0
                for k in range(n_hidden_perceptron[m]):
                    z = p_index - 1
                    w_index_restart = w_index + count
                    print('--------------')
                    for l in range(len(answers[i])):
                        w_index_restart -= n_hidden_perceptron[m]
                        print('index weight:', w_index_restart)
                        
                        e_sum += weights[w_index_restart] * delta[z]
                        print('perceptron:', z)
                        z -= 1
                        
                    delta[delta_index] = activation_derivative(perceptron_sums[delta_index]) * e_sum
                    count += 1
                    delta_index += 1
                    print(k+1, ' finished')
                    print('--------------')
                        
                    #perceptron_sums[p_index] = p_sum
                    #perceptron_sums_activated[p_index] = activation(p_sum)
                    #delta[p_index] = (answers[i][k] - activation(p_sum)) * activation_derivative(p_sum)
                    #print(e_sum)
                    
                    #update weights
                    for o in range(n_hidden_perceptron[m-1]):
                        weights[update_weight_index] = weights[update_weight_index] + learning_rate * perceptron_sums_activated[delta_index_update] * delta[delta_index_update]
                        
                        print('weight index update:', update_weight_index)
                        update_weight_index += 1
                    print('perceptron updated:', delta_index_update, '/10')
                    delta_index_update += 1
                #print(weights)
            #w_index = 0
            #p_index = 0
            elif m == 0:
                delta_index = 0
                delta_index_update = 0
                update_weight_index = 0
                # calculate the index of the first perceptron to iterate
                f = 0
                for z in n_hidden_perceptron:
                    f += z
                f -= n_hidden_perceptron[m]
                f += len(answers[0])
                # calculate the amount of weights from the next layers (including the output), to find the correct index
                for idx, val in enumerate(n_hidden_perceptron):
                    if(idx == 0): i_index = val * len(inputs[i])
                    
                count = 0
                for k in range(n_hidden_perceptron[m]):                    
                    z = p_index - f
                    #print(z)
                    i_index_restart = i_index + count
                    print('--------------')
                    for l in range(n_hidden_perceptron[m+1]):
                        print('index weight:', i_index_restart)
                        
                        e_sum += weights[i_index_restart] * delta[z]
                        print('perceptron:', z)
                        z += 1
                        i_index_restart += n_hidden_perceptron[m]
                        
                        
                    delta[delta_index] = activation_derivative(perceptron_sums[delta_index]) * e_sum
                    count += 1
                    delta_index += 1
                    print(k+1, ' finished')
                    print('--------------')
                    
                    #update weights
                    for o in range(len(inputs[i])):
                        weights[update_weight_index] = weights[update_weight_index] + learning_rate * perceptron_sums_activated[delta_index_update] * delta[delta_index_update]
                        
                        print('weight index update:', update_weight_index)
                        update_weight_index += 1
                    print('perceptron updated:', delta_index_update, '/10')
                    delta_index_update += 1
            #else:
                        
         
        print('w index', w_index)
        print('p_index', p_index)       
        
        print(weights)
        #print()

if __name__ == "__main__":
    n_hidden_perceptron = [4,4]
    learning_rate = 0.1
    
    values, weights, answers = dataset(n_hidden_perceptron)
    train_network(values, n_hidden_perceptron, weights, answers, learning_rate)
    
    
    