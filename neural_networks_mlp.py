# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 10:56:26 2018

@author: gustavo.collaco
"""

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

# momentum
def momentum(learning, alfa, weight_diff):
    return learning + alfa * weight_diff

def split_test_validation(total):
    # sixty and eighty
    sixty_percent = int(round(len(total) * 0.6, 2))
    eighty_percent = int(round(len(total) * 0.8, 2))
    
    # shuffle array
    np.random.shuffle(total)
    
    # return splitted dataset
    return total[:sixty_percent], total[sixty_percent:eighty_percent], total[eighty_percent:] 

# dataset preparation function
def dataset(n_hidden_perceptron):
    # read csv file
    data = pd.read_csv("rede_mlp_iris.csv", header=None)

    # inputs
    values = data.iloc[:, :-1].values

    # expected outputs
    answers_factorized = pd.factorize(data[np.size(values, 1)])[0]
    answers = np.zeros(shape=[np.size(values, 0), np.size(values, 1)-1])

    # populating the output matrix which will have its dimension based on the amount of 'unique' values X amount of inputs
    count = 0
    for i in answers_factorized:
        for j  in np.unique(answers_factorized):
            answers[count][j] = 1 if (i == j) else 0
        count += 1

    # calculating the weight matriz size
    weights_matrix = 0
    layers = len(n_hidden_perceptron)
    for i in range(layers+1):
        if i == 0:
            # +1 because of the bias
            weights_matrix += (np.size(values, 1)+1) * n_hidden_perceptron[i]
        elif i == layers:
            # +1 because of the bias
            weights_matrix += np.size(answers, 1) * (n_hidden_perceptron[i-1]+1)
        else:
            # +1 because of the bias
            weights_matrix += n_hidden_perceptron[i] * (n_hidden_perceptron[i-1]+1)

    # weight matrix creation
    weights = np.random.uniform(low=0.0, high=0.1, size=(weights_matrix, 1))
    
    # print matrices
    print(values)
    print(weights)
    print(answers)

    # returning inputs, weights and outputs
    return values, weights, answers

# train network function
def train_network(inputs, validation_inputs, n_hidden_perceptron, weights, answers, learning_rate, min_error, momentum_term):
    # getting the amount of perceptrons and storing in 'p'
    p = 0
    for z in n_hidden_perceptron:
        p += z
    p += len(answers[0])

    # matrix to keep the perceptron sum values for each perceptron
    perceptron_sums = np.zeros((p, 1))
    perceptron_sums_activated = np.zeros((p, 1))

    # matrix to keep the delta values for each perceptron
    delta = np.zeros((p, 1))
    
    # acceptable error
    error = 1
    
    # index of epoch
    #epoch = 0
    
    # values to plot
    epochs = []
    error_per_input = []
    error_train = []
    error_test = []
    n_epochs = 50
    
    # while error still not good enough
    #while (error > min_error) :
    for epoch in range(n_epochs):
        
    # iterating through all inputs
        for i in range(len(inputs)):
            w_index = 0
            p_index = 0
            layers = len(n_hidden_perceptron)
    
            # get the output, and calculate the error
            for j in range(layers+1):
                p_sum = 0
    
                # checking if it is the first hidden layer
                if j == 0:
                    for k in range(n_hidden_perceptron[j]):
    
                        # iterate through the inputs layer
                        for l in range(len(inputs[i])+1):
                            # checking if it is a bias
                            if l == len(inputs[i]):
                                p_sum += 1 * weights[w_index][0]
                                w_index += 1
                            else:
                                p_sum += inputs[i][l] * weights[w_index][0]
                                w_index += 1
    
                        # storing p_sum
                        perceptron_sums[p_index] = p_sum
                        perceptron_sums_activated[p_index] = activation(p_sum)
                        delta[p_index] = 0
                        p_index += 1
    
                        # resetting p_sum
                        p_sum = 0
    
                # checking if it is the output layer
                elif j == layers:
                    #print('input number ', i)
                    for k in range(len(answers[i])):
                        y = n_hidden_perceptron[j-1]
    
                        # iterate through the last hidden layer
                        for l in range(n_hidden_perceptron[j-1]+1):
                            # checking if it is a bias
                            if l == n_hidden_perceptron[j-1]:
                                p_sum += 1 * weights[w_index][0]
                                w_index += 1
                            else:
                                p_sum += perceptron_sums[p_index-y][0] * weights[w_index][0]
                                y -= 1
                                w_index += 1
    
                        # storing p_sum
                        perceptron_sums[p_index] = p_sum
                        perceptron_sums_activated[p_index] = activation(p_sum)
    
                         # storing delta (on output layer first)
                        delta[p_index] = (answers[i][k] - activation(p_sum)) * activation_derivative(p_sum)
                        w_index -= n_hidden_perceptron[j-1]+1
    
                        # iterate through last hidden layer (updating weights)
                        for l in range(n_hidden_perceptron[j-1]+1):
                            weights[w_index] = weights[w_index] + learning_rate * perceptron_sums_activated[p_index] * delta[p_index]
                            #weights[w_index] = momentum(learning_rate * perceptron_sums_activated[p_index] * delta[p_index], momentum_term, weights[w_index] - weights[w_index-1])
    
                            #print('perceptron updated:', p_index, '/', (p-1),' (', n_hidden_perceptron[j-1]+1, ' weights) -> weight index:', w_index)
                            w_index += 1
    
                        p_index += 1
                        #print('output', activation(p_sum))
                        #print(weights)
    
                        # resetting p_sum
                        p_sum = 0
    
                # checking if it is a middle hidden layer (or last)
                else:
                    for k in range(n_hidden_perceptron[j]):
                        y = n_hidden_perceptron[j-1]
    
                        # iterate through the hidden layer before this one
                        for l in range(n_hidden_perceptron[j-1]+1):
                            # checking if it is a bias
                            if l == n_hidden_perceptron[j-1]:
                                p_sum += 1 * weights[w_index][0]
                                w_index += 1
                            else:
                                p_sum += perceptron_sums[p_index-y][0] * weights[w_index][0]
                                y -= 1
                                w_index += 1
    
                        # storing p_sum
                        perceptron_sums[p_index] = p_sum
                        perceptron_sums_activated[p_index] = activation(p_sum)
                        delta[p_index] = 0
                        p_index += 1
    
                        # resetting p_sum
                        p_sum = 0
                
                
            # ________________________________________________________________________________________________
    
            # recalculate the weights
            
            #print('w index', w_index)
            #print('p_index', p_index)
    
            # backpropagation / recalculate the weights
            for m in reversed(range(layers)):
                e_sum = 0
    
                # checking if it is the last hidden layer
                if m == (layers-1):
                    #print('_____________________LAST HIDDEN__________________________')
                    delta_index = p_index - len(answers[i]) - n_hidden_perceptron[m]
                    delta_index_update = p_index - len(answers[i]) - n_hidden_perceptron[m]
    
                    update_weight_index = w_index - ((n_hidden_perceptron[m]+1) * len(answers[i]) + (n_hidden_perceptron[m]) * (n_hidden_perceptron[m-1]+1))
                    #print('UP', update_weight_index, w_index)
                    count = 0
                    for k in range(n_hidden_perceptron[m]):
                        z = p_index - 1
                        w_index_restart = w_index + count
                        #print('--------------')
                        for l in range(len(answers[i])):
                            w_index_restart -= n_hidden_perceptron[m]+1
                            #print('index weight:', w_index_restart)
    
                            e_sum += weights[w_index_restart] * delta[z]
                            #print('perceptron:', z)
                            z -= 1
    
                        delta[delta_index] = activation_derivative(perceptron_sums[delta_index]) * e_sum
                        count += 1
                        delta_index += 1
                        #print(k+1, ' finished')
                        #print('--------------')
    
                        # update weights
                        for o in range(n_hidden_perceptron[m-1]+1):
                            weights[update_weight_index] = weights[update_weight_index] + learning_rate * perceptron_sums_activated[delta_index_update] * delta[delta_index_update]
                            #weights[update_weight_index] = momentum(learning_rate * perceptron_sums_activated[delta_index_update] * delta[delta_index_update], momentum_term, weights[update_weight_index] - weights[update_weight_index-1])
    
                            #print('weight index update:', update_weight_index)
                            update_weight_index += 1
                        #print('perceptron updated:', delta_index_update, '/', (p-1))
                        delta_index_update += 1
    
                # checking if it is the first hidden layer
                elif m == 0:
                    #print('____________________FIRST HIDDEN___________________________')
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
                        if(idx == 0): i_index = val * (len(inputs[i])+1)
    
                    count = 0
                    for k in range(n_hidden_perceptron[m]):
                        z = p_index - f
                        i_index_restart = i_index + count
                        #print('--------------')
                        for l in range(n_hidden_perceptron[m+1]):
                            #print('index weight:', i_index_restart)
    
                            e_sum += weights[i_index_restart] * delta[z]
                            #print('perceptron:', z)
                            z += 1
                            i_index_restart += n_hidden_perceptron[m]
    
    
                        delta[delta_index] = activation_derivative(perceptron_sums[delta_index]) * e_sum
                        count += 1
                        delta_index += 1
                        #print(k+1, ' finished')
                        #print('--------------')
    
                        # update weights
                        for o in range(len(inputs[i])+1):
                            weights[update_weight_index] = weights[update_weight_index] + learning_rate * perceptron_sums_activated[delta_index_update] * delta[delta_index_update]
                            #weights[update_weight_index] = momentum(learning_rate * perceptron_sums_activated[delta_index_update] * delta[delta_index_update], momentum_term, weights[update_weight_index] - weights[update_weight_index-1])
    
                            #print('weight index update:', update_weight_index)
                            update_weight_index += 1
                        #print('perceptron updated:', delta_index_update, '/', (p-1))
                        delta_index_update += 1
    
                # else, it is a middle hidden layer
                else:
                    print('_______________________MIDDLE HIDDENS________________________')
                    i_index = 0
                    i_index_b = 0
                    qty = 0
                    qty_perceptron_after = 0
                    for idx, val in enumerate(n_hidden_perceptron):
                        if(idx >= m):
                            if(idx == (layers-1)):
                                i_index += val * len(answers[i])
                                i_index_b += (val+1) * len(answers[i])
                                qty += len(answers[i]) + n_hidden_perceptron[idx]
                            else:
                                i_index += val * n_hidden_perceptron[idx]
                                i_index_b += (val+1) * n_hidden_perceptron[idx]
                                qty += n_hidden_perceptron[idx]
                        if(idx > m):
                            if(idx == (layers-1)):
                                qty_perceptron_after += len(answers[i]) + n_hidden_perceptron[idx]
                            else:
                                qty_perceptron_after += n_hidden_perceptron[idx]
    
                        #print(idx, val, qty, p-qty, i_index, weights.size,qty_perceptron_after, delta_index)
    
                    delta_index = p-qty
                    delta_index_update = p-qty
                    #update_weight_index = weights.size - i_index - ((n_hidden_perceptron[m-1]+1) * n_hidden_perceptron[m])
                    update_weight_index = weights.size - i_index_b - ((n_hidden_perceptron[m-1]+1) * n_hidden_perceptron[m])
                    count = 0
                    for k in range(n_hidden_perceptron[m]):
                        z = p_index - qty_perceptron_after
                        i_index_restart = (weights.size - i_index_b) + count
                        #print('--------------')
                        for l in range(n_hidden_perceptron[m+1]):
                            #print('index weight:', i_index_restart)
    
                            e_sum += weights[i_index_restart] * delta[z]
                            #print('perceptron:', z)
                            z += 1
                            i_index_restart += n_hidden_perceptron[m]
    
    
                        delta[delta_index] = activation_derivative(perceptron_sums[delta_index]) * e_sum
                        count += 1
                        delta_index += 1
                        print(k+1, ' finished')
                        print('--------------')
    
                        # update weights
                        for o in range(n_hidden_perceptron[m-1]+1):
                            weights[update_weight_index] = weights[update_weight_index] + learning_rate * perceptron_sums_activated[delta_index_update] * delta[delta_index_update]
                            #weights[update_weight_index] = momentum(learning_rate * perceptron_sums_activated[delta_index_update] * delta[delta_index_update], momentum_term, weights[update_weight_index] - weights[update_weight_index-1])
    
                            print('weight index update:', update_weight_index)
                            update_weight_index += 1
                        print('perceptron updated:', delta_index_update, '/', (p-1))
                        delta_index_update += 1
                        
                    
            
            error_per_input.append(sum(np.abs(delta[(-1)*len(answers[i]):]))/len(answers[i]))
            #print(sum(np.abs(delta[len(delta)-len(answers[i]):]))/len(answers[i]))
            #error.append(np.median(np.absolute(erros[n_camadas-1])))
        
        error = np.mean(error_per_input)
        error_train.append(error)
        
        e = epoch + 1
        epochs.append(e)
        
        print('ERROR', error)
        print('EPOCH', e)
            

    #    print()
    #print('ERROR', np.mean(np.abs(delta)))
    #print('DELTA', np.abs(delta))
    plt.plot(epochs, error_train, label='training')
    plt.show()

# main function
if __name__ == "__main__":
    # setting two hidden layers with 4 perceptrons each and the learning rate
    n_hidden_perceptron = [4,4]
    learning_rate = 0.1
    momentum_term = 0.3
    min_error = 0.01

    # returning values, weights, answers from the dataset function
    values, weights, answers = dataset(n_hidden_perceptron)

    # split values into categories
    training_set, test_set, training_tags, test_tags = train_test_split(values, answers, test_size=0.2, random_state=30)
    #training_set, validation_set, test_set = split_test_validation(values)

    # training the neural network with the parameters sent
    train_network(training_set, test_set, n_hidden_perceptron, weights, answers, learning_rate, min_error, momentum_term)
