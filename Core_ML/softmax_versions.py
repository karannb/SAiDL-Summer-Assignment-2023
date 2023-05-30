#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 16:15:47 2023

@author: karan_bania
"""


from __future__ import print_function
import numpy as np
import matplotlib as mpl
#mpl.use('TkAgg')
import matplotlib.pyplot as plt
from docs.classifiers.cnn import ConvolutionalNeuralNet
from docs.data_utils import get_CIFAR100_data
#from docs.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
#from layers import *
from docs.fast_layers import *
from docs.solver import Solver

#plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
#plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'

data = get_CIFAR100_data() #num_training = 10000

for k, v in data.items():
  print('%s: ' % k, v.shape)


#model = ConvolutionalNeuralNet(paddings = [3])
  
model = ConvolutionalNeuralNet(layers = 6, num_filters = [6, 8, 10, 12], filter_sizes = [3, 3, 3, 1], strides = [1, 1, 1, 1], 
                               pools = [2, 2, 2, 2],paddings = [1, 1, 1, 0], temperature = 0)

model_ = ConvolutionalNeuralNet(layers = 6, num_filters = [6, 8, 10, 12], filter_sizes = [3, 3, 3, 1], strides = [1, 1, 1, 1], 
                                pools = [2, 2, 2, 2],paddings = [1, 1, 1, 0], temperature = 0.00001)

solver = Solver(model, data, update_rule = 'adam', optim_config = {'learning_rate' : 3e-3},
                lr_decay = 0.95, num_epochs = 1, batch_size = 1000, print_every = 1)

solver_ = Solver(model_, data, update_rule = 'adam', optim_config = {'learning_rate' : 3e-3},
                lr_decay = 0.95, num_epochs = 1, batch_size = 1000, print_every = 1)

print("GUMBEL SOFTMAX -")
solver_.train()

print("NORMAL SOFTMAX -")
solver.train()

acc_n = solver.check_accuracy(data['X_test'], data['y_test'], batch_size = 1000)
acc_g = solver_.check_accuracy(data['X_test'], data['y_test'], batch_size = 1000)

print(acc_n, acc_g)