#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple Convolutional Neural Network
for MNIST handwritten digit recognition
using only numpy and matplotlib
"""

__author__ = [ "Amittai Siavava" ]


import numpy as np
import matplotlib.pyplot as plt

from utils import Sigmoid, Tanh, ReLU, Softmax, Linear, CE, MSE, MAE

class DenseLayer:
  def __init__(self, neurons: int, activation: str):
    
    self.neurons = neurons

    # initialize weights and biases
    self.weights = None
    self.bias = np.random.randn(neurons)
    
    match activation:
      case "sigmoid":
        self.activation = Sigmoid()
        
      case "tanh":
        self.activation = Tanh()
        
      case "relu":
        self.activation = ReLU()
        
      case "softmax":
        self.activation = Softmax()
        
      case "linear":
        self.activation = Linear()
        
      case _:
        raise ValueError(f"Unknown activation function: {activation}")
      
  def initialize(self, input_shape: int):
    """
    Initialize weights
    """
    self.weights = np.random.randn(self.neurons, input_shape)
  
  def forward(self, inputs, is_train=False):
    """
    Single Layer Forward Propagation
    """
    
    if self.weights is None:
      raise ValueError("Weights not initialized. Call initialize() first.")
    
    # forward-propagate through activation function
    self.output = self.activation( (self.weights @ inputs) + self.bias, is_train)
    return self.output
    
    
    # return NotImplemented
  
  def backward(self, output_error, learning_rate):
    """
    Single Layer Backward Propagation
    """
    
    # backward-propagate through activation function
    activation_error = self.activation.backward(output_error)
    
    # backward-propagate through layer
    input_error = np.dot(self.weights.T, activation_error)
    
    # update weights and biases
    self.weights -= learning_rate * np.dot(activation_error, self.output.T)
    self.bias -= learning_rate * activation_error
    
    return input_error
    
    
    # return NotImplemented
  def __repr__(self):
    """
    Print Layer
    """
    return f"in: {self.weights.shape[1]}, neurons: {self.neurons}, activation: {self.activation.name}"
      
    

class ConvNet:
  """
    A simple CNN
  """
  
  def __init__(self):
    self.layers = []
    self.epochs = 0
    
  def add(self, layer):
    """
      Add layer to network
    """
    self.layers.append(layer)
    
  def initialize(self, input_shape):
    """
      Initialize weights
    """
    for layer in self.layers:
      layer.initialize(input_shape)
      input_shape = layer.neurons

  def forward(self, X, is_train=False):
    """
      Propagate input through network
      and return output
    """
    output = X
    for layer in self.layers:
      # print(f"before: {output.shape}")
      # print(f"{layer = }")
      output = layer.forward(output, is_train)
      # print(f"{output[:5] = }")
      # print(f"after: {output.shape}")
      
    # print(output.shape)
    return output
  
  def backward(self, output_error, learning_rate):
    """
      Propagate output_error backward
      through network
    """
    for layer in reversed(self.layers):
      output_error = layer.backward(output_error, learning_rate)
    return output_error
  
  def __call__(self, X):
    """
      Call the forward function
    """
    return self.forward(X)
  
  def __repr__(self):
    """
      Print network
    """
    return "\n".join([str(layer) for layer in self.layers])
  
  def get_random_batch(self, X, y, batch_size):
    """
      Get a random batch from the dataset
    """
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]
  
  def train(self, dataset, epochs, learning_rate, batch_size=128):
    """
      Train the network
    """
    
    X = dataset["train"]["images"]
    y = dataset["train"]["labels"]
    
    # each epoch trains on a random sequence of 1000 datapoints
    for epoch in range(self.epochs, self.epochs + epochs):
      loss = 0
      # pick random sample from dataset
      X_batch, y_batch = self.get_random_batch(X, y, batch_size)
      
      for x, y_true in zip(X_batch, y_batch):
        # forward propagation
        y_pred = self.forward(x, is_train=True)
        
        # calculate loss (cross-Entropy)
        loss += CE(y_true, y_pred)
        
        # backward propagation
        self.backward(y_pred - y_true, learning_rate)
        
      if epoch % 100 == 0:
        print(f"Epoch {epoch:5d} of {self.epochs + epochs:5d}: {loss/len(X_batch)}")
        
    self.epochs += epochs
