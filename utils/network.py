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

from utils import Sigmoid, Tanh, ReLU, Softmax

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
  
  def train(self, X, y, epochs, learning_rate):
    """
      Train the network
    """
    for epoch in range(epochs):
      error = 0
      for X_batch, y_batch in zip(X, y):
        output = self.forward(X_batch, is_train=True)
        error += np.square(output - y_batch).sum()
        self.backward(output - y_batch, learning_rate)
      print(f"Epoch {epoch}: Error {error/len(X)}")
