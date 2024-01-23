#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = [ "Amittai Siavava" ]
__credits__ = [ "Amittai Siavava" ]

import numpy as np

def nan_to_zero(func):
  """
    Decorator to replace NaN values with 0
  """
  def wrapper(*args, **kwargs):
    result = func(*args, **kwargs)
    return np.nan_to_num(result)
  return wrapper

class Module:
  
  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)

class Sigmoid(Module):
  """
    Sigmoid Activation function
  """
  
  def __init__(self):
    self.out = None
    self.name = "sigmoid"
    self.learning_rate = 0.1
    
  @nan_to_zero
  def forward(self, x, is_train=False):
    """
      Forward Propagation
    """
    result = 1 / (1 + np.exp(-x))
    
    if is_train:
      self.out = result
      
    return result

  @nan_to_zero
  def backward(self, error):
    """
      Backward Propagation
    """
    # if self.out is None:
    #   raise ValueError("No forward propagation found")
    # return dx * self.out * (1 - self.out)
    if self.out is None:
      raise ValueError("No forward propagation found")
    
    derivative = self.out * (1 - self.out)
    
    
    return error * derivative * learning_rate
    

class Tanh(Module):
  """
    Hyperbolic tangent activation function
  """
  
  def __init__(self):
    self.out = None
    self.name = "tanh"
    self.learning_rate = 0.1
    
  @nan_to_zero
  def forward(self, x, is_train=False):
    """
      Forward Propagation
    """
    result = np.tanh(x)
    
    if is_train:
      self.out = result
      
    return result

  @nan_to_zero
  def backward(self, error):
    """
      Backward Propagation
    """
    if self.out is None:
      raise ValueError("No forward propagation found")

    derivative = 1 - np.power(self.out, 2)
    
    return error * derivative * self.learning_rate

class ReLU(Module):
  """
    Rectified Linear Unit activation function
  """
  
  def __init__(self):
    self.out = None
    self.name = "relu"
    self.learning_rate = 0.1
    
  @nan_to_zero
  def forward(self, x, is_train=False):
    """
      Forward Propagation
    """
    result = np.maximum(0, x)
    
    if is_train:
      self.out = result
      
    return result

  @nan_to_zero
  def backward(self, error):
    """
      Backward Propagation
    """
    if self.out is None:
      raise ValueError("No forward propagation found")

    derivative = np.where(self.out <= 0, 0, 1)
    
    return error * derivative * self.learning_rate


class Softmax(Module):
  """
    Softmax activation function
  """
  
  def __init__(self):
    self.out = None
    self.name = "softmax"
    self.learning_rate = 0.1
    
  @nan_to_zero
  def forward(self, x, is_train=False):
    """
      Forward Propagation
    """
    exp_scores = np.exp(x)
    probs = exp_scores / np.sum(exp_scores)
    
    if is_train:
      self.out = probs
      
    return probs

  @nan_to_zero
  def backward(self, error):
    """
      Backward Propagation
    """
    if self.out is None:
      raise ValueError("No forward propagation found")

    derivative = self.out * (1 - self.out)
    
    return error * derivative * self.learning_rate
