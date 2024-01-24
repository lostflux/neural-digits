#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Activation functions
"""

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
  
class Linear(Module):
  """
    Linear Activation function
  """
  
  def __init__(self):
    self.out = None
    self.name = "linear"
    
  @nan_to_zero
  def forward(self, x, is_train=False):
    """
      Forward Propagation
    """
    if is_train:
      self.out = x
      
    return x

  @nan_to_zero
  def backward(self, error):
    """
      Backward Propagation
    """
    if self.out is None:
      raise ValueError("No forward propagation found")
    
    derivative = np.ones_like(self.out)
    
    return error * derivative

class Sigmoid(Module):
  """
    Sigmoid Activation function
  """
  
  def __init__(self):
    self.out = None
    self.name = "sigmoid"
    
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
    if self.out is None:
      raise ValueError("No forward propagation found")
    
    derivative = self.out * (1 - self.out)
    
    
    return error * derivative
    

class Tanh(Module):
  """
    Hyperbolic tangent activation function
  """
  
  def __init__(self):
    self.out = None
    self.name = "tanh"
    
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
    
    return error * derivative

class ReLU(Module):
  """
    Rectified Linear Unit activation function
  """
  
  def __init__(self):
    self.out = None
    self.name = "relu"
    
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
    
    return error * derivative


class Softmax(Module):
  """
    Softmax activation function
  """
  
  def __init__(self):
    self.out = None
    self.name = "softmax"
    
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
    
    return error * derivative
