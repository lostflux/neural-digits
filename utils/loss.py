#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Loss functions.
"""

import numpy as np

def MSE(y_true, y_pred):
  """
    Mean Squared Error Loss
  """
  return np.mean((y_true - y_pred)**2)

def MAE(y_true, y_pred):
  """
    Mean Absolute Error Loss
  """
  return np.mean(np.abs(y_true - y_pred))

def CE(y_true, y_pred):
  """
    Cross Entropy Loss
  """
  return -np.sum(y_true * np.log(y_pred + 1e-15))
