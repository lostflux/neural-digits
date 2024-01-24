#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Useful functions for processing / visualizing data
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset 

def load_data():
  """
  Load MNIST dataset
  """
  
  dataset = load_dataset("mnist")

  # split into train and test
  train = dataset["train"]
  test = dataset["test"]

  x_train = np.array([np.array(x["image"]) for x in train])
  x_train = x_train.reshape(x_train.shape[0], -1)
  
  y_train = np.array([x["label"] for x in train])
  y_train = pd.get_dummies(y_train, dtype=float).values
  
  x_test = np.array([np.array(x["image"]) for x in test])
  x_test = x_test.reshape(x_test.shape[0], -1)
  
  y_test = np.array([x["label"] for x in test])
  y_test = pd.get_dummies(y_test, dtype=float).values
  
  return {
    "train": {
      "images": x_train,
      "labels":  y_train
      },
    "test": {
      "images": x_test,
      "labels": y_test
    }
  }


def visualize(x, y_true, /, y_pred=None, rows=2, cols=5, split="train"):
  """
  Visualize images and labels / predictions
  """
  
  images = np.reshape(x, (-1, 28, 28))  
  
  # get values from one-hot encoded vectors
  if y_true.ndim == 2:
    y_true = np.argmax(y_true, axis=1)

  fig, axes = plt.subplots(rows, cols, figsize=(1.5*cols,2*rows))
  
  for pos in range(rows * cols):
    
    ax = axes[pos//cols, pos%cols]
    
    ax.imshow(images[pos], cmap="gray")
    
    if split == "train":
      ax.set_title(f"Label: {y_true[pos]}")
    elif split == "test":
      col = "green"
      if y_predicted[pos] != y_true[pos]:
        col = "red"
      ax.set_title(f"Label: {y_true[pos]} vs {y_predicted[pos]}", color = col)
          
  plt.tight_layout()
  plt.show()
